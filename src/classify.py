"""
Classify module for CloudServe Support System.

This is the SECOND stage of the pipeline: Ingest → **Classify** → Retrieve → Route → Generate → Validate

Purpose:
    Take an ingested ticket and predict:
    1. Intent — what the customer is asking about (22 categories)
    2. Urgency — how urgent it is (low / medium / high)
    3. Answerable from docs — can we answer this from our 29 documentation articles?
    4. Confidence — how sure is the model about each prediction?

Design decisions:
    1. We use the Groq LLM (via OpenAI-compatible API) for classification.
       Why LLM instead of a traditional ML classifier (e.g., sklearn)?
       → 22 intent categories with only ~23 samples each is too few for supervised ML.
         An LLM can generalize from the category names and descriptions alone (zero-shot).
    2. We ask the LLM to return structured JSON, then validate with Pydantic.
       Why? Same principle as ingest: fail fast at the boundary. If the LLM returns
       garbage JSON or an unknown intent, we catch it immediately.
    3. Confidence scores are critical for routing. A ticket classified with 0.95
       confidence gets different treatment than one at 0.55. The router uses this.
    4. must_not_auto_respond is derived deterministically from the intent, not
       predicted by the LLM. The four flagged intents (compliance_request,
       security_incident, feature_request, unclear_request) are ALWAYS escalated
       regardless of confidence. This is a safety guardrail, not a prediction.

Interview context:
    "Why not fine-tune a model for classification?"
    → Fine-tuning needs hundreds of labeled examples per class. We have ~23 per intent
      in 500 tickets. Zero-shot LLM classification is the right tool for low-data
      scenarios. If CloudServe scales to 50,000 tickets, fine-tuning becomes viable.
    "What if the LLM hallucinates an intent that doesn't exist?"
    → Pydantic validation catches it. The ClassificationResult model only accepts
      intents from the VALID_INTENTS set. Unknown intents raise ValidationError.
"""

import json
import logging
import re
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel, Field, field_validator, model_validator

from src.config import GROQ_API_KEY, MODEL_NAME, CONFIDENCE_THRESHOLD
from src.ingest import StandardTicket

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — the 22 valid intents and 4 must-not-auto-respond intents
# ---------------------------------------------------------------------------

VALID_INTENTS = {
    "account_access",
    "api_key_issue",
    "api_usage_question",
    "authentication_failure",
    "billing_query",
    "compliance_request",
    "configuration_help",
    "data_export",
    "data_residency",
    "database_issue",
    "deployment_failure",
    "feature_request",
    "integration_help",
    "onboarding",
    "performance_degradation",
    "quota_or_overage",
    "rate_limit",
    "rollback_request",
    "security_incident",
    "sso_configuration",
    "unclear_request",
    "webhook_issue",
}

# These four intents must NEVER receive an automated response.
# This is a hard business rule from the stakeholder interviews:
#   - compliance_request: legal/regulatory — must go to compliance team
#   - security_incident: potential breach — must go to security team
#   - feature_request: no doc exists to answer — must go to product team
#   - unclear_request: ambiguous — needs human judgment to interpret
MUST_NOT_AUTO_RESPOND_INTENTS = {
    "compliance_request",
    "security_incident",
    "feature_request",
    "unclear_request",
}

VALID_URGENCIES = {"low", "medium", "high"}


# ---------------------------------------------------------------------------
# Pydantic model for classification output — the CONTRACT
# ---------------------------------------------------------------------------

class ClassificationResult(BaseModel):
    """
    The output of the classify stage.

    Every field is validated. If the LLM returns something unexpected,
    this model raises immediately rather than letting bad data flow
    into the retriever or router.
    """
    intent: str
    intent_confidence: float = Field(ge=0.0, le=1.0)
    urgency: str
    urgency_confidence: float = Field(ge=0.0, le=1.0)
    answerable_from_docs: bool
    answerable_confidence: float = Field(ge=0.0, le=1.0)
    must_not_auto_respond: bool = False
    reasoning: str = ""

    @field_validator("intent")
    @classmethod
    def validate_intent(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in VALID_INTENTS:
            raise ValueError(
                f"Unknown intent '{v}'. Must be one of: {sorted(VALID_INTENTS)}"
            )
        return v

    @field_validator("urgency")
    @classmethod
    def validate_urgency(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in VALID_URGENCIES:
            raise ValueError(
                f"Unknown urgency '{v}'. Must be one of: {sorted(VALID_URGENCIES)}"
            )
        return v

    @model_validator(mode="after")
    def set_must_not_auto_respond(self):
        """
        Deterministically set must_not_auto_respond based on intent.

        This is NOT a prediction — it's a business rule. The LLM does not
        decide whether something is safe to auto-respond. The intent
        classification determines it, and these four intents are hardcoded
        as never-auto-respond.
        """
        self.must_not_auto_respond = self.intent in MUST_NOT_AUTO_RESPOND_INTENTS
        return self


# ---------------------------------------------------------------------------
# The classification prompt — this is what the LLM sees
# ---------------------------------------------------------------------------

CLASSIFICATION_PROMPT = """You are a support ticket classifier for CloudServe Solutions, a cloud infrastructure provider.

Analyze the following support ticket and classify it.

## Valid Intents (choose exactly one):
- account_access: Customer cannot access their account or needs permission changes
- api_key_issue: Problems creating, rotating, or using API keys
- api_usage_question: Questions about how to use CloudServe APIs
- authentication_failure: Login failures, credential issues, auth errors
- billing_query: Questions about invoices, charges, payment methods
- compliance_request: Requests driven by auditors, regulators, or compliance reviews — even if they involve exporting data or logs. If the REASON is an audit, compliance review, or regulatory requirement, this is compliance_request, not data_export
- configuration_help: Help configuring CloudServe services or settings
- data_export: Requests to export or download data for the customer's OWN operational use (not for auditors or compliance). Pure data export without a compliance/audit context
- data_residency: Questions about where data is stored geographically
- database_issue: Problems with managed database services
- deployment_failure: Application deployment errors or failures
- feature_request: Requesting new features or capabilities
- integration_help: Help connecting CloudServe with third-party services
- onboarding: New customer setup, getting started questions
- performance_degradation: Slow response times, high latency, throughput issues
- quota_or_overage: Resource limit questions, quota increases, overage charges
- rate_limit: API rate limiting issues, throttling
- rollback_request: Requesting to revert a deployment or configuration change
- security_incident: Suspected security breach, unauthorized access, vulnerabilities
- sso_configuration: Setting up or troubleshooting Single Sign-On
- unclear_request: The request is ambiguous or lacks enough detail to classify
- webhook_issue: Problems with webhook delivery, configuration, or payloads

## Urgency Levels:
- low: General questions, no service impact, can wait 24+ hours
- medium: Some inconvenience but workarounds exist, should be addressed within hours
- high: Service is down, security risk, or business-critical blocker needing immediate attention

## Answerable from Documentation:
CloudServe has 29 official documentation articles covering these specific topics:
- Authentication: login credential errors, MFA setup/recovery, SSO/SAML configuration, API key rotation and scopes
- Deployment: container health-check failures, rolling back releases, dependency resolution build failures, environment variables and secrets
- API: rate limits and quota tiers, pagination and large result sets, webhook delivery/retries/signature verification
- Performance: response latency diagnosis, autoscaling behaviour and instance limits, database connection pool exhaustion
- Billing: invoice and usage breakdown, changing plans and proration, usage limits/overage/spend controls
- Data management: data export and scheduled extracts, backups/retention/point-in-time restore, data residency and regional storage
- Security: suspected account compromise response, exposed secrets and credential rotation, audit logging and compliance evidence
- Account: team member roles and permissions, organisation setup and project structure
- Integration: CI pipeline connections, third-party monitoring and log forwarding
- Onboarding: first deployment walkthrough, migrating existing applications

## How to determine answerable_from_docs:
"Answerable from docs" means our documentation contains the information needed to answer the customer's core question. It does NOT mean "should auto-respond" — some answerable tickets still need human handling for other reasons.

**Always false (0% answerable):**
- feature_request: Docs never cover features that don't exist yet.
- unclear_request: If you can't tell what they're asking, docs can't answer it.

**Almost always true (>85% answerable) — set false only if the specific question falls outside doc coverage:**
- api_usage_question, data_export, rate_limit, onboarding, billing_query, quota_or_overage, sso_configuration

**Usually true (~65-80%) — decide based on the specific question:**
- account_access, api_key_issue, authentication_failure, compliance_request, configuration_help, data_residency, database_issue, deployment_failure, integration_help, performance_degradation, rollback_request, security_incident, webhook_issue

**The deciding factor for mixed intents:** Does a documented PROCEDURE or TROUBLESHOOTING GUIDE apply to this ticket's problem type, or is this a MYSTERY requiring investigation of the customer's specific environment?

Set answerable_from_docs = TRUE when:
- The ticket describes a KNOWN problem type that matches a documented troubleshooting topic (e.g., "dependency resolution build failures," "MFA recovery," "credential rotation after exposure")
- The customer asks how to perform a task covered by docs (e.g., "how do I roll back," "how do I export data," "how do I set up SSO")
- Even if the customer describes THEIR specific experience of the problem, if the docs have procedures for that TYPE of problem, it is answerable

Set answerable_from_docs = FALSE when:
- The problem is a MYSTERY — something unexpectedly broke or changed and the cause is unknown ("key stopped working without any changes on our side," "access not working and we can't figure out why")
- The issue requires INVESTIGATING the customer's specific account, infrastructure, or data to diagnose
- The customer cannot find expected data or logs in their environment (needs someone to look at their system)
- The request is for something the docs don't cover at all

Examples:
- "Builds are failing during dependency resolution" → true (docs cover dependency resolution build failure troubleshooting)
- "Our API key started returning 401 without any change on our side" → false (mystery — needs investigation of what happened to their key)
- "Need to rotate an exposed key without downtime" → true (docs cover exposed secrets and credential rotation procedures)
- "MFA code keeps getting rejected" → true (docs cover MFA setup/recovery troubleshooting)
- "Changed our shared account password and now automated jobs are failing" → false (account-specific configuration issue needing investigation)
- "How do I export access records for our auditor?" → true (docs cover data export and audit logging)
- "We need to demonstrate read access is logged but can't find the events" → false (needs investigation of their specific audit setup)

## Instructions:
1. Read the ticket carefully. Consider both subject and body.
2. Choose the single best matching intent.
3. Assess urgency based on business impact, not just the customer's tone.
4. Determine if existing documentation could answer this question.
5. Provide confidence scores (0.0 to 1.0) for each prediction.
6. Write brief reasoning explaining your classification.

Respond with ONLY valid JSON in this exact format:
{{
    "intent": "<intent_name>",
    "intent_confidence": <0.0-1.0>,
    "urgency": "<low|medium|high>",
    "urgency_confidence": <0.0-1.0>,
    "answerable_from_docs": <true|false>,
    "answerable_confidence": <0.0-1.0>,
    "reasoning": "<brief explanation>"
}}

## Ticket to classify:
{ticket_text}"""


# ---------------------------------------------------------------------------
# Groq client initialization
# ---------------------------------------------------------------------------

def get_groq_client() -> OpenAI:
    """
    Create an OpenAI-compatible client pointing to Groq's API.

    Why OpenAI client for Groq?
    → Groq's API is OpenAI-compatible. Using the OpenAI SDK means we can
      swap providers (OpenAI, Groq, Together, Ollama) by changing the
      base_url alone. This is a good engineering practice — don't couple
      your code to one vendor's SDK.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set. Check your .env file.")

    return OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        max_retries=0,  # Disable built-in retry; our _call_with_retry handles backoff
    )


# ---------------------------------------------------------------------------
# Helper: format retrieval context for the classification prompt
# ---------------------------------------------------------------------------

def _format_retrieval_context(retrieval_result) -> str:
    """
    Format retrieval results into a context block for the classifier.

    When retrieval results are available, the classifier sees which docs
    were actually found and their relevance scores. This lets it make an
    INFORMED decision about answerable_from_docs instead of guessing.

    The retrieval is local (sentence-transformers + ChromaDB) — free,
    instant, no API call. Running it before classification costs nothing
    but gives the classifier the data it needs.
    """
    if retrieval_result is None:
        return "(No retrieval results available — use your best judgment based on the topic list above.)"

    if not retrieval_result.chunks:
        return "## Retrieval Results:\nNo relevant documentation was found for this ticket. Set answerable_from_docs=false."

    lines = ["## Retrieval Results (from vector search of the actual documentation):"]
    for i, chunk in enumerate(retrieval_result.chunks):
        lines.append(
            f"  {i+1}. [{chunk.doc_id}] \"{chunk.title}\" — section: {chunk.section_name} "
            f"(relevance: {chunk.relevance_score:.2f})"
        )
    lines.append("")
    lines.append(
        "If the retrieved documents are relevant to the ticket's question "
        "(relevance ≥ 0.40), set answerable_from_docs=true. "
        "If the documents are unrelated or low-relevance, set answerable_from_docs=false."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core classification function
# ---------------------------------------------------------------------------

def classify_ticket(ticket: StandardTicket, client: Optional[OpenAI] = None) -> ClassificationResult:
    """
    Classify a single support ticket using the Groq LLM.

    This function:
    1. Builds the classification prompt with the ticket text
    2. Includes retrieval context (if available) so the classifier can
       make an INFORMED decision about answerable_from_docs instead of guessing
    3. Sends it to the LLM via Groq's API
    4. Parses the JSON response
    5. Validates it through the ClassificationResult Pydantic model
    6. Returns the validated classification

    Args:
        ticket: A StandardTicket from the ingest stage
        client: Optional pre-configured OpenAI client (for reuse / testing)
    Returns:
        ClassificationResult with intent, urgency, confidence scores, etc.

    Raises:
        ValueError: If the LLM returns invalid/unparseable output
        Exception: If the API call fails (network, auth, rate limit)
    """
    if client is None:
        client = get_groq_client()

    # Build the prompt with the ticket's combined text
    ticket_text = ticket.combined_text()
    prompt = CLASSIFICATION_PROMPT.format(
        ticket_text=ticket_text,
    )

    logger.info("Classifying ticket %s", ticket.ticket_id)

    # Call the LLM
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a precise support ticket classifier. Always respond with valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,  # Low temperature for consistent, deterministic classification
        max_tokens=500,   # Classification doesn't need long responses
    )

    raw_response = response.choices[0].message.content.strip()
    logger.debug("Raw LLM response for %s: %s", ticket.ticket_id, raw_response)

    # Parse the JSON from the response
    parsed = _extract_json(raw_response)

    # Validate through Pydantic — this is where bad predictions get caught
    result = ClassificationResult(**parsed)

    logger.info(
        "Classified %s → intent=%s (%.2f), urgency=%s (%.2f), docs=%s (%.2f), must_not_auto=%s",
        ticket.ticket_id,
        result.intent,
        result.intent_confidence,
        result.urgency,
        result.urgency_confidence,
        result.answerable_from_docs,
        result.answerable_confidence,
        result.must_not_auto_respond,
    )

    return result


def classify_tickets(
    tickets: list[StandardTicket],
    client: Optional[OpenAI] = None,
) -> list[ClassificationResult]:
    """
    Classify a batch of tickets.

    Processes sequentially to respect Groq's rate limits on the free tier.
    Each ticket gets its own API call.

    Args:
        tickets: List of StandardTickets to classify
        client: Optional pre-configured client (reused across all calls)

    Returns:
        List of ClassificationResults in the same order as input
    """
    if client is None:
        client = get_groq_client()

    results = []
    for i, ticket in enumerate(tickets):
        logger.info("Classifying ticket %d/%d: %s", i + 1, len(tickets), ticket.ticket_id)
        try:
            result = classify_ticket(ticket, client=client)
            results.append(result)
        except Exception as e:
            logger.error("Failed to classify %s: %s", ticket.ticket_id, e)
            # Return a low-confidence unclear_request as fallback
            # This ensures the pipeline never crashes on a single bad ticket
            results.append(ClassificationResult(
                intent="unclear_request",
                intent_confidence=0.0,
                urgency="medium",
                urgency_confidence=0.0,
                answerable_from_docs=False,
                answerable_confidence=0.0,
                must_not_auto_respond=True,
                reasoning=f"Classification failed: {e}",
            ))

    return results


# ---------------------------------------------------------------------------
# Helper: extract JSON from LLM response
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """
    Extract a JSON object from the LLM's response text.

    LLMs sometimes wrap JSON in markdown code fences (```json ... ```)
    or add extra text before/after. This function handles those cases.

    Args:
        text: Raw text from the LLM

    Returns:
        Parsed dictionary

    Raises:
        ValueError: If no valid JSON can be extracted
    """
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fences
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding a JSON object anywhere in the text
    json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from LLM response: {text[:200]}")

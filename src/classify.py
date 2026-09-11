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
    1. We use LangChain's ChatOpenAI for LLM classification.
       Why LLM instead of a traditional ML classifier (e.g., sklearn)?
       → 22 intent categories with only ~23 samples each is too few for supervised ML.
         An LLM can generalize from the category names and descriptions alone (zero-shot).
       Why LangChain? → Provides a consistent interface across providers (OpenRouter,
         Groq, etc.) and integrates with the LangGraph pipeline orchestration.
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

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field, field_validator, model_validator

from src.config import CONFIDENCE_THRESHOLD, get_llm
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
Urgency reflects BUSINESS IMPACT to the customer, not the technical severity of the topic.

- **low**: No active service disruption. Informational questions, how-to requests, setup tasks, billing inquiries about past charges, feature requests, configuration that isn't blocking work. The customer can continue their normal work while waiting. Can wait 24+ hours.
  Examples: "How do I set up SSO?", "What are my API rate limits?", "Can you explain this invoice charge?", "I'd like to request a new feature", "Help me configure webhooks", "How do I export data?", "Questions about data residency policies"

- **medium**: The customer is experiencing a problem that affects their work but is NOT a complete outage. Individual errors, single-user access issues, non-critical failures with potential workarounds. Something is broken for THIS customer but not necessarily an emergency. Should be addressed within hours.
  Examples: "My API key is returning 401 errors", "I can't log in to my account", "My deployment failed", "Build is failing during dependency resolution", "SSO authentication is failing for our team", "Webhook deliveries are delayed", "I need to roll back my last release"

- **high**: ONLY for widespread outage, confirmed security breach, or an issue that is actively causing financial/data loss RIGHT NOW. The key test: is this affecting multiple users, losing money, or risking data? If it's a single user's individual problem (even a frustrating one), it's medium, not high.
  Examples: "Our production database is completely down", "We detected unauthorized access to our account", "Billing system charged us $50,000 incorrectly and we need immediate reversal", "All customer-facing APIs returning 500 errors", "Data breach — customer PII may be exposed", "Complete service outage affecting all our users"

## Common urgency mistakes to avoid:
- A single user's authentication failure = medium (not high), unless it indicates a security breach
- API key errors for one customer = medium (not high)
- Deployment failures = medium (not high), unless production is down for end users
- Billing questions about understanding charges = low (not medium)
- Feature requests = low, always
- Rollback requests = medium (not high), unless there's an active production outage
- "Not working" or "error" in the ticket does NOT automatically mean high — most individual errors are medium
- Billing disputes involving large incorrect charges or financial harm = high (not low)

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

**IMPORTANT — "not working" ≠ mystery:**
When a customer says something "isn't working" or they "can't figure out why," that does NOT automatically make it a mystery. Ask: does the documentation have a troubleshooting guide for this TYPE of problem? If yes, the docs CAN answer it even though the customer hasn't diagnosed it yet. The customer not knowing the cause is exactly WHY they need the troubleshooting guide.

True mysteries are when the problem type itself is unusual or unprecedented — not when the customer simply hasn't followed the troubleshooting steps yet.

Examples:
- "Builds are failing during dependency resolution" → true (docs cover dependency resolution build failure troubleshooting)
- "Our API key started returning 401 without any change on our side" → false (mystery — needs investigation of what happened to their key)
- "Need to rotate an exposed key without downtime" → true (docs cover exposed secrets and credential rotation procedures)
- "MFA code keeps getting rejected" → true (docs cover MFA setup/recovery troubleshooting)
- "Changed our shared account password and now automated jobs are failing" → false (account-specific configuration issue needing investigation)
- "How do I export access records for our auditor?" → true (docs cover data export and audit logging)
- "We need to demonstrate read access is logged but can't find the events" → false (needs investigation of their specific audit setup)
- "Staging environment is picking up production config values and we can't figure out why" → true (docs cover environment variables and secrets configuration — the troubleshooting guide addresses config resolution order)
- "Team member permissions aren't taking effect after we changed their role" → true (docs cover team member roles and permissions — the guide covers permission propagation)
- "Latency has been gradually increasing and memory usage is high" → true (docs cover response latency diagnosis — the troubleshooting steps cover memory and performance investigation)

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
# Post-classification safety overrides (v3 — fixes DEV-0030 pattern)
# ---------------------------------------------------------------------------

# Keywords that indicate ACTIVE customer-facing breakage requiring incident
# response, not just a procedural rollback.  When a rollback_request ticket
# contains these signals, the system must escalate to a human even though the
# rollback *procedure* is documented — the root cause still needs investigation.
_ACTIVE_BREAKAGE_KEYWORDS = [
    "breaking", "broken", "down", "outage", "incident",
    "customers affected", "customer-facing", "impacting customers",
    "production issue", "service disruption", "critical failure",
    "cannot process", "payments failing", "checkout broken",
    "orders failing", "500 errors in production",
]


def _apply_answerable_overrides(
    result: ClassificationResult,
    ticket_text: str,
) -> ClassificationResult:
    """
    Apply deterministic safety overrides AFTER the LLM classification.

    Why a post-LLM layer?
    The LLM decides answerable_from_docs based on whether a *procedure*
    exists in the docs.  But some tickets describe situations where the
    procedure alone is not enough — the customer needs incident-level
    human attention even though a how-to guide exists.

    This function catches those cases with keyword heuristics, similar to
    how must_not_auto_respond is already set deterministically from the
    intent rather than trusting the LLM.

    Interview context:
        "Why not just improve the prompt?"
        → The prompt already explains procedure-vs-mystery, and the LLM
          follows it well (86% accuracy).  But the remaining failures are
          *edge cases* where the LLM correctly identifies a procedure but
          misses the incident signal.  A deterministic keyword check is
          more reliable for safety-critical overrides than asking the LLM
          to weigh two competing signals.  Defense in depth.
    """
    text_lower = ticket_text.lower()

    # Override 1: rollback_request + active breakage → escalate
    # Rationale (DEV-0030 pattern):  When a customer says "bad release
    # breaking checkout," the rollback procedure exists in docs, but the
    # CAUSE of the bad release needs human investigation.  Sending a
    # generic rollback how-to while customers can't check out is dangerous.
    if result.intent == "rollback_request" and result.answerable_from_docs:
        for keyword in _ACTIVE_BREAKAGE_KEYWORDS:
            if keyword in text_lower:
                logger.info(
                    "Override: rollback_request with active breakage keyword "
                    "'%s' → answerable_from_docs=False (needs incident response)",
                    keyword,
                )
                result.answerable_from_docs = False
                result.answerable_confidence = max(0.3, result.answerable_confidence - 0.3)
                result.reasoning += (
                    f" [OVERRIDE: Active breakage detected ('{keyword}'). "
                    f"Rollback procedure exists in docs but this ticket "
                    f"describes a live incident needing human investigation.]"
                )
                break

    # Override 2: database_issue + active breakage → escalate
    # Rationale (DEV-0032 pattern):  Connection pool exhaustion with active
    # impact should not be auto-responded even if docs cover the topic.
    if result.intent == "database_issue" and result.answerable_from_docs:
        db_incident_keywords = [
            "exhausted", "no available connections", "connection refused",
            "database down", "cannot connect", "production database",
        ]
        for keyword in db_incident_keywords:
            if keyword in text_lower:
                logger.info(
                    "Override: database_issue with incident keyword '%s' "
                    "→ answerable_from_docs=False",
                    keyword,
                )
                result.answerable_from_docs = False
                result.answerable_confidence = max(0.3, result.answerable_confidence - 0.3)
                result.reasoning += (
                    f" [OVERRIDE: Active database incident detected ('{keyword}'). "
                    f"Needs human investigation of customer's specific environment.]"
                )
                break

    return result


# ---------------------------------------------------------------------------
# Core classification function
# ---------------------------------------------------------------------------

def classify_ticket(
    ticket: StandardTicket,
    llm: Optional[ChatOpenAI] = None,
) -> ClassificationResult:
    """
    Classify a single support ticket using LangChain ChatOpenAI.

    This function:
    1. Builds the classification prompt with the ticket text
    2. Sends it to the LLM via LangChain's ChatOpenAI interface
    3. Parses the JSON response
    4. Validates it through the ClassificationResult Pydantic model
    5. Applies post-classification safety overrides
    6. Returns the validated classification

    Args:
        ticket: A StandardTicket from the ingest stage
        llm: Optional pre-configured ChatOpenAI instance (for reuse / testing)
    Returns:
        ClassificationResult with intent, urgency, confidence scores, etc.

    Raises:
        ValueError: If the LLM returns invalid/unparseable output
        Exception: If the API call fails (network, auth, rate limit)
    """
    if llm is None:
        llm = get_llm()

    # Build the prompt with the ticket's combined text
    ticket_text = ticket.combined_text()
    prompt = CLASSIFICATION_PROMPT.format(
        ticket_text=ticket_text,
    )

    logger.info("Classifying ticket %s", ticket.ticket_id)

    # Call the LLM via LangChain
    messages = [
        SystemMessage(content="You are a precise support ticket classifier. Always respond with valid JSON only."),
        HumanMessage(content=prompt),
    ]

    # Override temperature for classification (deterministic)
    response = llm.invoke(messages, temperature=0.1, max_tokens=500)

    raw_response = response.content.strip()
    logger.debug("Raw LLM response for %s: %s", ticket.ticket_id, raw_response)

    # Parse the JSON from the response
    parsed = _extract_json(raw_response)

    # Validate through Pydantic — this is where bad predictions get caught
    result = ClassificationResult(**parsed)

    # Apply post-classification safety overrides
    result = _apply_answerable_overrides(result, ticket_text)

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
    llm: Optional[ChatOpenAI] = None,
) -> list[ClassificationResult]:
    """
    Classify a batch of tickets.

    Processes sequentially to respect rate limits on the free tier.
    Each ticket gets its own API call.

    Args:
        tickets: List of StandardTickets to classify
        llm: Optional pre-configured ChatOpenAI instance (reused across all calls)

    Returns:
        List of ClassificationResults in the same order as input
    """
    if llm is None:
        llm = get_llm()

    results = []
    for i, ticket in enumerate(tickets):
        logger.info("Classifying ticket %d/%d: %s", i + 1, len(tickets), ticket.ticket_id)
        try:
            result = classify_ticket(ticket, llm=llm)
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

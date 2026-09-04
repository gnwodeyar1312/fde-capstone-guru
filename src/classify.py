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
- compliance_request: Regulatory compliance questions (GDPR, SOC2, HIPAA, etc.)
- configuration_help: Help configuring CloudServe services or settings
- data_export: Requests to export or download their data
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
- true: The answer likely exists in CloudServe's official documentation (29 articles covering API guides, deployment, auth, billing, data management, etc.)
- false: Requires human investigation, account-specific actions, or involves topics not covered in docs

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
    )


# ---------------------------------------------------------------------------
# Core classification function
# ---------------------------------------------------------------------------

def classify_ticket(ticket: StandardTicket, client: Optional[OpenAI] = None) -> ClassificationResult:
    """
    Classify a single support ticket using the Groq LLM.

    This function:
    1. Builds the classification prompt with the ticket text
    2. Sends it to the LLM via Groq's API
    3. Parses the JSON response
    4. Validates it through the ClassificationResult Pydantic model
    5. Returns the validated classification

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
    prompt = CLASSIFICATION_PROMPT.format(ticket_text=ticket_text)

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

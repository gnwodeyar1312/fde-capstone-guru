"""
Route module for CloudServe Support System.

This is the FOURTH stage of the pipeline: Ingest → Classify → Retrieve → **Route** → Generate → Validate

Purpose:
    Given a classification result, decide whether to auto-respond or escalate.
    This is a DETERMINISTIC, RULE-BASED decision — no LLM involved.

Design decisions:
    1. Routing is rule-based, not ML-based.
       Why? Routing decisions have clear business rules that must be followed
       exactly. An LLM might "decide" to auto-respond to a security incident
       because the customer sounds calm. Rules don't make that mistake.
    2. The rules are ordered by priority (safety first).
       Why? The order matters: must_not_auto_respond is checked BEFORE
       confidence thresholds. A security_incident with 0.99 confidence
       still gets escalated. Safety trumps efficiency.
    3. Every route decision includes a reason string.
       Why? The decision log needs to explain WHY something was escalated.
       "Escalated because intent is security_incident (must-not-auto-respond)"
       is auditable. "Escalated" is not.
    4. The confidence threshold (0.80) comes from config, not hardcoded.
       Why? We might want to adjust it based on evaluation results. Too low
       and we auto-respond to things we shouldn't. Too high and we escalate
       everything, defeating the purpose.

Interview context:
    "Why not let the LLM decide the route?"
    → The routing rules encode BUSINESS POLICY, not prediction. Compliance
      requests MUST go to the compliance team — that's a legal requirement,
      not a pattern to learn. Mixing policy with prediction is how you get
      "the AI decided not to escalate the data breach."
    "What if the classifier is wrong?"
    → That's exactly what the confidence threshold handles. A ticket
      classified as billing_query with 0.55 confidence gets escalated
      because we're not sure enough to auto-respond. The threshold is
      the bridge between classification uncertainty and routing safety.
    "Expected routing split?"
    → From discovery analysis: ~311 auto_respond, ~189 escalate out of 500.
      This means ~62% of tickets get automated responses — a significant
      workload reduction while keeping the safety guardrails.
"""

import logging
from enum import Enum

from pydantic import BaseModel

from src.classify import ClassificationResult
from src.config import CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Route decision model
# ---------------------------------------------------------------------------


class RouteAction(str, Enum):
    """The two possible routing outcomes."""

    AUTO_RESPOND = "auto_respond"
    ESCALATE = "escalate"


class RouteDecision(BaseModel):
    """
    The output of the route stage.

    Every decision carries a reason — this feeds the decision log
    and makes the system auditable.
    """

    action: RouteAction
    reason: str
    escalation_target: str | None = None  # Which team to escalate to
    confidence_met: bool = True  # Whether confidence was above threshold
    rule_triggered: str = ""  # Which rule caused the decision

    @property
    def is_auto_respond(self) -> bool:
        return self.action == RouteAction.AUTO_RESPOND

    @property
    def is_escalate(self) -> bool:
        return self.action == RouteAction.ESCALATE


# ---------------------------------------------------------------------------
# Escalation targets — which team handles what
# ---------------------------------------------------------------------------

ESCALATION_TARGETS = {
    "compliance_request": "compliance_team",
    "security_incident": "security_team",
    "feature_request": "product_team",
    "unclear_request": "tier1_support",
}


# ---------------------------------------------------------------------------
# Core routing function
# ---------------------------------------------------------------------------


def route_ticket(
    classification: ClassificationResult,
    threshold: float | None = None,
) -> RouteDecision:
    """
    Route a classified ticket to auto-respond or escalate.

    Rules are applied in priority order (safety first):
        1. must_not_auto_respond intents → ALWAYS escalate
        2. Not answerable from docs → escalate
        3. Intent confidence below threshold → escalate
        4. All checks pass → auto-respond

    Args:
        classification: The ClassificationResult from the classify stage
        threshold: Confidence threshold (default: CONFIDENCE_THRESHOLD from config)

    Returns:
        RouteDecision with action, reason, and metadata
    """
    if threshold is None:
        threshold = CONFIDENCE_THRESHOLD

    # ── Rule 1: Must-not-auto-respond intents (SAFETY FIRST) ──────────
    # These four intents are NEVER auto-responded, regardless of confidence.
    # This is a hard business rule, not a prediction.
    if classification.must_not_auto_respond:
        target = ESCALATION_TARGETS.get(classification.intent, "tier1_support")
        decision = RouteDecision(
            action=RouteAction.ESCALATE,
            reason=(
                f"Intent '{classification.intent}' requires human review "
                f"(must-not-auto-respond policy). Routing to {target}."
            ),
            escalation_target=target,
            confidence_met=True,  # Confidence is irrelevant here
            rule_triggered="must_not_auto_respond",
        )
        logger.info(
            "Route: ESCALATE (must_not_auto_respond) → %s | intent=%s, conf=%.2f",
            target,
            classification.intent,
            classification.intent_confidence,
        )
        return decision

    # ── Rule 2: Not answerable from docs ──────────────────────────────
    # If the classifier says docs can't answer this, there's no point
    # trying to generate a response from them.
    if not classification.answerable_from_docs:
        decision = RouteDecision(
            action=RouteAction.ESCALATE,
            reason=(
                f"Ticket classified as not answerable from documentation "
                f"(confidence: {classification.answerable_confidence:.2f}). "
                f"Requires human investigation."
            ),
            escalation_target="tier1_support",
            confidence_met=True,
            rule_triggered="not_answerable_from_docs",
        )
        logger.info(
            "Route: ESCALATE (not answerable) | intent=%s, answerable_conf=%.2f",
            classification.intent,
            classification.answerable_confidence,
        )
        return decision

    # ── Rule 3: Low confidence → escalate ─────────────────────────────
    # If we're not confident enough in the classification, don't risk
    # sending a wrong automated response.
    if classification.intent_confidence < threshold:
        decision = RouteDecision(
            action=RouteAction.ESCALATE,
            reason=(
                f"Intent confidence ({classification.intent_confidence:.2f}) "
                f"is below threshold ({threshold:.2f}). "
                f"Not confident enough to auto-respond."
            ),
            escalation_target="tier1_support",
            confidence_met=False,
            rule_triggered="low_confidence",
        )
        logger.info(
            "Route: ESCALATE (low confidence %.2f < %.2f) | intent=%s",
            classification.intent_confidence,
            threshold,
            classification.intent,
        )
        return decision

    # ── Rule 4: All checks pass → auto-respond ────────────────────────
    decision = RouteDecision(
        action=RouteAction.AUTO_RESPOND,
        reason=(
            f"Intent '{classification.intent}' with confidence "
            f"{classification.intent_confidence:.2f} (≥{threshold:.2f}), "
            f"answerable from docs. Safe to auto-respond."
        ),
        confidence_met=True,
        rule_triggered="all_checks_passed",
    )
    logger.info(
        "Route: AUTO_RESPOND | intent=%s, conf=%.2f, answerable=True",
        classification.intent,
        classification.intent_confidence,
    )
    return decision

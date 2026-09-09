"""
Unit tests for the route module.

Tests all routing rules, edge cases, and the priority ordering.
These are pure logic tests — no LLM or API calls needed.
"""

import pytest

from src.classify import ClassificationResult
from src.route import RouteAction, RouteDecision, route_ticket

# ---------------------------------------------------------------------------
# Helper to build ClassificationResult quickly
# ---------------------------------------------------------------------------


def make_classification(
    intent="billing_query",
    intent_confidence=0.90,
    urgency="low",
    urgency_confidence=0.85,
    answerable_from_docs=True,
    answerable_confidence=0.85,
    reasoning="test",
) -> ClassificationResult:
    return ClassificationResult(
        intent=intent,
        intent_confidence=intent_confidence,
        urgency=urgency,
        urgency_confidence=urgency_confidence,
        answerable_from_docs=answerable_from_docs,
        answerable_confidence=answerable_confidence,
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Rule 1: Must-not-auto-respond intents
# ---------------------------------------------------------------------------


class TestMustNotAutoRespond:
    """Rule 1: These four intents ALWAYS escalate, regardless of confidence."""

    @pytest.mark.parametrize(
        "intent,expected_target",
        [
            ("compliance_request", "compliance_team"),
            ("security_incident", "security_team"),
            ("feature_request", "product_team"),
            ("unclear_request", "tier1_support"),
        ],
    )
    def test_must_not_auto_respond_intents(self, intent, expected_target):
        c = make_classification(intent=intent, intent_confidence=0.99)
        r = route_ticket(c)
        assert r.is_escalate
        assert r.rule_triggered == "must_not_auto_respond"
        assert r.escalation_target == expected_target

    def test_security_incident_even_with_perfect_confidence(self):
        """Even 1.0 confidence doesn't override the must-not rule."""
        c = make_classification(
            intent="security_incident",
            intent_confidence=1.0,
            answerable_from_docs=True,
        )
        r = route_ticket(c)
        assert r.is_escalate
        assert r.rule_triggered == "must_not_auto_respond"


# ---------------------------------------------------------------------------
# Rule 2: Not answerable from docs
# ---------------------------------------------------------------------------


class TestNotAnswerableFromDocs:
    """Rule 2: If docs can't answer it, escalate."""

    def test_not_answerable_escalates(self):
        c = make_classification(
            intent="database_issue",
            intent_confidence=0.95,
            answerable_from_docs=False,
        )
        r = route_ticket(c)
        assert r.is_escalate
        assert r.rule_triggered == "not_answerable_from_docs"

    def test_not_answerable_goes_to_tier1(self):
        c = make_classification(
            intent="database_issue",
            intent_confidence=0.95,
            answerable_from_docs=False,
        )
        r = route_ticket(c)
        assert r.escalation_target == "tier1_support"


# ---------------------------------------------------------------------------
# Rule 3: Low confidence
# ---------------------------------------------------------------------------


class TestLowConfidence:
    """Rule 3: Below threshold → escalate."""

    def test_below_threshold_escalates(self):
        c = make_classification(intent_confidence=0.55)
        r = route_ticket(c)
        assert r.is_escalate
        assert r.rule_triggered == "low_confidence"
        assert not r.confidence_met

    def test_just_below_threshold_escalates(self):
        """0.79 is below the 0.80 threshold."""
        c = make_classification(intent_confidence=0.79)
        r = route_ticket(c)
        assert r.is_escalate
        assert r.rule_triggered == "low_confidence"

    def test_custom_threshold(self):
        """Can override the default threshold."""
        c = make_classification(intent_confidence=0.85)
        # With default (0.80), this would pass. With 0.90, it should fail.
        r = route_ticket(c, threshold=0.90)
        assert r.is_escalate
        assert r.rule_triggered == "low_confidence"


# ---------------------------------------------------------------------------
# Rule 4: All checks pass → auto-respond
# ---------------------------------------------------------------------------


class TestAutoRespond:
    """Rule 4: When everything checks out, auto-respond."""

    def test_all_checks_pass(self):
        c = make_classification(
            intent="billing_query",
            intent_confidence=0.92,
            answerable_from_docs=True,
        )
        r = route_ticket(c)
        assert r.is_auto_respond
        assert r.rule_triggered == "all_checks_passed"
        assert r.confidence_met

    def test_exactly_at_threshold_passes(self):
        """0.80 exactly meets the >= 0.80 threshold."""
        c = make_classification(intent_confidence=0.80)
        r = route_ticket(c)
        assert r.is_auto_respond


# ---------------------------------------------------------------------------
# Rule priority ordering
# ---------------------------------------------------------------------------


class TestRulePriority:
    """Verify that rules are applied in the correct priority order."""

    def test_must_not_trumps_answerable(self):
        """Rule 1 > Rule 2: must_not checked before answerable."""
        c = make_classification(
            intent="compliance_request",
            intent_confidence=0.99,
            answerable_from_docs=True,  # Even if answerable, still escalate
        )
        r = route_ticket(c)
        assert r.rule_triggered == "must_not_auto_respond"

    def test_must_not_trumps_confidence(self):
        """Rule 1 > Rule 3: must_not checked before confidence."""
        c = make_classification(
            intent="security_incident",
            intent_confidence=0.30,  # Low confidence, but rule 1 fires first
        )
        r = route_ticket(c)
        assert r.rule_triggered == "must_not_auto_respond"

    def test_not_answerable_trumps_confidence(self):
        """Rule 2 > Rule 3: not_answerable checked before confidence."""
        c = make_classification(
            intent="billing_query",
            intent_confidence=0.50,  # Low confidence
            answerable_from_docs=False,  # But rule 2 fires first
        )
        r = route_ticket(c)
        assert r.rule_triggered == "not_answerable_from_docs"


# ---------------------------------------------------------------------------
# RouteDecision model tests
# ---------------------------------------------------------------------------


class TestRouteDecision:
    """Tests for the RouteDecision Pydantic model."""

    def test_is_auto_respond_property(self):
        d = RouteDecision(
            action=RouteAction.AUTO_RESPOND,
            reason="test",
        )
        assert d.is_auto_respond
        assert not d.is_escalate

    def test_is_escalate_property(self):
        d = RouteDecision(
            action=RouteAction.ESCALATE,
            reason="test",
        )
        assert d.is_escalate
        assert not d.is_auto_respond

    def test_reason_is_always_present(self):
        """Every route decision from route_ticket has a non-empty reason."""
        c = make_classification()
        r = route_ticket(c)
        assert len(r.reason) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

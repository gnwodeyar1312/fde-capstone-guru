"""
Tests for the classify module.

Three test classes:
1. TestClassificationResult — Pydantic validation of the output model
2. TestExtractJson — JSON extraction from messy LLM responses
3. TestClassifyTicket — End-to-end classification (requires Groq API key)
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from src.classify import (
    ClassificationResult,
    VALID_INTENTS,
    MUST_NOT_AUTO_RESPOND_INTENTS,
    _extract_json,
    classify_ticket,
)
from src.ingest import StandardTicket


# ---------------------------------------------------------------------------
# Test fixtures — reusable test data
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_ticket():
    """A typical support ticket for testing."""
    return StandardTicket(
        ticket_id="TEST-001",
        channel="email",
        subject="Cannot deploy my application",
        body="I keep getting error 503 when trying to deploy. My deployment keeps dying after the health check. This has been happening since yesterday.",
        received_at="2026-01-15T10:30:00Z",
        customer_id="CUST-100",
        customer_name="Test User",
        customer_tier="standard",
        customer_region="us-east",
        language_fluency="fluent",
    )


@pytest.fixture
def security_ticket():
    """A security incident ticket — must not auto-respond."""
    return StandardTicket(
        ticket_id="TEST-002",
        channel="email",
        subject="Unauthorized access detected",
        body="We noticed login attempts from unknown IP addresses to our admin panel. We suspect our API keys may have been compromised.",
        received_at="2026-01-15T11:00:00Z",
        customer_id="CUST-200",
        customer_name="Security Alert User",
        customer_tier="enterprise",
        customer_region="eu-west",
        language_fluency="fluent",
    )


# ---------------------------------------------------------------------------
# TestClassificationResult — does the Pydantic model validate correctly?
# ---------------------------------------------------------------------------

class TestClassificationResult:
    """Test the Pydantic output model validation."""

    def test_valid_classification(self):
        """A well-formed classification should pass validation."""
        result = ClassificationResult(
            intent="deployment_failure",
            intent_confidence=0.92,
            urgency="high",
            urgency_confidence=0.85,
            answerable_from_docs=True,
            answerable_confidence=0.78,
            reasoning="Customer reports deployment error 503",
        )
        assert result.intent == "deployment_failure"
        assert result.urgency == "high"
        assert result.must_not_auto_respond is False

    def test_invalid_intent_rejected(self):
        """An intent not in our 22 valid intents should be rejected."""
        with pytest.raises(ValueError, match="Unknown intent"):
            ClassificationResult(
                intent="pizza_order",  # not a real intent
                intent_confidence=0.9,
                urgency="high",
                urgency_confidence=0.9,
                answerable_from_docs=False,
                answerable_confidence=0.9,
            )

    def test_invalid_urgency_rejected(self):
        """An urgency not in {low, medium, high} should be rejected."""
        with pytest.raises(ValueError, match="Unknown urgency"):
            ClassificationResult(
                intent="billing_query",
                intent_confidence=0.9,
                urgency="critical",  # not valid
                urgency_confidence=0.9,
                answerable_from_docs=False,
                answerable_confidence=0.9,
            )

    def test_confidence_out_of_range_rejected(self):
        """Confidence must be between 0.0 and 1.0."""
        with pytest.raises(ValueError):
            ClassificationResult(
                intent="billing_query",
                intent_confidence=1.5,  # too high
                urgency="low",
                urgency_confidence=0.9,
                answerable_from_docs=False,
                answerable_confidence=0.9,
            )

    def test_must_not_auto_respond_set_for_compliance(self):
        """compliance_request should auto-set must_not_auto_respond=True."""
        result = ClassificationResult(
            intent="compliance_request",
            intent_confidence=0.88,
            urgency="medium",
            urgency_confidence=0.75,
            answerable_from_docs=False,
            answerable_confidence=0.80,
        )
        assert result.must_not_auto_respond is True

    def test_must_not_auto_respond_set_for_security(self):
        """security_incident should auto-set must_not_auto_respond=True."""
        result = ClassificationResult(
            intent="security_incident",
            intent_confidence=0.91,
            urgency="high",
            urgency_confidence=0.95,
            answerable_from_docs=False,
            answerable_confidence=0.90,
        )
        assert result.must_not_auto_respond is True

    def test_must_not_auto_respond_set_for_feature_request(self):
        """feature_request should auto-set must_not_auto_respond=True."""
        result = ClassificationResult(
            intent="feature_request",
            intent_confidence=0.85,
            urgency="low",
            urgency_confidence=0.80,
            answerable_from_docs=False,
            answerable_confidence=0.90,
        )
        assert result.must_not_auto_respond is True

    def test_must_not_auto_respond_set_for_unclear(self):
        """unclear_request should auto-set must_not_auto_respond=True."""
        result = ClassificationResult(
            intent="unclear_request",
            intent_confidence=0.60,
            urgency="medium",
            urgency_confidence=0.50,
            answerable_from_docs=False,
            answerable_confidence=0.40,
        )
        assert result.must_not_auto_respond is True

    def test_must_not_auto_respond_false_for_normal_intent(self):
        """A normal intent like billing_query should NOT be flagged."""
        result = ClassificationResult(
            intent="billing_query",
            intent_confidence=0.90,
            urgency="low",
            urgency_confidence=0.85,
            answerable_from_docs=True,
            answerable_confidence=0.88,
        )
        assert result.must_not_auto_respond is False

    def test_all_22_intents_accepted(self):
        """Every one of our 22 valid intents should pass validation."""
        for intent in VALID_INTENTS:
            result = ClassificationResult(
                intent=intent,
                intent_confidence=0.80,
                urgency="medium",
                urgency_confidence=0.80,
                answerable_from_docs=False,
                answerable_confidence=0.80,
            )
            assert result.intent == intent

    def test_intent_case_insensitive(self):
        """Intent matching should be case-insensitive."""
        result = ClassificationResult(
            intent="Billing_Query",
            intent_confidence=0.90,
            urgency="LOW",
            urgency_confidence=0.85,
            answerable_from_docs=True,
            answerable_confidence=0.88,
        )
        assert result.intent == "billing_query"
        assert result.urgency == "low"


# ---------------------------------------------------------------------------
# TestExtractJson — can we handle messy LLM output?
# ---------------------------------------------------------------------------

class TestExtractJson:
    """Test JSON extraction from various LLM response formats."""

    def test_clean_json(self):
        """Direct JSON without any wrapping."""
        text = '{"intent": "billing_query", "intent_confidence": 0.9}'
        result = _extract_json(text)
        assert result["intent"] == "billing_query"

    def test_json_in_code_fence(self):
        """JSON wrapped in markdown code fences."""
        text = '```json\n{"intent": "billing_query", "intent_confidence": 0.9}\n```'
        result = _extract_json(text)
        assert result["intent"] == "billing_query"

    def test_json_in_plain_code_fence(self):
        """JSON wrapped in code fences without language tag."""
        text = '```\n{"intent": "billing_query", "intent_confidence": 0.9}\n```'
        result = _extract_json(text)
        assert result["intent"] == "billing_query"

    def test_json_with_extra_text(self):
        """JSON embedded in surrounding text."""
        text = 'Here is my analysis:\n{"intent": "billing_query", "intent_confidence": 0.9}\nDone.'
        result = _extract_json(text)
        assert result["intent"] == "billing_query"

    def test_invalid_json_raises(self):
        """Completely invalid text should raise ValueError."""
        with pytest.raises(ValueError, match="Could not extract valid JSON"):
            _extract_json("This is not JSON at all")


# ---------------------------------------------------------------------------
# TestClassifyTicket — end-to-end with mock LLM
# ---------------------------------------------------------------------------

class TestClassifyTicket:
    """Test the full classify_ticket function with mocked LLM."""

    def _mock_client(self, response_json: dict) -> MagicMock:
        """Create a mock OpenAI client that returns the given JSON."""
        mock_message = MagicMock()
        mock_message.content = json.dumps(response_json)

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        return mock_client

    def test_classify_deployment_ticket(self, sample_ticket):
        """A deployment failure ticket should be classified correctly."""
        mock_response = {
            "intent": "deployment_failure",
            "intent_confidence": 0.92,
            "urgency": "high",
            "urgency_confidence": 0.85,
            "answerable_from_docs": True,
            "answerable_confidence": 0.78,
            "reasoning": "Customer reports 503 error during deployment",
        }
        client = self._mock_client(mock_response)

        result = classify_ticket(sample_ticket, client=client)

        assert result.intent == "deployment_failure"
        assert result.urgency == "high"
        assert result.answerable_from_docs is True
        assert result.must_not_auto_respond is False
        assert result.intent_confidence == 0.92

    def test_classify_security_ticket(self, security_ticket):
        """A security incident must be flagged as must_not_auto_respond."""
        mock_response = {
            "intent": "security_incident",
            "intent_confidence": 0.95,
            "urgency": "high",
            "urgency_confidence": 0.97,
            "answerable_from_docs": False,
            "answerable_confidence": 0.90,
            "reasoning": "Reports unauthorized access and potential key compromise",
        }
        client = self._mock_client(mock_response)

        result = classify_ticket(security_ticket, client=client)

        assert result.intent == "security_incident"
        assert result.must_not_auto_respond is True
        assert result.urgency == "high"

    def test_classify_handles_llm_failure_gracefully(self, sample_ticket):
        """If the LLM returns garbage, classify_ticket should raise."""
        mock_message = MagicMock()
        mock_message.content = "I don't understand the question"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with pytest.raises(ValueError, match="Could not extract valid JSON"):
            classify_ticket(sample_ticket, client=mock_client)

    def test_classify_strips_code_fences(self, sample_ticket):
        """LLM response wrapped in code fences should still work."""
        response_json = {
            "intent": "deployment_failure",
            "intent_confidence": 0.88,
            "urgency": "medium",
            "urgency_confidence": 0.75,
            "answerable_from_docs": True,
            "answerable_confidence": 0.82,
            "reasoning": "Deployment issue",
        }
        # Simulate LLM wrapping response in code fences
        mock_message = MagicMock()
        mock_message.content = f"```json\n{json.dumps(response_json)}\n```"

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        result = classify_ticket(sample_ticket, client=mock_client)
        assert result.intent == "deployment_failure"

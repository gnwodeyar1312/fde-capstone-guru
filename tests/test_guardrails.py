"""
Unit tests for the guardrails module.

Tests each guardrail check independently and the combined validation.
All deterministic — no LLM or API calls.
"""

import pytest

from src.guardrails import (
    validate_response,
    _check_citations,
    _check_no_fabricated_urls,
    _check_no_pii_leakage,
    _check_response_length,
    _check_automated_footer,
    GuardrailResult,
    GuardrailCheck,
)
from src.generate import GeneratedResponse
from src.retrieve import RetrievedChunk, RetrievalResult


# ---------------------------------------------------------------------------
# Test data factories
# ---------------------------------------------------------------------------

def make_response(text: str = None, cited_ids: list[str] = None) -> GeneratedResponse:
    if text is None:
        text = (
            "Hello,\n\n"
            "Based on our documentation [DOC-DEPLOY-001], you should check "
            "your Dockerfile configuration.\n\n"
            "---\n"
            "This is an automated response from CloudServe Support."
        )
    if cited_ids is None:
        cited_ids = ["DOC-DEPLOY-001"]
    return GeneratedResponse(
        response_text=text,
        cited_doc_ids=cited_ids,
    )


def make_retrieval(doc_ids: list[str] = None) -> RetrievalResult:
    if doc_ids is None:
        doc_ids = ["DOC-DEPLOY-001"]
    chunks = [
        RetrievedChunk(
            doc_id=did,
            title=f"Doc {did}",
            category="test",
            section_name="Resolution",
            content="Some content.",
            similarity_score=0.3,
        )
        for did in doc_ids
    ]
    return RetrievalResult(query="test", chunks=chunks)


# ---------------------------------------------------------------------------
# Citation validation tests
# ---------------------------------------------------------------------------

class TestCitationValidation:
    """Guardrail 1: Citation validation."""

    def test_valid_citations_pass(self):
        response = make_response(cited_ids=["DOC-DEPLOY-001"])
        retrieval = make_retrieval(doc_ids=["DOC-DEPLOY-001", "DOC-AUTH-002"])
        result = _check_citations(response, retrieval)
        assert result.passed

    def test_hallucinated_citation_fails(self):
        response = make_response(cited_ids=["DOC-DEPLOY-001", "DOC-FAKE-999"])
        retrieval = make_retrieval(doc_ids=["DOC-DEPLOY-001"])
        result = _check_citations(response, retrieval)
        assert not result.passed
        assert result.severity == "high"
        assert "DOC-FAKE-999" in result.reason

    def test_no_citations_with_chunks_warns(self):
        response = make_response(cited_ids=[])
        retrieval = make_retrieval(doc_ids=["DOC-DEPLOY-001"])
        result = _check_citations(response, retrieval)
        assert not result.passed
        assert result.severity == "medium"

    def test_no_citations_no_chunks_passes(self):
        response = make_response(cited_ids=[])
        retrieval = RetrievalResult(query="test", chunks=[])
        result = _check_citations(response, retrieval)
        assert result.passed

    def test_multiple_valid_citations_pass(self):
        response = make_response(cited_ids=["DOC-DEPLOY-001", "DOC-AUTH-002"])
        retrieval = make_retrieval(doc_ids=["DOC-DEPLOY-001", "DOC-AUTH-002", "DOC-PERF-003"])
        result = _check_citations(response, retrieval)
        assert result.passed


# ---------------------------------------------------------------------------
# URL check tests
# ---------------------------------------------------------------------------

class TestNoFabricatedUrls:
    """Guardrail 2: No fabricated URLs."""

    def test_no_urls_passes(self):
        response = make_response(text="Go to Settings > API Keys to find your key.")
        result = _check_no_fabricated_urls(response)
        assert result.passed

    def test_http_url_fails(self):
        response = make_response(text="Visit https://cloudserve.com/reset for help.")
        result = _check_no_fabricated_urls(response)
        assert not result.passed
        assert result.severity == "high"

    def test_multiple_urls_reported(self):
        response = make_response(
            text="See https://fake.com and http://also-fake.com for details."
        )
        result = _check_no_fabricated_urls(response)
        assert not result.passed
        assert "2" in result.reason


# ---------------------------------------------------------------------------
# PII leakage tests
# ---------------------------------------------------------------------------

class TestNoPiiLeakage:
    """Guardrail 3: No PII leakage."""

    def test_no_pii_passes(self):
        response = make_response(text="Please check your deployment settings.")
        result = _check_no_pii_leakage(response)
        assert result.passed

    def test_email_detected(self):
        response = make_response(text="We see your account john@example.com has issues.")
        result = _check_no_pii_leakage(response)
        assert not result.passed
        assert result.severity == "high"

    def test_allowed_email_passes(self):
        response = make_response(text="Contact support@cloudserve.com for help.")
        result = _check_no_pii_leakage(response)
        assert result.passed

    def test_phone_detected(self):
        response = make_response(text="Your number 555-123-4567 is on file.")
        result = _check_no_pii_leakage(response)
        assert not result.passed

    def test_credit_card_detected(self):
        response = make_response(text="Card ending in 4111-1111-1111-1111 was charged.")
        result = _check_no_pii_leakage(response)
        assert not result.passed

    def test_pii_is_redacted_in_reason(self):
        """The actual PII value should be redacted in the check reason."""
        response = make_response(text="Email john@example.com found.")
        result = _check_no_pii_leakage(response)
        assert "john@example.com" not in result.reason
        assert "***" in result.reason


# ---------------------------------------------------------------------------
# Response length tests
# ---------------------------------------------------------------------------

class TestResponseLength:
    """Guardrail 4: Response length bounds."""

    def test_normal_length_passes(self):
        response = make_response(text="A" * 500)
        result = _check_response_length(response)
        assert result.passed

    def test_too_short_fails(self):
        response = make_response(text="Hi.")
        result = _check_response_length(response)
        assert not result.passed
        assert result.severity == "high"

    def test_too_long_warns(self):
        response = make_response(text="A" * 3500)
        result = _check_response_length(response)
        assert not result.passed
        assert result.severity == "medium"

    def test_boundary_50_passes(self):
        response = make_response(text="A" * 50)
        result = _check_response_length(response)
        assert result.passed

    def test_boundary_49_fails(self):
        response = make_response(text="A" * 49)
        result = _check_response_length(response)
        assert not result.passed


# ---------------------------------------------------------------------------
# Automated footer tests
# ---------------------------------------------------------------------------

class TestAutomatedFooter:
    """Guardrail 5: Automated response footer."""

    def test_footer_present_passes(self):
        response = make_response(
            text="Here's your answer.\n---\nThis is an automated response."
        )
        result = _check_automated_footer(response)
        assert result.passed

    def test_footer_missing_warns(self):
        response = make_response(text="Here's your answer. Best regards.")
        result = _check_automated_footer(response)
        assert not result.passed
        assert result.severity == "medium"

    def test_auto_generated_variant_passes(self):
        response = make_response(text="This reply was auto-generated by our system.")
        result = _check_automated_footer(response)
        assert result.passed


# ---------------------------------------------------------------------------
# Combined validation tests
# ---------------------------------------------------------------------------

class TestValidateResponse:
    """Tests for the combined validate_response function."""

    def test_good_response_passes_all(self):
        response = make_response()
        retrieval = make_retrieval()
        result = validate_response(response, retrieval)
        assert result.passed
        assert len(result.failed_checks) == 0

    def test_hallucinated_citation_blocks(self):
        response = make_response(cited_ids=["DOC-FAKE-999"])
        retrieval = make_retrieval(doc_ids=["DOC-DEPLOY-001"])
        result = validate_response(response, retrieval)
        assert not result.passed
        assert "citation_validation" in result.failed_checks

    def test_all_checks_run_even_if_one_fails(self):
        """All 5 guardrails should run regardless of individual results."""
        response = make_response(
            text="Bad",  # too short
            cited_ids=["DOC-FAKE-999"],  # hallucinated
        )
        retrieval = make_retrieval(doc_ids=["DOC-DEPLOY-001"])
        result = validate_response(response, retrieval)
        assert len(result.checks) == 5  # All 5 checks ran
        assert not result.passed

    def test_warnings_dont_block(self):
        """Medium-severity failures are warnings, not blockers."""
        response = make_response(
            text=(
                "Here is a detailed answer based on [DOC-DEPLOY-001]. "
                + "More detail. " * 200  # ~3200 chars — triggers length warning
            ),
            cited_ids=["DOC-DEPLOY-001"],
        )
        retrieval = make_retrieval(doc_ids=["DOC-DEPLOY-001"])
        result = validate_response(response, retrieval)
        # Length warning and missing footer warning shouldn't block
        assert result.passed or all(
            c.severity == "medium" for c in result.checks if not c.passed
        )

    def test_recommendation_on_failure(self):
        response = make_response(cited_ids=["DOC-FAKE-999"])
        retrieval = make_retrieval(doc_ids=["DOC-DEPLOY-001"])
        result = validate_response(response, retrieval)
        assert "Escalate" in result.recommendation

    def test_recommendation_on_success(self):
        response = make_response()
        retrieval = make_retrieval()
        result = validate_response(response, retrieval)
        assert "Safe to send" in result.recommendation


# ---------------------------------------------------------------------------
# GuardrailResult model tests
# ---------------------------------------------------------------------------

class TestGuardrailResultModel:
    """Tests for the GuardrailResult Pydantic model."""

    def test_empty_checks_passes(self):
        result = GuardrailResult(checks=[])
        assert result.passed

    def test_all_passed_checks(self):
        checks = [
            GuardrailCheck(name="check1", passed=True, reason="ok"),
            GuardrailCheck(name="check2", passed=True, reason="ok"),
        ]
        result = GuardrailResult(checks=checks)
        assert result.passed
        assert result.failed_checks == []

    def test_high_severity_failure_blocks(self):
        checks = [
            GuardrailCheck(name="check1", passed=False, reason="bad", severity="high"),
        ]
        result = GuardrailResult(checks=checks)
        assert not result.passed
        assert "check1" in result.failed_checks

    def test_medium_severity_failure_warns(self):
        checks = [
            GuardrailCheck(name="check1", passed=False, reason="meh", severity="medium"),
        ]
        result = GuardrailResult(checks=checks)
        assert result.passed  # medium doesn't block
        assert "check1" in result.warnings


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Unit tests for the generate module.

Tests the generation pipeline WITHOUT calling the real LLM.
We mock the Groq/OpenAI client to test prompt construction,
citation extraction, and the could_answer detection logic.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.generate import (
    generate_response,
    _extract_citations,
    _format_context_block,
    _indicates_cannot_answer,
    GeneratedResponse,
    GENERATION_PROMPT,
)
from src.classify import ClassificationResult
from src.ingest import StandardTicket
from src.retrieve import RetrievedChunk, RetrievalResult


# ---------------------------------------------------------------------------
# Test data factories
# ---------------------------------------------------------------------------

def make_ticket(**overrides) -> StandardTicket:
    defaults = dict(
        ticket_id="TICKET-001",
        subject="Cannot deploy my container",
        body="My deployment keeps failing with exit code 1 during health check.",
        channel="email",
        customer_tier="professional",
        timestamp="2025-01-15T10:30:00Z",
        received_at="2025-01-15T10:30:00Z",
        customer_id="CUST-001",
    )
    defaults.update(overrides)
    return StandardTicket(**defaults)


def make_classification(**overrides) -> ClassificationResult:
    defaults = dict(
        intent="deployment_failure",
        intent_confidence=0.92,
        urgency="medium",
        urgency_confidence=0.85,
        answerable_from_docs=True,
        answerable_confidence=0.88,
        reasoning="Customer reports container deployment failure.",
    )
    defaults.update(overrides)
    return ClassificationResult(**defaults)


def make_retrieval_result(num_chunks=2) -> RetrievalResult:
    chunks = []
    for i in range(num_chunks):
        chunks.append(RetrievedChunk(
            doc_id=f"DOC-DEPLOY-{i+1:03d}",
            title=f"Deployment Guide {i+1}",
            category="deployment",
            section_name="Resolution",
            content=f"Step {i+1}: Check your Dockerfile health check configuration.",
            similarity_score=0.3 + (i * 0.1),
            applies_to="All plans",
        ))
    return RetrievalResult(query="container deployment failing", chunks=chunks)


def make_mock_client(response_text: str) -> MagicMock:
    """Create a mock OpenAI client that returns the given response text."""
    client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = response_text
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    client.chat.completions.create.return_value = mock_response
    return client


# ---------------------------------------------------------------------------
# Citation extraction tests
# ---------------------------------------------------------------------------

class TestExtractCitations:
    """Tests for _extract_citations — regex-based doc_id extraction."""

    def test_single_citation(self):
        text = "Check your config [DOC-AUTH-001] for details."
        assert _extract_citations(text) == ["DOC-AUTH-001"]

    def test_multiple_citations(self):
        text = "See [DOC-AUTH-001] and [DOC-DEPLOY-002] for steps."
        assert _extract_citations(text) == ["DOC-AUTH-001", "DOC-DEPLOY-002"]

    def test_duplicate_citations_deduplicated(self):
        text = "First [DOC-AUTH-001], then again [DOC-AUTH-001]."
        assert _extract_citations(text) == ["DOC-AUTH-001"]

    def test_preserves_first_appearance_order(self):
        text = "[DOC-DEPLOY-002] comes before [DOC-AUTH-001] here."
        assert _extract_citations(text) == ["DOC-DEPLOY-002", "DOC-AUTH-001"]

    def test_no_citations(self):
        text = "This response has no document references."
        assert _extract_citations(text) == []

    def test_malformed_citations_ignored(self):
        """Citations that don't match the DOC-XXX-NNN pattern are skipped."""
        text = "See [AUTH-001] and [doc-auth-001] and [DOC-AUTH-1]."
        assert _extract_citations(text) == []

    def test_citation_in_multiline_text(self):
        text = (
            "Hello,\n\n"
            "Please check [DOC-DEPLOY-001].\n"
            "Also see [DOC-PERF-003] for performance.\n\n"
            "Best regards"
        )
        assert _extract_citations(text) == ["DOC-DEPLOY-001", "DOC-PERF-003"]


# ---------------------------------------------------------------------------
# Cannot-answer detection tests
# ---------------------------------------------------------------------------

class TestIndicatesCannotAnswer:
    """Tests for _indicates_cannot_answer — graceful degradation detection."""

    def test_clear_answer_returns_false(self):
        text = "Here are the steps to fix your deployment issue."
        assert _indicates_cannot_answer(text) is False

    def test_unable_to_find_returns_true(self):
        text = "I was unable to find information about this in our docs."
        assert _indicates_cannot_answer(text) is True

    def test_escalate_phrase_returns_true(self):
        text = "Let me escalate this to a specialist who can help."
        assert _indicates_cannot_answer(text) is True

    def test_not_covered_returns_true(self):
        text = "This topic is not covered in our documentation."
        assert _indicates_cannot_answer(text) is True

    def test_case_insensitive(self):
        text = "I WASN'T ABLE TO FIND specific documentation."
        assert _indicates_cannot_answer(text) is True

    def test_refer_to_specialist_returns_true(self):
        text = "I'll refer this to a specialist for further review."
        assert _indicates_cannot_answer(text) is True


# ---------------------------------------------------------------------------
# Context block formatting tests
# ---------------------------------------------------------------------------

class TestFormatContextBlock:
    """Tests for _format_context_block — turns chunks into prompt context."""

    def test_empty_retrieval(self):
        retrieval = RetrievalResult(query="test", chunks=[])
        result = _format_context_block(retrieval)
        assert "No relevant documentation found" in result

    def test_single_chunk_formatting(self):
        retrieval = make_retrieval_result(num_chunks=1)
        result = _format_context_block(retrieval)
        assert "Document 1" in result
        assert "DOC-DEPLOY-001" in result
        assert "Deployment Guide 1" in result
        assert "Resolution" in result
        assert "Check your Dockerfile" in result

    def test_multiple_chunks_numbered(self):
        retrieval = make_retrieval_result(num_chunks=3)
        result = _format_context_block(retrieval)
        assert "Document 1" in result
        assert "Document 2" in result
        assert "Document 3" in result

    def test_includes_relevance_score(self):
        retrieval = make_retrieval_result(num_chunks=1)
        result = _format_context_block(retrieval)
        # The relevance score should appear (formatted to 2 decimal places)
        assert "Relevance:" in result


# ---------------------------------------------------------------------------
# GeneratedResponse model tests
# ---------------------------------------------------------------------------

class TestGeneratedResponseModel:
    """Tests for the GeneratedResponse Pydantic model."""

    def test_defaults(self):
        r = GeneratedResponse(response_text="Hello")
        assert r.is_automated is True
        assert r.prompt_version == "v1.0"
        assert r.cited_doc_ids == []
        assert r.could_answer is True
        assert r.model_used == ""

    def test_all_fields(self):
        r = GeneratedResponse(
            response_text="Here's your answer [DOC-AUTH-001].",
            cited_doc_ids=["DOC-AUTH-001"],
            is_automated=True,
            prompt_version="v1.0",
            model_used="llama-3.1-8b-instant",
            reasoning="Generated from 3 chunks.",
            could_answer=True,
        )
        assert r.cited_doc_ids == ["DOC-AUTH-001"]
        assert r.model_used == "llama-3.1-8b-instant"


# ---------------------------------------------------------------------------
# generate_response integration tests (mocked LLM)
# ---------------------------------------------------------------------------

class TestGenerateResponse:
    """Tests for generate_response with a mocked LLM client."""

    def test_basic_generation(self):
        """generate_response returns a valid GeneratedResponse."""
        mock_response = (
            "Hello,\n\n"
            "Based on our documentation [DOC-DEPLOY-001], you should check "
            "your Dockerfile health check configuration.\n\n"
            "---\n"
            "This is an automated response from CloudServe Support."
        )
        client = make_mock_client(mock_response)
        ticket = make_ticket()
        classification = make_classification()
        retrieval = make_retrieval_result()

        result = generate_response(ticket, classification, retrieval, client=client)

        assert isinstance(result, GeneratedResponse)
        assert "DOC-DEPLOY-001" in result.cited_doc_ids
        assert result.is_automated is True
        assert result.could_answer is True
        assert len(result.response_text) > 0

    def test_prompt_includes_ticket_info(self):
        """The prompt sent to the LLM contains ticket subject and body."""
        client = make_mock_client("Test response.")
        ticket = make_ticket(subject="Billing error", body="I was charged twice")
        classification = make_classification(intent="billing_query")
        retrieval = make_retrieval_result()

        generate_response(ticket, classification, retrieval, client=client)

        # Check what was sent to the LLM
        call_args = client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_prompt = messages[1]["content"]
        assert "Billing error" in user_prompt
        assert "I was charged twice" in user_prompt
        assert "billing_query" in user_prompt

    def test_prompt_includes_context_block(self):
        """The prompt sent to the LLM contains the retrieved documentation."""
        client = make_mock_client("Test response.")
        ticket = make_ticket()
        classification = make_classification()
        retrieval = make_retrieval_result()

        generate_response(ticket, classification, retrieval, client=client)

        call_args = client.chat.completions.create.call_args
        user_prompt = call_args.kwargs["messages"][1]["content"]
        assert "DOC-DEPLOY-001" in user_prompt
        assert "Deployment Guide 1" in user_prompt

    def test_could_answer_false_when_escalation_detected(self):
        """could_answer is False when the LLM says it can't find the answer."""
        escalation_response = (
            "I wasn't able to find a specific answer in our documentation. "
            "Let me escalate this to a specialist."
        )
        client = make_mock_client(escalation_response)
        ticket = make_ticket()
        classification = make_classification()
        retrieval = make_retrieval_result()

        result = generate_response(ticket, classification, retrieval, client=client)
        assert result.could_answer is False

    def test_temperature_is_low(self):
        """Generation uses temperature 0.3 — creative enough but grounded."""
        client = make_mock_client("Response.")
        ticket = make_ticket()
        classification = make_classification()
        retrieval = make_retrieval_result()

        generate_response(ticket, classification, retrieval, client=client)

        call_args = client.chat.completions.create.call_args
        assert call_args.kwargs["temperature"] == 0.3

    def test_max_tokens_is_reasonable(self):
        """max_tokens is set to keep responses concise."""
        client = make_mock_client("Response.")
        ticket = make_ticket()
        classification = make_classification()
        retrieval = make_retrieval_result()

        generate_response(ticket, classification, retrieval, client=client)

        call_args = client.chat.completions.create.call_args
        assert call_args.kwargs["max_tokens"] == 1000

    def test_multiple_citations_extracted(self):
        """Multiple citations in the response are all captured."""
        multi_cite_response = (
            "Check your auth config [DOC-AUTH-001]. "
            "For deployment, see [DOC-DEPLOY-002]. "
            "Also review [DOC-PERF-003] for performance."
        )
        client = make_mock_client(multi_cite_response)
        ticket = make_ticket()
        classification = make_classification()
        retrieval = make_retrieval_result()

        result = generate_response(ticket, classification, retrieval, client=client)
        assert result.cited_doc_ids == ["DOC-AUTH-001", "DOC-DEPLOY-002", "DOC-PERF-003"]

    def test_reasoning_field_populated(self):
        """The reasoning field records chunk and citation counts."""
        client = make_mock_client("Response [DOC-DEPLOY-001].")
        ticket = make_ticket()
        classification = make_classification()
        retrieval = make_retrieval_result(num_chunks=3)

        result = generate_response(ticket, classification, retrieval, client=client)
        assert "3 retrieved chunks" in result.reasoning
        assert "1 citations" in result.reasoning


# ---------------------------------------------------------------------------
# GENERATION_PROMPT template tests
# ---------------------------------------------------------------------------

class TestGenerationPrompt:
    """Tests for the prompt template itself."""

    def test_prompt_has_required_placeholders(self):
        """The prompt template contains all required format placeholders."""
        required = ["{subject}", "{body}", "{channel}", "{customer_tier}",
                    "{intent}", "{urgency}", "{context_block}"]
        for placeholder in required:
            assert placeholder in GENERATION_PROMPT, f"Missing {placeholder}"

    def test_prompt_instructs_grounded_generation(self):
        """The prompt tells the LLM to ONLY use provided documentation."""
        assert "ONLY" in GENERATION_PROMPT
        assert "provided documentation" in GENERATION_PROMPT.lower() or \
               "documentation provided" in GENERATION_PROMPT.lower()

    def test_prompt_requires_citation_format(self):
        """The prompt specifies the [DOC-ID] citation format."""
        assert "[doc_id]" in GENERATION_PROMPT.lower() or \
               "[DOC-" in GENERATION_PROMPT

    def test_prompt_flags_automated_response(self):
        """The prompt instructs adding the automated response footer."""
        assert "automated response" in GENERATION_PROMPT.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

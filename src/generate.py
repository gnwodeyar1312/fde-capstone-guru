"""
Generate module for CloudServe Support System.

This is the FIFTH stage of the pipeline: Ingest → Classify → Retrieve → Route → **Generate** → Validate

Purpose:
    Draft a customer-facing support response using retrieved documentation.
    This module is ONLY called for tickets routed to auto_respond.
    Escalated tickets skip this stage entirely.

Design decisions:
    1. The prompt enforces grounded generation.
       Why? The LLM must answer FROM the retrieved docs, not from its
       training data. If the docs don't contain the answer, the response
       must say so. This prevents hallucination — the #1 risk in RAG systems.
    2. Citations use real doc_ids from the retrieved chunks.
       Why? Every claim in the response must trace back to a source article.
       This satisfies Marcus's requirement for explainability ("engineers
       screenshot and share publicly") and Ravi's request for honesty
       ("cite sources, show confidence, flag that it is automated").
    3. The response is flagged as automated.
       Why? Ravi explicitly said he calibrates trust differently for human
       vs. machine responses. Transparency builds trust. Hiding that a
       response is automated erodes it.
    4. We pass the classification result to the generator.
       Why? The generator can use the intent to focus its response. A
       billing_query gets a billing-focused response even if the retrieval
       returned some deployment docs (which can happen at lower relevance).
    5. Temperature is kept low (0.3) but not at 0.1.
       Why? Classification needs determinism (0.1). Generation needs a
       bit more creativity to write natural-sounding responses, but not
       so much that it starts inventing information.
    6. We use LangChain's ChatOpenAI for LLM generation.
       Why? Consistent interface across providers (OpenRouter, Groq, etc.)
       and integrates with the LangGraph pipeline orchestration.

Interview context:
    "What happens if the retrieved docs don't answer the question?"
    → The prompt instructs the LLM to say so explicitly: "I wasn't able
      to find a specific answer in our documentation for this. Let me
      escalate this to a specialist." This is a GRACEFUL DEGRADATION
      pattern — better to admit uncertainty than hallucinate an answer.
    "How do you prevent hallucinated citations?"
    → The prompt provides the exact doc_ids available. The guardrails
      stage (next) validates that every cited doc_id actually exists in
      the retrieved set. If a citation is hallucinated, the guardrail
      catches it.
    "Why not use a template-based approach?"
    → 22 intent categories, each with variable sub-problems. Templates
      would need 100+ variants and still couldn't handle novel phrasings.
      The LLM generates natural responses that match the customer's
      specific situation while staying grounded in documentation.
"""

import logging
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel

from src.classify import ClassificationResult
from src.config import MODEL_NAME, get_llm
from src.ingest import StandardTicket
from src.retrieve import RetrievalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic model for generation output — the CONTRACT
# ---------------------------------------------------------------------------


class GeneratedResponse(BaseModel):
    """
    The output of the generate stage.

    Contains the draft response, citations, and metadata about
    how the response was generated.
    """

    response_text: str
    cited_doc_ids: list[str] = []
    is_automated: bool = True
    prompt_version: str = "v1.0"
    model_used: str = ""
    reasoning: str = ""
    could_answer: bool = True  # False if the docs didn't contain the answer


# ---------------------------------------------------------------------------
# The generation prompt — PR-02 in the Prompt Library
# ---------------------------------------------------------------------------

GENERATION_PROMPT = """You are a helpful customer support agent for CloudServe Solutions, a cloud infrastructure provider.

Draft a response to the following support ticket using ONLY the documentation provided below.

## Rules:
1. ONLY use information from the provided documentation chunks. Do NOT use any other knowledge.
2. If the documentation does not contain enough information to fully answer the question, say so honestly and suggest the ticket will be reviewed by a specialist.
3. Cite your sources using the format [doc_id] (e.g., [DOC-AUTH-001]) after any claim that comes from a specific document.
4. Be professional, empathetic, and concise. Use the customer's name if available.
5. Address the specific issue described in the ticket, not generic advice.
6. Include concrete steps the customer can take (numbered list if multiple steps).
7. End with an offer to help further.
8. Flag that this is an automated response at the end.

## Customer Ticket:
<ticket>
Subject: {subject}
Body: {body}
Channel: {channel}
Customer Tier: {customer_tier}
</ticket>

## Classification:
Intent: {intent}
Urgency: {urgency}

## Retrieved Documentation (use ONLY these as sources):
{context_block}

## Response Format:
Write ONLY the response text. Do not include any JSON, headers, or metadata.
Start directly with the greeting. End with:
"---
This is an automated response from CloudServe Support. If this doesn't resolve your issue, a team member will follow up."
"""


# ---------------------------------------------------------------------------
# Helper: format retrieved chunks into a context block
# ---------------------------------------------------------------------------


def _format_context_block(retrieval: RetrievalResult) -> str:
    """
    Format retrieved chunks into a readable context block for the prompt.

    Each chunk is labeled with its doc_id, title, and section so the LLM
    can cite specific sources.
    """
    if not retrieval.chunks:
        return "(No relevant documentation found.)"

    blocks = []
    for i, chunk in enumerate(retrieval.chunks):
        block = (
            f"--- Document {i + 1} ---\n"
            f"Doc ID: {chunk.doc_id}\n"
            f"Title: {chunk.title}\n"
            f"Section: {chunk.section_name}\n"
            f"Applies to: {chunk.applies_to}\n"
            f"Relevance: {chunk.relevance_score:.2f}\n"
            f"\n{chunk.content}\n"
        )
        blocks.append(block)

    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------


def generate_response(
    ticket: StandardTicket,
    classification: ClassificationResult,
    retrieval: RetrievalResult,
    llm: ChatOpenAI | None = None,
) -> GeneratedResponse:
    """
    Generate a support response for a ticket using retrieved documentation.

    This function:
    1. Builds the generation prompt with ticket, classification, and context
    2. Sends it to the LLM via LangChain's ChatOpenAI interface
    3. Extracts cited doc_ids from the response
    4. Returns the validated GeneratedResponse

    Args:
        ticket: The StandardTicket from ingest
        classification: The ClassificationResult from classify
        retrieval: The RetrievalResult from retrieve
        llm: Optional pre-configured ChatOpenAI instance

    Returns:
        GeneratedResponse with the draft text and metadata

    Raises:
        Exception: If the API call fails
    """
    if llm is None:
        llm = get_llm()

    # Build the context block from retrieved chunks
    context_block = _format_context_block(retrieval)

    # Build the prompt
    prompt = GENERATION_PROMPT.format(
        subject=ticket.subject,
        body=ticket.body,
        channel=ticket.channel,
        customer_tier=ticket.customer_tier,
        intent=classification.intent,
        urgency=classification.urgency,
        context_block=context_block,
    )

    logger.info(
        "Generating response for ticket %s (intent=%s)",
        ticket.ticket_id,
        classification.intent,
    )

    # Call the LLM via LangChain
    messages = [
        SystemMessage(
            content=(
                "You are a professional CloudServe support agent. "
                "Write clear, helpful responses grounded in the provided documentation. "
                "Always cite sources with [DOC-ID] format."
            ),
        ),
        HumanMessage(content=prompt),
    ]

    # Override temperature for generation (slightly more creative than classification)
    response = llm.invoke(messages, temperature=0.3, max_tokens=1000)

    raw_response = response.content.strip()
    logger.debug("Raw generation for %s: %s", ticket.ticket_id, raw_response[:200])

    # Extract cited doc_ids from the response text
    cited_ids = _extract_citations(raw_response)

    # Determine if the response indicates it couldn't answer
    could_answer = not _indicates_cannot_answer(raw_response)

    result = GeneratedResponse(
        response_text=raw_response,
        cited_doc_ids=cited_ids,
        is_automated=True,
        prompt_version="v1.0",
        model_used=MODEL_NAME,
        reasoning=f"Generated from {len(retrieval.chunks)} retrieved chunks, {len(cited_ids)} citations used.",
        could_answer=could_answer,
    )

    logger.info(
        "Generated response for %s: %d chars, %d citations (%s), could_answer=%s",
        ticket.ticket_id,
        len(result.response_text),
        len(result.cited_doc_ids),
        result.cited_doc_ids,
        result.could_answer,
    )

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_citations(text: str) -> list[str]:
    """
    Extract doc_id citations from the response text.

    Looks for patterns like [DOC-AUTH-001], [DOC-DEPLOY-002], etc.
    Returns unique doc_ids in the order they first appear.
    """
    pattern = r"\[(DOC-[A-Z]+-\d{3})\]"
    matches = re.findall(pattern, text)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for doc_id in matches:
        if doc_id not in seen:
            seen.add(doc_id)
            unique.append(doc_id)
    return unique


def _indicates_cannot_answer(text: str) -> bool:
    """
    Check if the response indicates it couldn't answer from docs.

    Looks for phrases that signal the LLM acknowledged it couldn't
    find the answer in the provided documentation.
    """
    cannot_answer_phrases = [
        "wasn't able to find",
        "unable to find",
        "not covered in",
        "documentation does not",
        "no specific documentation",
        "couldn't find relevant",
        "not available in our documentation",
        "escalate this to",
        "refer this to a specialist",
        "beyond what our documentation covers",
    ]
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in cannot_answer_phrases)

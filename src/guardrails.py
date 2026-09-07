"""
Guardrails module for CloudServe Support System.

This is the SIXTH stage of the pipeline: Ingest → Classify → Retrieve → Route → Generate → **Validate**

Purpose:
    Validate a generated response before it reaches the customer.
    This is the LAST LINE OF DEFENSE — if a response passes guardrails,
    it gets sent. If it fails, it gets escalated to a human.

Design decisions:
    1. Each guardrail is an independent check function.
       Why? Separation of concerns. Each check tests ONE thing and returns
       a clear pass/fail with a reason. This makes it easy to add new
       guardrails without touching existing ones.
    2. ALL guardrails run, even if one fails early.
       Why? We want the full picture. If a response has a hallucinated
       citation AND a fabricated URL, the escalation reason should mention
       both, not just the first one found.
    3. Guardrails are DETERMINISTIC — no LLM calls.
       Why? The whole point is to catch LLM mistakes. Using an LLM to
       validate an LLM's output adds more unpredictability, not less.
       Pattern matching and set comparison are reliable.
    4. Failed guardrails trigger escalation, not silent fixing.
       Why? Auto-fixing is tempting ("just remove the bad URL") but
       dangerous. The response was generated as a coherent whole — removing
       a part might make the rest incoherent or misleading. Better to
       escalate and let a human rewrite it.
    5. The guardrail result carries structured metadata.
       Why? The decision log needs to record exactly which checks passed
       and which failed, with reasons. "Guardrail failed" is not debuggable.
       "citation_check failed: DOC-FAKE-999 not in retrieved set" is.

Interview context:
    "What if guardrails are too strict and block good responses?"
    → That's a tuning question we answer with data. Run the evaluation
      harness, measure the false-positive rate (good responses blocked),
      and adjust thresholds. Too strict is always safer than too loose —
      a blocked good response gets handled by a human (minor delay),
      while a bad response reaching a customer is a trust violation.
    "Why not use an LLM to judge response quality?"
    → LLM-as-judge is great for evaluation (we use it there). But for
      production guardrails, deterministic rules are more reliable.
      An LLM might "decide" a hallucinated URL looks fine because the
      domain name sounds plausible.
    "How do you handle edge cases?"
    → Each guardrail has explicit edge case handling. For example, the
      citation check handles the case where the response has NO citations
      (that's a separate check — a response grounded in docs should cite
      them). The URL check ignores mailto: links. The PII check uses
      patterns, not exact matching, so it catches formatted and
      unformatted phone numbers.
"""

import re
import logging
from typing import Optional

from pydantic import BaseModel, Field

from src.generate import GeneratedResponse
from src.retrieve import RetrievalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models for guardrail output — the CONTRACT
# ---------------------------------------------------------------------------

class GuardrailCheck(BaseModel):
    """Result of a single guardrail check."""
    name: str
    passed: bool
    reason: str
    severity: str = "high"  # high = blocks response, medium = warning


class GuardrailResult(BaseModel):
    """
    The output of the guardrails stage.

    Contains results of all checks and the final pass/fail decision.
    A response passes ONLY if ALL high-severity checks pass.
    """
    passed: bool = True
    checks: list[GuardrailCheck] = []
    failed_checks: list[str] = []
    warnings: list[str] = []
    recommendation: str = ""

    def model_post_init(self, __context):
        self.failed_checks = [c.name for c in self.checks if not c.passed and c.severity == "high"]
        self.warnings = [c.name for c in self.checks if not c.passed and c.severity == "medium"]
        self.passed = len(self.failed_checks) == 0
        if not self.passed:
            self.recommendation = (
                f"Response blocked by guardrails: {', '.join(self.failed_checks)}. "
                "Escalate to human review."
            )
        elif self.warnings:
            self.recommendation = (
                f"Response passed with warnings: {', '.join(self.warnings)}. "
                "Safe to send but review recommended."
            )
        else:
            self.recommendation = "All checks passed. Safe to send."


# ---------------------------------------------------------------------------
# Individual guardrail checks
# ---------------------------------------------------------------------------

def _check_citations(
    response: GeneratedResponse,
    retrieval: RetrievalResult,
) -> GuardrailCheck:
    """
    Guardrail 1: Verify all cited doc_ids exist in the retrieved set.

    A hallucinated citation — one that references a doc_id not in the
    retrieved chunks — is a critical failure. It means the LLM invented
    a source, which destroys trust in the entire response.

    Also warns if a grounded response has zero citations (suspicious
    but not necessarily wrong — some responses are procedural).
    """
    retrieved_doc_ids = set(retrieval.unique_doc_ids)
    cited_doc_ids = set(response.cited_doc_ids)

    # Check for hallucinated citations (cited but not retrieved)
    hallucinated = cited_doc_ids - retrieved_doc_ids

    if hallucinated:
        return GuardrailCheck(
            name="citation_validation",
            passed=False,
            reason=(
                f"Hallucinated citation(s): {sorted(hallucinated)}. "
                f"These doc_ids were cited but not in the retrieved set "
                f"({sorted(retrieved_doc_ids)})."
            ),
            severity="high",
        )

    # Warn if no citations at all (but docs were retrieved)
    if not cited_doc_ids and retrieval.chunks:
        return GuardrailCheck(
            name="citation_validation",
            passed=False,
            reason=(
                "Response contains no citations despite having retrieved "
                f"{len(retrieval.chunks)} documentation chunks. "
                "A grounded response should cite its sources."
            ),
            severity="medium",
        )

    return GuardrailCheck(
        name="citation_validation",
        passed=True,
        reason=f"All {len(cited_doc_ids)} citation(s) verified against retrieved set.",
    )


def _check_no_fabricated_urls(response: GeneratedResponse) -> GuardrailCheck:
    """
    Guardrail 2: Check for fabricated URLs in the response.

    LLMs sometimes invent plausible-looking URLs. Any URL in the response
    is flagged because our documentation doesn't contain customer-facing
    URLs — all guidance is procedural ("go to Settings > API Keys"), not
    link-based.

    Exception: mailto: links are ignored (the LLM might reference
    support@cloudserve.com which is fine).
    """
    # Match http/https URLs
    url_pattern = r'https?://[^\s\)\]\"\'<>]+'
    urls_found = re.findall(url_pattern, response.response_text)

    # Filter out mailto-style links (not URLs)
    # Also allow cloudserve documentation references if they appear in docs
    suspicious_urls = [u for u in urls_found if not u.startswith("mailto:")]

    if suspicious_urls:
        return GuardrailCheck(
            name="no_fabricated_urls",
            passed=False,
            reason=(
                f"Found {len(suspicious_urls)} URL(s) in response: {suspicious_urls}. "
                "The LLM may have fabricated these. CloudServe documentation "
                "uses procedural instructions, not direct links."
            ),
            severity="high",
        )

    return GuardrailCheck(
        name="no_fabricated_urls",
        passed=True,
        reason="No URLs found in response.",
    )


def _check_no_pii_leakage(response: GeneratedResponse) -> GuardrailCheck:
    """
    Guardrail 3: Check for PII patterns in the response.

    The response should not echo back sensitive customer data like
    email addresses, phone numbers, or credit card numbers. Even if
    the ticket contained them, the response shouldn't repeat them.

    This uses pattern matching — it may have false positives (e.g.,
    example.com in a generic instruction) but false positives are
    safe (they just trigger a review), while false negatives are not.
    """
    pii_patterns = {
        "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        "phone": r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        "ssn": r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
    }

    # Allowlisted patterns (support emails that are fine to include)
    allowed_emails = {
        "support@cloudserve.com",
        "security@cloudserve.com",
        "billing@cloudserve.com",
        "noreply@cloudserve.com",
    }

    found_pii = []
    for pii_type, pattern in pii_patterns.items():
        matches = re.findall(pattern, response.response_text)
        for match in matches:
            if pii_type == "email" and match.lower() in allowed_emails:
                continue
            found_pii.append((pii_type, match))

    if found_pii:
        # Redact the actual values in the reason (don't log PII!)
        redacted = [(t, f"{v[:3]}***") for t, v in found_pii]
        return GuardrailCheck(
            name="no_pii_leakage",
            passed=False,
            reason=(
                f"Potential PII found in response: {redacted}. "
                "Automated responses should not echo back sensitive data."
            ),
            severity="high",
        )

    return GuardrailCheck(
        name="no_pii_leakage",
        passed=True,
        reason="No PII patterns detected in response.",
    )


def _check_response_length(response: GeneratedResponse) -> GuardrailCheck:
    """
    Guardrail 4: Check response length is within acceptable bounds.

    Too short (<50 chars): Likely a failed generation or empty response.
    Too long (>3000 chars): Likely rambling or repeating content.

    These thresholds are generous — a typical good response is 200-1500 chars.
    """
    length = len(response.response_text)

    if length < 50:
        return GuardrailCheck(
            name="response_length",
            passed=False,
            reason=f"Response too short ({length} chars). Minimum is 50.",
            severity="high",
        )

    if length > 3000:
        return GuardrailCheck(
            name="response_length",
            passed=False,
            reason=f"Response too long ({length} chars). Maximum is 3000.",
            severity="medium",  # Warning, not blocking — some responses are legitimately detailed
        )

    return GuardrailCheck(
        name="response_length",
        passed=True,
        reason=f"Response length ({length} chars) is within bounds (50-3000).",
    )


def _check_automated_footer(response: GeneratedResponse) -> GuardrailCheck:
    """
    Guardrail 5: Verify the automated response footer is present.

    Ravi's requirement: customers must know when they're reading an
    automated response. The footer "This is an automated response"
    must appear somewhere in the response text.
    """
    footer_phrases = [
        "automated response",
        "auto-generated",
        "automatically generated",
    ]

    text_lower = response.response_text.lower()
    has_footer = any(phrase in text_lower for phrase in footer_phrases)

    if not has_footer:
        return GuardrailCheck(
            name="automated_footer",
            passed=False,
            reason="Response is missing the automated response disclosure footer.",
            severity="medium",  # Warning — the prompt should add it, but the footer can be appended
        )

    return GuardrailCheck(
        name="automated_footer",
        passed=True,
        reason="Automated response footer is present.",
    )


# ---------------------------------------------------------------------------
# Core guardrails function
# ---------------------------------------------------------------------------

def validate_response(
    response: GeneratedResponse,
    retrieval: RetrievalResult,
) -> GuardrailResult:
    """
    Run all guardrail checks on a generated response.

    This is the final validation gate. ALL checks run regardless of
    individual results — we want the complete picture for logging.

    Args:
        response: The GeneratedResponse from the generate stage
        retrieval: The RetrievalResult (needed for citation validation)

    Returns:
        GuardrailResult with pass/fail and detailed check results
    """
    checks = [
        _check_citations(response, retrieval),
        _check_no_fabricated_urls(response),
        _check_no_pii_leakage(response),
        _check_response_length(response),
        _check_automated_footer(response),
    ]

    result = GuardrailResult(checks=checks)

    logger.info(
        "Guardrails: %s | %d/%d checks passed, %d failed, %d warnings",
        "PASSED" if result.passed else "FAILED",
        sum(1 for c in checks if c.passed),
        len(checks),
        len(result.failed_checks),
        len(result.warnings),
    )

    if result.failed_checks:
        logger.warning("Failed checks: %s", result.failed_checks)
    if result.warnings:
        logger.info("Warnings: %s", result.warnings)

    return result

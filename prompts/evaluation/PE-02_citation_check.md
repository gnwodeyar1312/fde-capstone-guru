# PE-02: Citation Accuracy Check

**Version:** v1.0  
**Purpose:** Validate that generated responses cite real, relevant documentation  
**Used in:** `src/guardrails.py` → `_check_citation_validity()`  
**Type:** Programmatic validation (deterministic — no LLM call)  

## Requirement Traceability

| Requirement | Source |
|-------------|--------|
| FR-05: Cite sources using doc_id format | PRD §4.1 |
| A5: Grounded generation with citations | Build Spec §Acceptance Criteria |
| A7: Guardrails block fabricated content | Build Spec §Acceptance Criteria |
| Governance: No hallucinated citations | Governance Framework §Guardrails |

## What This Check Does

The citation check is a **guardrail** that runs after response generation (Stage 6: Validate). It ensures every document citation in the response refers to a real document that was actually retrieved for this ticket.

### Check Logic

```python
def _check_citation_validity(generation, retrieval):
    """
    Validate cited doc_ids exist in the retrieved set.
    
    1. Extract all [DOC-XXX-NNN] citations from response_text
    2. Get the set of doc_ids from retrieved chunks
    3. Flag any citation not in the retrieved set as fabricated
    """
    cited_ids = generation.cited_doc_ids
    retrieved_ids = {chunk.doc_id for chunk in retrieval.chunks}
    
    fabricated = [cid for cid in cited_ids if cid not in retrieved_ids]
    
    if fabricated:
        return CheckResult(
            passed=False,
            reason=f"Fabricated citations: {fabricated}",
            severity="high"
        )
    return CheckResult(passed=True, reason="All citations valid")
```

### Why This Matters

Hallucinated citations are a critical failure mode in RAG systems. A response that cites "DOC-AUTH-005" when that document was never retrieved (or doesn't exist) undermines trust. Ravi (stakeholder) explicitly said: "engineers screenshot and share publicly" — a fabricated citation shared publicly would damage CloudServe's credibility.

## Citation Extraction Pattern

Citations are extracted using regex from the response text:

```
Pattern: \[(DOC-[A-Z]+-\d{3})\]
Examples matched: [DOC-AUTH-001], [DOC-DEPLOY-002], [DOC-PERF-003]
Examples NOT matched: [AUTH-001], [doc-auth-001], [DOC-AUTH-1]
```

Duplicates are removed while preserving first-appearance order.

## Evaluation Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| Citation validity rate | % of responses with all citations valid | 100% |
| Fabricated citation rate | % of citations pointing to non-retrieved docs | 0% |
| Citation coverage | % of responses with ≥1 citation | ≥90% for auto_respond |
| Expected doc overlap | % of ground_truth expected_doc_ids cited | ≥60% |

## Failure Handling

When this check fails:
- The guardrail blocks the response (`passed=False`)
- `final_action` becomes `"blocked_by_guardrails"`
- The ticket is escalated to a human agent
- The failed check is logged in the decision record

This is a **hard block** — fabricated citations are never sent to customers.

## Version History

| Version | Date | Change | Reason |
|---------|------|--------|--------|
| v1.0 | 2026-08-28 | Initial citation check | Prevent hallucinated references |

# Prompt Register

All prompts are versioned, documented, and traceable to requirements.  
Prompts are design artefacts — they get version control like everything else.

## Build Prompts (used inside the system)

| ID | File | Purpose | Version | Requirement | Temperature |
|----|------|---------|---------|-------------|-------------|
| PR-01 | `build/PR-01_classification.md` | Classify tickets by intent, urgency, doc-answerability | v1.0 | FR-01, FR-02, FR-03, A2 | 0.1 |
| PR-02 | `build/PR-02_generation.md` | Generate grounded, cited customer responses | v1.0 | FR-04, FR-05, FR-06, A5, A6 | 0.3 |

**Note on PR-03 and PR-04:** The guardrail checks (grounding check, tone/scope check) are implemented as deterministic programmatic rules in `src/guardrails.py`, not as LLM prompts. This is a deliberate design choice — guardrails that rely on an LLM to judge another LLM's output introduce a second point of failure. Deterministic checks are faster, cheaper, and fully reproducible.

## Evaluation Prompts (used to judge output)

| ID | File | Purpose | Version | Requirement | Type |
|----|------|---------|---------|-------------|------|
| PE-01 | `evaluation/PE-01_quality_assessment.md` | Evaluate response quality across 5 dimensions | v1.0 | A9, A10 | Programmatic |
| PE-02 | `evaluation/PE-02_citation_check.md` | Validate citation accuracy in generated responses | v1.0 | FR-05, A5, A7 | Programmatic |

## Design Decisions

1. **Two LLM prompts, not four.** Only classification and generation use LLM calls. Routing is deterministic rules. Guardrails are deterministic checks. This reduces API cost, latency, and failure surface.

2. **Evaluation is programmatic.** All metrics are computed from the evaluation harness output JSON without additional LLM calls. This ensures reproducibility — running the same evaluation twice on the same results file produces identical metrics.

3. **Temperature choices.** Classification uses 0.1 (near-deterministic) because consistent labelling matters more than creative expression. Generation uses 0.3 — enough for natural prose but low enough to stay grounded in the provided documentation.

## Traceability Summary

| Requirement | Prompt(s) |
|-------------|-----------|
| FR-01: Intent classification | PR-01 |
| FR-02: Urgency assessment | PR-01 |
| FR-03: Doc-answerability check | PR-01 |
| FR-04: Grounded response generation | PR-02 |
| FR-05: Source citation | PR-02, PE-02 |
| FR-06: Automated response flagging | PR-02 |
| FR-07: Graceful degradation | PR-02 |
| A2: 22-intent classification | PR-01 |
| A5: Grounded generation with citations | PR-02, PE-02 |
| A6: Escalation when docs insufficient | PR-02 |
| A7: Guardrails block fabricated content | PE-02 |
| A9: Full unattended evaluation | PE-01 |
| A10: Business metrics | PE-01 |

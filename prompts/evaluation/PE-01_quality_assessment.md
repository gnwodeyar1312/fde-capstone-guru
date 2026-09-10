# PE-01: Response Quality Assessment

**Version:** v1.0  
**Purpose:** Evaluate the quality of generated support responses against ground truth  
**Used in:** `src/monitoring.py` → `compute_metrics()`  
**Type:** Programmatic evaluation (no LLM call — deterministic metric computation)  

## Requirement Traceability

| Requirement | Source |
|-------------|--------|
| A9: Full unattended evaluation run | Build Spec §Acceptance Criteria |
| A10: Business metrics alongside technical | Build Spec §Acceptance Criteria |
| Evaluation Tier 1: Component metrics | Evaluation Framework §Tiers |
| Evaluation Tier 2: Pipeline metrics | Evaluation Framework §Tiers |
| Evaluation Tier 3: Business metrics | Evaluation Framework §Tiers |

## Quality Dimensions Assessed

### 1. Classification Quality
- **Intent accuracy:** Does `predictions.intent` match `ground_truth.intent`?
- **Urgency accuracy:** Does `predictions.urgency` match `ground_truth.urgency`?
- **Per-class precision/recall:** Breakdown by intent category
- **Confidence calibration:** Do stated confidence scores match observed accuracy?

### 2. Routing Quality
- **Route accuracy:** Does `route.action` match `ground_truth.expected_route`?
- **Safe escalation rate:** % of must_not_auto_respond tickets correctly escalated
- **False escalation rate:** % of auto_respond tickets unnecessarily escalated

### 3. Retrieval Quality
- **Retrieval hit rate:** % of tickets with ≥1 relevant chunk retrieved
- **Citation overlap:** % of expected doc_ids appearing in cited_doc_ids
- **Mean chunks retrieved:** Average number of chunks per ticket

### 4. Generation Quality
- **Could-answer rate:** % of auto_respond tickets where generation found sufficient docs
- **Citation count:** Mean citations per response
- **Response length:** Character count distribution

### 5. Guardrail Quality
- **Pass rate:** % of generated responses passing all guardrail checks
- **Block rate by check:** Which guardrail checks block most often

## Metric Computation

All metrics are computed programmatically from the evaluation harness output JSON. No LLM is used for evaluation — this ensures reproducibility and eliminates evaluator bias.

```python
# Example: Intent accuracy
intent_correct = sum(1 for r in results
    if r["status"] == "success"
    and r["predictions"]["intent"] == r["ground_truth"]["intent"])
accuracy = intent_correct / total_successful
```

## Thresholds (from Build Spec & Evaluation Framework)

| Metric | Target | Rationale |
|--------|--------|-----------|
| Intent accuracy | ≥85% | Build Spec minimum |
| Route accuracy | ≥80% | Ensures correct auto-respond/escalate split |
| Guardrail pass rate | ≥90% | Most auto-respond tickets should pass |
| Citation overlap | ≥60% | At least one expected doc cited |
| Error rate | ≤5% | Pipeline reliability |

## Version History

| Version | Date | Change | Reason |
|---------|------|--------|--------|
| v1.0 | 2026-08-28 | Initial assessment framework | Baseline evaluation |

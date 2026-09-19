# Architecture

## System Overview

CloudServe Support Intelligence is an automated customer support pipeline that classifies incoming tickets, retrieves relevant documentation, and either generates a validated response or escalates to the appropriate human team. The system is built on LangGraph's StateGraph, giving each ticket a deterministic path through six processing stages.

## High-Level Pipeline

```
Ingest → Classify → Retrieve → Route ─┬─ AUTO_RESPOND → Generate → Validate → Sent
                                       └─ ESCALATE ───────────────────────────→ Escalated
```

Escalated tickets skip generation and validation entirely — there is no partial response for a ticket the system cannot safely handle.

## Pipeline State

Every ticket flows through a single `PipelineState` (TypedDict) that accumulates outputs from each stage:

| Key | Type | Set By |
|-----|------|--------|
| `ticket` | StandardTicket | Ingest |
| `llm` | ChatOpenAI | Pipeline init |
| `classification` | ClassificationResult | Classify |
| `retrieval` | RetrievalResult | Retrieve |
| `route_decision` | RouteDecision | Route |
| `generation` | GeneratedResponse | Generate |
| `guardrails` | GuardrailResult | Validate |
| `final_action` | str (`sent` / `escalated` / `blocked_by_guardrails`) | Terminal nodes |
| `error` | str | Any stage on failure |

The state key for routing is `route_decision` (not `route`) to avoid collision with the LangGraph node name.

## Stage Details

### 1. Ingest

Parses raw ticket data into a `StandardTicket` with Pydantic validation at the boundary. Ground-truth labels are separated at this point and never enter the pipeline.

**StandardTicket fields:** `ticket_id`, `channel` (email / chat / docs_comment / forum), `subject`, `body`, `received_at`, `customer_id`, `customer_name`, `customer_tier` (standard / business / enterprise), `customer_region`, `language_fluency`.

### 2. Classify

An LLM call (temperature=0.1, max_tokens=500) produces structured JSON output validated by Pydantic.

**Outputs:** intent (one of 22 categories), intent_confidence (0–1), urgency (low / medium / high), answerable_from_docs (bool), must_not_auto_respond (deterministic from intent), reasoning.

**22 valid intents:** account_access, api_key_issue, api_usage_question, authentication_failure, billing_query, compliance_request, configuration_help, data_export, data_residency, database_issue, deployment_failure, feature_request, integration_help, onboarding, performance_degradation, quota_or_overage, rate_limit, rollback_request, security_incident, sso_configuration, unclear_request, webhook_issue.

**Must-not-auto-respond intents** — four intents that always escalate regardless of confidence or answerability. This is a hardcoded business rule, not an LLM prediction:

| Intent | Escalation Target |
|--------|-------------------|
| compliance_request | compliance_team |
| security_incident | security_team |
| feature_request | product_team |
| unclear_request | tier1_support |

**Post-classification overrides** — keyword-based safety nets that catch edge cases the LLM misses:
- rollback_request + active breakage keywords → force answerable_from_docs=False
- database_issue + incident keywords → force answerable_from_docs=False

**Error fallback:** Returns `unclear_request` with 0.0 confidence, must_not_auto_respond=True (fails safe to escalation).

### 3. Retrieve

Searches a ChromaDB vector store built from 29 CloudServe documentation articles (`data/documentation.json`). Embeddings use the `all-MiniLM-L6-v2` sentence-transformer model.

**Configuration:** top-k chunks (default 5, configurable via `RETRIEVAL_TOP_K`).

**Output:** RetrievalResult containing ranked chunks (doc_id, title, section_name, content, relevance_score), unique_doc_ids, and num_results.

### 4. Route

Entirely deterministic — no LLM call. Rules evaluated in priority order:

1. `must_not_auto_respond` is True → **ESCALATE** (safety first)
2. Not answerable from docs → **ESCALATE**
3. Intent confidence below threshold (default 0.80) → **ESCALATE**
4. All checks pass → **AUTO_RESPOND**

**Output:** RouteDecision with action (AUTO_RESPOND / ESCALATE), reason, escalation_target, confidence_met, rule_triggered.

**Expected split:** ~62% auto-respond, ~38% escalate.

### 5. Generate

Only runs for AUTO_RESPOND tickets. LLM call with temperature=0.3, max_tokens=1000.

**Prompt constraints:** use only retrieved documentation, cite sources with `[DOC-ID]` format, include numbered steps, flag as automated response, graceful degradation if docs are insufficient.

**Cannot-answer detection:** Checks response text for phrases like "wasn't able to find" or "escalate this to" — if the LLM signals it cannot answer, the ticket is marked accordingly.

**Output:** GeneratedResponse with response_text, cited_doc_ids, is_automated=True, prompt_version, model_used, reasoning, could_answer.

### 6. Validate (Guardrails)

Five deterministic checks run on every generated response — no LLM involved. All checks execute regardless of individual failures.

| Check | Severity | What It Catches |
|-------|----------|-----------------|
| citation_validation | high (hallucinated) / medium (missing) | Cited doc_ids not in retrieved set; zero citations when docs were retrieved |
| no_fabricated_urls | high | Any http/https URLs in the response (CloudServe uses procedural instructions, not links) |
| no_pii_leakage | high | Email, phone, credit card, SSN patterns (allowlist for @cloudserve.com) |
| response_length | high (<50 chars) / medium (>3000 chars) | Responses too short to be useful or too long |
| automated_footer | medium | Missing "automated response" / "auto-generated" disclosure |

**Pass logic:** the response passes only if all high-severity checks pass. Medium-severity failures produce warnings but do not block sending.

## Conditional Graph Edges

The `should_generate` function is the conditional edge after the Route node:
- `route.is_auto_respond` → proceeds to "generate" → "validate" → END (final_action="sent" or "blocked_by_guardrails")
- Escalated → proceeds to "escalate" → END (final_action="escalated")

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Orchestration | LangChain + LangGraph (StateGraph) |
| LLM | Groq API (Llama 3.1 8B) via LangChain ChatOpenAI |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB |
| Data Validation | Pydantic v2 |
| Decision Log | SQLite (append-only) |
| API | FastAPI + Uvicorn |
| Monitoring | Prometheus + Grafana |
| Dashboard | Streamlit |
| Testing | pytest + pytest-cov |

## LLM Configuration

The system uses Groq's free-tier API through a LangChain `ChatOpenAI` interface (configured in `src/config.py` via `get_llm()`). The pipeline makes exactly two LLM calls per auto-responded ticket (classify + generate) and one per escalated ticket (classify only).

**Rate-limit handling:** The evaluation harness uses exponential backoff with jitter (base 5s, max 120s, up to 5 retries) and a 6-second delay between tickets to stay within Groq's free-tier limits (30 RPM, 200K tokens/day).

## Decision Logging

Every ticket processed is logged to a SQLite database (`storage/decisions.db`) with full traceability: ticket_id, timestamp, classification outputs, routing decision, retrieved doc IDs, generated response text, cited sources, guardrail results, and final_action. This is an append-only audit trail.

## Monitoring and Observability

### Prometheus Metrics

The FastAPI application exposes a `/metrics` endpoint scraped by Prometheus at 2-second intervals. Metrics include request counts, latencies, and pipeline stage outcomes.

### Alert Rules

Three alert rules are configured in `monitoring/prometheus/alert.rules.yml`:

| Alert | Condition | Severity |
|-------|-----------|----------|
| HighPipelineErrorRate | Error rate > 10% over 5m | critical |
| GuardrailFailureSpike | Guardrail failure rate > 20% over 5m | warning |
| HighRateLimitBackoff | Rate-limit backoff > 0.2/s over 5m | warning |

### Grafana Dashboard

A provisioned Grafana dashboard (`monitoring/grafana/dashboards/cloudserve_support_system.json`) visualizes pipeline throughput, latency percentiles, error rates, and guardrail pass/fail breakdowns.

**Docker deployment:** `docker-compose -f docker-compose.monitoring.yml up` — Prometheus at :9090, Grafana at :3000.

**Native Windows deployment:** `scripts/start_monitoring_windows.ps1` downloads portable Prometheus and Grafana, configures local provisioning (datasource pointing to localhost:9090, dashboard JSON copied into Grafana's directory), and launches both.

### Streamlit Dashboard

`streamlit run src/dashboard.py` provides a post-hoc analysis dashboard with KPI cards, per-intent accuracy charts, routing distributions, guardrail results, error breakdowns, and run comparisons.

## Evaluation Results

Across 4 evaluation runs (233 successfully processed tickets):

| Metric | Result |
|--------|--------|
| Intent Classification Accuracy | 96.7% |
| Macro F1 (22 intents) | 0.970 |
| Perfect-score intents | 18 of 22 |
| Must-not-auto-respond compliance | 100% |
| Guardrail pass rate | 96% |
| ECE (Expected Calibration Error) | 0.0053 |

## Design Principles

1. **Fail safe to escalation** — unknown intents, low confidence, and any classification error default to human escalation. No ticket is silently dropped.

2. **Deterministic routing** — the router uses no LLM. Business rules are explicit, auditable, and testable without API calls.

3. **Deterministic guardrails** — validation uses pattern matching, not LLM judgment. The guardrails catch LLM mistakes rather than introducing new ones.

4. **Safety-first escalation** — must-not-auto-respond is a hardcoded business rule applied after classification. The LLM cannot override it.

5. **Defense in depth** — post-classification keyword overrides add a second safety layer for edge cases the LLM misses (e.g., active database incidents that should not get a templated response).

6. **Full auditability** — every decision is logged to SQLite with reasoning. The evaluation harness scores predictions against ground truth without the pipeline ever seeing the labels.

7. **Separation of concerns** — ground truth is stripped at ingest and never enters the pipeline. Evaluation is separate from processing.

8. **Pydantic at every boundary** — all inter-stage data is validated by Pydantic models. Malformed data fails loudly at the stage that produced it.

## Directory Structure

```
├── src/
│   ├── api.py              # FastAPI application with /metrics endpoint
│   ├── classify.py          # Intent classification (LLM)
│   ├── config.py            # LLM, embedding, and system configuration
│   ├── dashboard.py         # Streamlit analytics dashboard
│   ├── generator.py         # Response generation (LLM)
│   ├── guardrails.py        # 5 deterministic validation checks
│   ├── ingest.py            # Ticket parsing and validation
│   ├── logging_store.py     # SQLite decision logging
│   ├── models.py            # Pydantic data models
│   ├── monitoring.py        # Batch evaluation metrics computation
│   ├── pipeline.py          # LangGraph StateGraph orchestration
│   ├── retriever.py         # ChromaDB vector search
│   └── router.py            # Deterministic routing rules
├── data/
│   ├── documentation.json   # 29 CloudServe knowledge articles
│   └── tickets.json         # Input ticket dataset
├── monitoring/
│   ├── prometheus/
│   │   ├── prometheus.yml   # Scrape configuration
│   │   └── alert.rules.yml  # 3 alert rules
│   └── grafana/
│       ├── dashboards/      # Dashboard JSON
│       └── provisioning/    # Datasource and dashboard provider configs
├── scripts/
│   ├── simulate_traffic.py          # Load generator for metrics
│   ├── start_monitoring_windows.ps1 # Native Windows monitoring setup
│   └── start_monitoring_windows.bat # Launcher for the PS1 script
├── tests/                   # pytest test suite
├── storage/                 # ChromaDB vectors + SQLite decisions (gitignored)
├── docker-compose.yml       # Application container
└── docker-compose.monitoring.yml  # Prometheus + Grafana containers
```

# Real-Time Observability Architecture — Prometheus & Grafana

## 1. Overview & Architecture

The CloudServe Support System has transitioned from static, post-hoc batch reporting to a **real-time observability architecture** powered by **Prometheus** and **Grafana**.

```
┌────────────────────────────────────────────────────────────────────────┐
│ CloudServe Support Pipeline & FastAPI Service                          │
│                                                                        │
│ Ingest ──► Classify ──► Retrieve ──► Route ──► Generate ──► Validate   │
│   │            │            │          │           │            │      │
│   └────────────┴────────────┴──────────┴───────────┴────────────┴──────┤
│                                                                        │
│                      src/metrics.py (Prometheus Registry)              │
│                      • Durations (p50/p95 histograms)                  │
│                      • In-flight gauge                                 │
│                      • Intent & Urgency distributions                  │
│                      • Routing decisions & safety compliance           │
│                      • Guardrail evaluation checks                     │
│                      • Errors & LLM Rate Limit events                  │
│                                                                        │
│                      HTTP /metrics endpoint (Port 8000)                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    │ Scrape (every 2s)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Prometheus (Port 9090)                                                 │
│ • TSDB Storage & Time Series Aggregations                              │
│ • Alert Rules (Error spikes, safety breaches, rate limiting)          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    │ PromQL Query API
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Grafana Live Operations Dashboard (Port 3000)                          │
│ • Key Performance Indicators (Throughput, Success %, Latency, SLAs)    │
│ • Stage Latency Breakdowns (Classify, Retrieve, Route, Generate)       │
│ • Intent & Urgency Distributions                                       │
│ • Routing & Must-Not-Auto-Respond Compliance                           │
│ • Vector Search & Grounding Score Tracking                             │
│ • Safety Guardrails & Failed Check Breakdown                           │
│ • Error Categorization & Rate Limit Backoff                            │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Metric Catalog & Reference

| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `cloudserve_tickets_total` | Counter | `channel`, `tier`, `status`, `final_action` | Total number of tickets processed |
| `cloudserve_tickets_in_flight` | Gauge | `channel` | Tickets currently executing in the pipeline |
| `cloudserve_ticket_processing_seconds` | Histogram | - | Total end-to-end processing time per ticket |
| `cloudserve_stage_duration_seconds` | Histogram | `stage` | Latency per pipeline stage (`classify`, `retrieve`, `route`, `generate`, `validate`) |
| `cloudserve_classification_intent_total` | Counter | `intent`, `urgency`, `answerable` | Intent classification distribution |
| `cloudserve_classification_confidence` | Histogram | `dimension` | Confidence scores for intent, urgency, and answerability |
| `cloudserve_retrieval_chunks_count` | Histogram | - | Count of document chunks retrieved per ticket |
| `cloudserve_retrieval_relevance_score` | Histogram | - | Cosine similarity / semantic relevance scores |
| `cloudserve_routing_decisions_total` | Counter | `action`, `rule_triggered`, `escalation_target` | Routing actions (auto_respond vs escalate) |
| `cloudserve_policy_compliance_total` | Counter | `policy`, `compliant` | Compliance rate for critical policies like `must_not_auto_respond` |
| `cloudserve_generation_total` | Counter | `could_answer`, `model` | Generated answers count and answerability |
| `cloudserve_generation_citations_count` | Histogram | - | Document citations per generated answer |
| `cloudserve_guardrails_evaluations_total`| Counter | `passed`, `action` | Safety evaluations passed vs blocked |
| `cloudserve_guardrails_checks_total` | Counter | `check_name`, `passed`, `severity` | Outcomes for individual safety checks (PII, hallucination, etc.) |
| `cloudserve_errors_total` | Counter | `error_type`, `stage` | Errors categorized by type and pipeline stage |
| `cloudserve_rate_limit_events_total` | Counter | `provider` | Number of 429 rate limit backoff triggers |
| `cloudserve_rate_limit_wait_seconds_total`| Counter | - | Total cumulative sleep time during rate limit backoffs |

---

## 3. Quick Start — Running Prometheus & Grafana

### Option A: Using Docker Compose (Recommended)

1. Start Prometheus and Grafana containers:
```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

2. Verify services:
- **Grafana Dashboard:** [http://localhost:3000](http://localhost:3000) (Default user: `admin`, password: `admin`)
- **Prometheus UI:** [http://localhost:9090](http://localhost:9090)

3. Run the application or traffic simulator:
```bash
# Option 1: Run the evaluation harness (auto-starts metrics on port 8000)
python evaluation_harness.py --input data/test_batch_20.json --output results/output_20.json

# Option 2: Run the live traffic simulator
python scripts/simulate_traffic.py --count 20 --delay 2.0

# Option 3: Run the FastAPI REST service
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

4. Open Grafana at `http://localhost:3000` to watch metrics update live in real time.

---

### Option B: Standalone / Local Binary Mode

If running Prometheus and Grafana without Docker:

1. **Prometheus:**
Download and run the Prometheus binary pointing to our configuration:
```bash
prometheus --config.file=monitoring/prometheus/prometheus.yml --web.listen-address=:9090
```

2. **Grafana:**
Install Grafana and configure the datasource pointing to `http://localhost:9090`, then import the dashboard JSON from `monitoring/grafana/dashboards/cloudserve_support_system.json`.

---

## 4. Key PromQL Queries for Operations

- **Current Ticket Ingestion Rate (Tickets/Min):**
  ```promql
  sum(rate(cloudserve_tickets_total[1m])) * 60
  ```

- **Pipeline 95th Percentile Stage Latency:**
  ```promql
  histogram_quantile(0.95, sum by (le, stage) (rate(cloudserve_stage_duration_seconds_bucket[5m])))
  ```

- **Policy Compliance Rate (`must_not_auto_respond`):**
  ```promql
  (sum(cloudserve_policy_compliance_total{policy="must_not_auto_respond", compliant="true"}) / sum(cloudserve_policy_compliance_total{policy="must_not_auto_respond"})) * 100
  ```

- **Guardrail Pass Rate:**
  ```promql
  (sum(cloudserve_guardrails_evaluations_total{passed="true"}) / sum(cloudserve_guardrails_evaluations_total)) * 100
  ```

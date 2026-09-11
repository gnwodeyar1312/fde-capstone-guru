# CloudServe Solutions — Intelligent Support System

An AI-powered customer support system that automatically resolves tickets using semantic retrieval over CloudServe's documentation, with intelligent routing, confidence-calibrated classification, and comprehensive guardrails.

## Problem

CloudServe Solutions receives 500+ support tickets weekly across four channels. Average response time is 7+ hours against a 2-hour SLA. First contact resolution is 43.8%. Discovery revealed that 71.4% of tickets are answerable from existing documentation — the problem is delivery, not knowledge.

## Architecture

```
Ticket → Ingest → Classify → Retrieve → Route → Generate → Validate → Response/Escalation
                                                    ↕
                                          Decision Log + Monitoring
```

The pipeline is orchestrated by **LangGraph StateGraph** with conditional edges — escalated tickets skip the Generate and Validate stages entirely.

**Components:** Ingest (4 channels → 1 format) · Classify (intent + urgency + confidence) · Retrieve (semantic search via Chroma) · Route (threshold-based auto-respond vs escalate) · Generate (cited answers) · Validate (guardrails that block)

## Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/gnwodeyar1312/fde-capstone-guru.git
cd fde-capstone-guru

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your API key:
#   OPENROUTER_API_KEY=your-key-here
```

The system uses OpenRouter as the primary LLM provider. See `.env.example` for all available configuration options including fallback provider settings.

### 3. Run the system

```bash
# Run the full evaluation harness
python evaluation_harness.py --input data/development_tickets.json --output results/output.json

# Run tests
python -m pytest tests/ -v

# Start the API server (exposes /metrics for Prometheus)
uvicorn src.api:app --host 0.0.0.0 --port 8000

# Start Prometheus + Grafana monitoring stack
docker compose -f docker-compose.monitoring.yml up -d
# Prometheus: http://localhost:9090  |  Grafana: http://localhost:3000
```

## Project Structure

```
├── README.md                       Setup and run instructions
├── requirements.txt                Pinned dependencies
├── .env.example                    Environment variable template
├── evaluation_harness.py           Evaluation harness entry point (FR-09)
├── src/
│   ├── ingest.py                   Normalise tickets from 4 channels
│   ├── classify.py                 Intent + urgency classification (LLM)
│   ├── retrieve.py                 Semantic search via LangChain Chroma
│   ├── route.py                    Threshold-based escalation routing
│   ├── generate.py                 Cited answer generation (LLM)
│   ├── guardrails.py               5 checks that can block a response
│   ├── pipeline.py                 LangGraph StateGraph orchestration
│   ├── config.py                   LLM + provider configuration
│   ├── logging_store.py            SQLite decision log
│   ├── metrics.py                  Prometheus metrics instrumentation
│   ├── monitoring.py               Metrics computation and reporting
│   ├── fairness_audit.py           Bias detection across customer tiers
│   ├── dashboard.py                Streamlit dashboard (legacy)
│   └── api.py                      FastAPI app with /metrics endpoint
├── prompts/
│   ├── README.md                   Prompt register and traceability
│   ├── build/
│   │   ├── PR-01_classification.md Classification prompt (FR-01/FR-02/FR-03)
│   │   └── PR-02_generation.md     Generation prompt (FR-04/FR-05/FR-06)
│   └── evaluation/
│       ├── PE-01_quality_assessment.md  Quality dimensions and thresholds
│       └── PE-02_citation_check.md      Citation validation logic
├── tests/                          157 tests across all components
├── evaluation/
│   ├── harness.py                  Module entry point
│   └── results/                    Dated output from each run
├── data/
│   ├── knowledge_base/             29 CloudServe documentation articles
│   ├── development_tickets.json    500 labeled tickets for development
│   └── validation_tickets.json     80 tickets for validation (do not tune)
├── monitoring/
│   ├── prometheus/
│   │   ├── prometheus.yml          Scrape config (2s interval)
│   │   └── alert.rules.yml         Alert rules (error rate, guardrails, rate limits)
│   └── grafana/
│       ├── dashboards/             Pre-built Grafana dashboard JSON
│       └── provisioning/           Auto-provisioned datasource + dashboard config
├── docker-compose.monitoring.yml   Prometheus + Grafana stack
├── docs/                           Architecture and design notes
└── .github/workflows/ci.yml        GitHub Actions CI pipeline
```

## Tech Stack

- **Language:** Python 3.10+
- **Orchestration:** LangChain + LangGraph (StateGraph with conditional edges)
- **LLM Provider:** OpenRouter (primary), Groq (fallback)
- **LLM Model:** Llama 3.1 8B Instant (free tier)
- **Vector Store:** LangChain Chroma with all-MiniLM-L6-v2 embeddings
- **API:** FastAPI with Prometheus /metrics endpoint
- **Monitoring:** Prometheus + Grafana (Docker Compose) with alert rules
- **Database:** SQLite (decision log)
- **CI:** GitHub Actions
- **Testing:** pytest (157 tests)

## Key Metrics

| Metric | Dev batch (20) | Validation (57/80) | Threshold |
|---|---|---|---|
| Intent accuracy | 100% | 96.5% | >= 80% |
| Route accuracy | 100% | 73.7% | >= 90% |
| Urgency accuracy | 30.0% | 40.4% | — |
| Doc hit rate | 93.3% | — | — |
| Guardrail pass rate | 92.3% | 97.2% | — |
| Pipeline crashes | 0 | 0 | 0 (NFR-02) |
| MNR compliance | 100% | — | 100% |

## Evaluation

The evaluation harness (FR-09) processes every ticket through the full LangGraph pipeline and produces a structured JSON output with predictions and ground truth for scoring:

```bash
python evaluation_harness.py --input data/development_tickets.json --output results/output.json
```

Individual ticket failures do not crash the pipeline — errors are logged and the harness continues (NFR-02). Rate limiting is handled with exponential backoff. Progress is logged every 10 tickets.

## Testing

```bash
# All tests
python -m pytest tests/ -v

# Individual modules
python -m pytest tests/test_classify.py -v
python -m pytest tests/test_generate.py -v
python -m pytest tests/test_guardrails.py -v
python -m pytest tests/test_route.py -v
python -m pytest tests/test_retrieve.py -v
python -m pytest tests/test_ingest.py -v
```

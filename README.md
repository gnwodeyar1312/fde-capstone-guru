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

**Components:** Ingest (4 channels → 1 format) · Classify (intent + urgency + confidence) · Retrieve (semantic search via Chroma) · Route (threshold-based auto-respond vs escalate) · Generate (cited answers) · Validate (guardrails that block)

## Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/gnwodeyar1312/fde-capstone-guru.git
cd fde-capstone-guru

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Run the system

```bash
# Start the API
python -m src.api

# Run the full evaluation (unattended)
python -m evaluation.harness --input data/validation_tickets.json --output evaluation/results/

# Run tests
python -m pytest tests/ -v
```

## Project Structure

```
├── README.md                    Setup and run instructions
├── requirements.txt             Pinned dependencies
├── .env.example                 Environment variable template
├── src/
│   ├── ingest.py               Normalise tickets from all four channels
│   ├── classify.py             Intent and urgency with confidence
│   ├── retrieve.py             Vector search over documentation
│   ├── route.py                Escalation decision and threshold
│   ├── generate.py             Answer drafting with citations
│   ├── guardrails.py           Checks that can block a response
│   ├── logging_store.py        Decision log
│   ├── api.py                  FastAPI application
│   └── config.py               Configuration management
├── prompts/
│   ├── build/                  Prompts used inside the system
│   └── evaluation/             Prompts used to judge output
├── tests/                      Test suite
├── evaluation/
│   ├── harness.py              Runs evaluation end to end
│   └── results/                Dated output from each run
├── docs/                       Architecture notes
├── data/                       Sample data only
└── .github/workflows/ci.yml    Continuous integration
```

## Tech Stack

- **Language:** Python 3.10+
- **Orchestration:** LangChain + LangGraph
- **Vector Store:** Chroma with all-MiniLM-L6-v2 embeddings
- **LLM:** Groq (Llama 3.1 8B, free tier)
- **API:** FastAPI
- **Database:** SQLite (decision log)
- **Monitoring:** Prometheus + Grafana
- **CI:** GitHub Actions

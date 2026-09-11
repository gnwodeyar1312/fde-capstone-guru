# CloudServe Support System — Command Reference

All commands are run from the project root: `C:\fde-capstone\fde-capstone-guru`

## 1. Setup (First Time)

```bash
# Create virtual environment
python -m venv .venv

# Activate it (Windows CMD)
.venv\Scripts\activate

# Activate it (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate it (Git Bash / Linux / Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment config and add your Groq API key
copy .env.example .env
# Edit .env and set GROQ_API_KEY=your_actual_key
```

## 2. Run Unit Tests

```bash
# Run ALL unit tests
python -m pytest tests/ -v

# Run tests for a specific module
python -m pytest tests/test_classify.py -v
python -m pytest tests/test_ingest.py -v
python -m pytest tests/test_retrieve.py -v
python -m pytest tests/test_route.py -v
python -m pytest tests/test_generate.py -v
python -m pytest tests/test_guardrails.py -v
python -m pytest tests/test_logging_store.py -v

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=term-missing
```

## 3. Run the Evaluation Pipeline

```bash
# Small batch (5 tickets) — quick sanity check (~2 min)
python evaluation_harness.py --input data/test_batch_5.json --output results/output_5.json

# Medium batch (20 tickets) — broader validation (~7 min)
python evaluation_harness.py --input data/test_batch_20.json --output results/output_20.json

# Full evaluation (500 tickets) — complete run (~60 min, watch rate limits)
python evaluation_harness.py --input data/development_tickets.json --output results/output_full.json
```

> **Rate Limit Note (Groq Free Tier):** 30 RPM, 200K tokens/day.
> A 50-ticket run uses ~20K tokens. Full 500-ticket run needs ~100K+ tokens
> and may need to be split across days or upgraded to a paid tier.

## 4. Real-Time Observability & Monitoring (Prometheus + Grafana)

You can run the monitoring stack in either of two ways: **Without Docker (Standalone Windows Binaries)** or **With Docker Desktop**.

### Option A: Without Docker (Standalone on Windows — Recommended if Docker is not installed)

We provide a 1-click script that automatically downloads portable standalone Windows versions of Prometheus & Grafana and starts both:

```bash
# In PowerShell:
powershell -ExecutionPolicy Bypass -File scripts/start_monitoring_windows.ps1

# Or in Windows CMD:
scripts\start_monitoring_windows.bat
```

- **Prometheus UI:** [http://localhost:9090](http://localhost:9090)
- **Grafana Live Dashboard:** [http://localhost:3000](http://localhost:3000) *(User: `admin` / Password: `admin`)*

---

### Option B: With Docker Desktop (If Docker is installed)

```bash
# Start Prometheus & Grafana in background (Docker Compose v2)
docker compose -f docker-compose.monitoring.yml up -d

# Or with legacy docker-compose:
docker-compose -f docker-compose.monitoring.yml up -d

# Stop monitoring stack when done:
docker compose -f docker-compose.monitoring.yml down
```

---

### Running Tickets & Live Observation

Once Prometheus & Grafana are running (via Option A or Option B), run any of the following to see live metrics flowing into Grafana:

```bash
# 1. Stream simulated tickets to observe live metric charts in Grafana in real-time
python scripts/simulate_traffic.py --count 20 --delay 2.0 --port 8000

# 2. Run evaluation harness with live Prometheus telemetry (exposed on port 8000)
python evaluation_harness.py --input data/test_batch_20.json --output results/output_20.json

# 3. Or launch the FastAPI REST application (exposes /metrics and /health at port 8000)
uvicorn src.api:app --host 0.0.0.0 --port 8000
# -> Test metrics endpoint: curl http://localhost:8000/metrics
# -> Test health endpoint:  curl http://localhost:8000/health

# 4. Post-hoc static analysis (legacy batch report)
python src/monitoring.py --input results/output_20.json
```

## 5. Individual Pipeline Stages (for debugging)

```bash
# Test ingestion only
python -c "from src.ingest import ingest_tickets; tickets = ingest_tickets('data/test_batch_5.json'); print(f'Ingested {len(tickets)} tickets')"

# Test classification on one ticket
python -c "
from src.ingest import ingest_tickets
from src.classify import classify_ticket
tickets = ingest_tickets('data/test_batch_5.json')
result = classify_ticket(tickets[0])
print(f'Intent: {result.intent}, Answerable: {result.answerable_from_docs}')
"

# Test retrieval
python -c "
from src.retrieve import build_vector_store, retrieve_for_ticket
from src.ingest import ingest_tickets
build_vector_store()
tickets = ingest_tickets('data/test_batch_5.json')
result = retrieve_for_ticket(tickets[0])
print(f'Retrieved {len(result.chunks)} chunks')
for c in result.chunks: print(f'  [{c.doc_id}] {c.title} (score: {c.relevance_score:.2f})')
"
```

## 6. Project Structure

```
fde-capstone-guru/
├── src/                    # Core pipeline modules
│   ├── ingest.py           # B-01: Ticket ingestion & normalization
│   ├── classify.py         # B-02: LLM classification (intent, urgency, answerable)
│   ├── retrieve.py         # B-03: Vector search (ChromaDB + MiniLM)
│   ├── route.py            # B-04: Deterministic routing (4 rules)
│   ├── generate.py         # B-07: Response generation with citations
│   ├── guardrails.py       # B-09: 5 rule-based validation checks
│   ├── logging_store.py    # B-10: SQLite decision logging
│   ├── monitoring.py       # B-12: Metrics collection & analysis
│   ├── dashboard.py        # B-12: Streamlit monitoring dashboard
│   └── config.py           # Configuration management
├── tests/                  # Unit tests for each module
├── data/                   # Input data (tickets, docs, ground truth)
├── results/                # Pipeline output files
├── docs/                   # Stage workbooks and design documents
├── evaluation_harness.py   # B-05: Full pipeline orchestrator
├── requirements.txt        # Python dependencies
├── .env.example            # Environment template
└── COMMANDS.md             # This file
```

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

## 4. Analyze Results

```bash
# Generate monitoring metrics from a pipeline run
python src/monitoring.py --input results/output_5.json

# Generate metrics and save report to file
python src/monitoring.py --input results/output_full.json --output results/metrics_report.json

# Launch the monitoring dashboard (opens in browser)
streamlit run src/dashboard.py
# Then open: http://localhost:8501
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

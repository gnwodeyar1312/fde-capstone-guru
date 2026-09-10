"""
Evaluation Harness for CloudServe Support System.

Build ID: B-05
Requirement: FR-09 — The evaluation harness shall accept --input and --output
file paths as command-line arguments. No file paths may be hardcoded.

Purpose:
    Run the COMPLETE LangGraph pipeline on every ticket in the input file and
    produce a structured output file with predictions, decisions, and ground
    truth for scoring. This is the script that THE GATE (B-11) will run.

Usage:
    python -m evaluation.harness --input data/development_tickets.json --output results/output.json

    Or from the project root:
    python evaluation_harness.py --input data/development_tickets.json --output results/output.json

Design decisions:
    1. Individual ticket failures do NOT crash the pipeline.
       Why? NFR-02 requires zero pipeline-level crashes. If ticket DEV-042
       fails (bad LLM response, timeout, etc.), we log the error, record
       it as a failed ticket, and continue to DEV-043. The output file
       includes the error so we can debug it.
    2. Rate limiting is handled with exponential backoff.
       Why? The free tier has rate limits. We need to process 500+ tickets
       without hitting the wall. Backoff with jitter lets us stay under
       the limit without wasting time.
    3. The output file includes BOTH predictions AND ground truth.
       Why? The evaluation report needs to compare them. Putting both in
       one file means the report script doesn't need to re-read the input.
    4. Progress is logged every 10 tickets.
       Why? Processing 500 tickets takes 30-60 minutes. Without progress
       updates, you'd stare at a blank terminal wondering if it's stuck.
    5. The vector store is built ONCE at startup, not per-ticket.
       Why? Embedding 29 docs takes ~5 seconds. Doing it 500 times would
       take 40 minutes of pure embedding overhead.
    6. The pipeline is orchestrated by LangGraph StateGraph.
       Why? LangGraph makes the pipeline stages explicit and composable.
       Each stage is a node; conditional edges handle routing (escalate
       vs. auto-respond). The harness invokes the compiled graph per ticket.

Interview context:
    "How do you run the full evaluation?"
    → python evaluation_harness.py --input data/development_tickets.json
      --output results/output.json
    "What if the LLM API is down?"
    → Each ticket has a try/except. Failed tickets are recorded with the
      error message. The pipeline continues. The output file tells you
      exactly which tickets failed and why.
    "How long does a full run take?"
    → ~30-60 minutes for 500 tickets on the free tier. The bottleneck
      is the LLM API calls (classification + generation = 2 calls per
      auto-respond ticket, 1 for escalated tickets). Retrieval and
      guardrails are instant (local embedding + deterministic checks).
"""

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import validate_config, get_llm
from src.ingest import ingest_tickets
from src.classify import classify_ticket
from src.retrieve import retrieve_for_ticket, build_vector_store
from src.route import route_ticket
from src.generate import generate_response
from src.guardrails import validate_response
from src.pipeline import run_ticket, get_pipeline
from src.logging_store import init_database, log_decision, build_decision_record, get_summary_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate limit handling
# ---------------------------------------------------------------------------

def _wait_with_backoff(attempt: int, base_delay: float = 5.0, max_delay: float = 120.0):
    """
    Exponential backoff with jitter for rate limit handling.

    Args:
        attempt: Which retry attempt (0-based)
        base_delay: Starting delay in seconds
        max_delay: Maximum delay cap
    """
    import random
    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 2), max_delay)
    logger.info("Rate limited. Waiting %.1f seconds (attempt %d)...", delay, attempt + 1)
    time.sleep(delay)


def _call_with_retry(func, *args, max_retries: int = 5, **kwargs):
    """
    Call a function with retry on rate limit errors.

    Catches common rate-limit and transient errors and retries with
    exponential backoff.
    """
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = any(phrase in error_str for phrase in [
                "rate_limit", "rate limit", "429", "too many requests",
                "quota", "tokens per minute", "requests per minute",
            ])
            is_transient = any(phrase in error_str for phrase in [
                "503", "502", "500", "service unavailable", "internal server error",
                "connection", "timeout",
            ])

            if (is_rate_limit or is_transient) and attempt < max_retries:
                _wait_with_backoff(attempt)
                continue
            raise  # Re-raise if not retryable or out of retries


# ---------------------------------------------------------------------------
# Process a single ticket through the full LangGraph pipeline
# ---------------------------------------------------------------------------

def process_ticket(ticket_data, ticket_index: int, total: int, llm=None) -> dict:
    """
    Run one ticket through all 6 pipeline stages via LangGraph.

    Returns a result dict with predictions, decisions, and metadata.
    On failure, returns a result dict with error information.
    """
    ticket = ticket_data.ticket
    labels = ticket_data.labels
    history = ticket_data.history

    start_time = time.time()

    result = {
        "ticket_id": ticket.ticket_id,
        "status": "success",
        "error": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "start_time": start_time,
        "end_time": None,

        # Ground truth (for evaluation)
        "ground_truth": {
            "intent": labels.intent if labels else "",
            "urgency": labels.urgency if labels else "",
            "expected_route": labels.expected_route if labels else "",
            "answerable_from_docs": labels.answerable_from_docs if labels else None,
            "expected_doc_ids": labels.expected_doc_ids if labels else [],
            "must_not_auto_respond": labels.must_not_auto_respond if labels else False,
        },

        # Predictions (filled in by pipeline stages)
        "predictions": {},
        "route": {},
        "retrieval": {},
        "generation": {},
        "guardrails": {},
        "final_action": "",
    }

    try:
        # Run the full pipeline via LangGraph with retry support
        pipeline_state = _call_with_retry(run_ticket, ticket, llm)

        # Extract classification results
        classification = pipeline_state.get("classification")
        if classification:
            result["predictions"] = {
                "intent": classification.intent,
                "intent_confidence": classification.intent_confidence,
                "urgency": classification.urgency,
                "urgency_confidence": classification.urgency_confidence,
                "answerable_from_docs": classification.answerable_from_docs,
                "answerable_confidence": classification.answerable_confidence,
                "must_not_auto_respond": classification.must_not_auto_respond,
                "reasoning": classification.reasoning,
            }

        # Extract retrieval results
        retrieval = pipeline_state.get("retrieval")
        if retrieval:
            result["retrieval"] = {
                "num_chunks": retrieval.num_results,
                "unique_doc_ids": retrieval.unique_doc_ids,
                "chunks": [
                    {
                        "doc_id": c.doc_id,
                        "section_name": c.section_name,
                        "relevance_score": round(c.relevance_score, 4),
                    }
                    for c in retrieval.chunks
                ],
            }

        # Extract route results
        route = pipeline_state.get("route_decision")
        if route:
            result["route"] = {
                "action": route.action.value,
                "reason": route.reason,
                "escalation_target": route.escalation_target,
                "rule_triggered": route.rule_triggered,
            }

        # Extract generation results (only for auto_respond)
        generation = pipeline_state.get("generation")
        if generation:
            result["generation"] = {
                "response_text": generation.response_text,
                "cited_doc_ids": generation.cited_doc_ids,
                "could_answer": generation.could_answer,
                "model_used": generation.model_used,
            }

        # Extract guardrail results
        guardrail_result = pipeline_state.get("guardrails")
        if guardrail_result:
            result["guardrails"] = {
                "passed": guardrail_result.passed,
                "failed_checks": guardrail_result.failed_checks,
                "warnings": guardrail_result.warnings,
                "recommendation": guardrail_result.recommendation,
                "checks": [
                    {
                        "name": c.name,
                        "passed": c.passed,
                        "reason": c.reason,
                        "severity": c.severity,
                    }
                    for c in guardrail_result.checks
                ],
            }

        # Final action from pipeline
        result["final_action"] = pipeline_state.get("final_action", "")

        # Log to decision database
        record = build_decision_record(
            ticket_id=ticket.ticket_id,
            classification=classification,
            route=route,
            retrieval_doc_ids=retrieval.unique_doc_ids if retrieval else [],
            num_chunks=retrieval.num_results if retrieval else 0,
            generation=generation,
            guardrails=guardrail_result,
            final_action=result["final_action"],
        )
        log_decision(record)

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {str(e)}"
        result["final_action"] = "error"
        logger.error(
            "Ticket %s (%d/%d) FAILED: %s",
            ticket.ticket_id, ticket_index + 1, total, result["error"]
        )
        logger.debug(traceback.format_exc())

    result["end_time"] = time.time()
    return result


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

def run_harness(input_path: str, output_path: str) -> dict:
    """
    Run the full evaluation harness.

    Args:
        input_path: Path to the input tickets JSON file
        output_path: Path to write the output results JSON file

    Returns:
        Summary statistics dict
    """
    start_time = time.time()

    # Validate configuration
    logger.info("Validating configuration...")
    validate_config()

    # Initialize LangChain LLM (shared across all tickets)
    logger.info("Initializing LangChain ChatOpenAI LLM...")
    llm = get_llm()

    # Initialize decision database
    logger.info("Initializing decision database...")
    init_database()

    # Build vector store via LangChain Chroma (once, not per-ticket)
    logger.info("Building vector store from documentation (LangChain Chroma)...")
    num_chunks = build_vector_store()
    logger.info("Vector store ready: %d chunks indexed", num_chunks)

    # Pre-compile the LangGraph pipeline
    logger.info("Compiling LangGraph pipeline...")
    get_pipeline()
    logger.info("Pipeline compiled and ready")

    # Ingest all tickets
    logger.info("Ingesting tickets from %s...", input_path)
    tickets = ingest_tickets(input_path)
    total = len(tickets)
    logger.info("Ingested %d tickets", total)

    # Process each ticket
    results = []
    success_count = 0
    error_count = 0

    for i, ticket_data in enumerate(tickets):
        # Progress logging every 10 tickets
        if i % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i / elapsed * 60) if elapsed > 0 and i > 0 else 0
            logger.info(
                "Progress: %d/%d tickets (%.0f%%) | %d success, %d errors | %.1f tickets/min",
                i, total, (i / total * 100), success_count, error_count, rate
            )

        result = process_ticket(ticket_data, i, total, llm=llm)
        results.append(result)

        if result["status"] == "success":
            success_count += 1
        else:
            error_count += 1

        # Delay between tickets to respect free tier rate limits.
        if i < total - 1:
            time.sleep(6.0)

    elapsed = time.time() - start_time

    # Build output
    output = {
        "metadata": {
            "input_file": str(input_path),
            "output_file": str(output_path),
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_tickets": total,
            "successful": success_count,
            "errors": error_count,
            "elapsed_seconds": round(elapsed, 1),
            "tickets_per_minute": round(total / (elapsed / 60), 1) if elapsed > 0 else 0,
        },
        "results": results,
    }

    # Write output file
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("Output written to %s", output_path)

    # Print summary
    logger.info("=" * 60)
    logger.info("EVALUATION HARNESS COMPLETE")
    logger.info("=" * 60)
    logger.info("Total tickets:  %d", total)
    logger.info("Successful:     %d (%.1f%%)", success_count, success_count / total * 100 if total else 0)
    logger.info("Errors:         %d (%.1f%%)", error_count, error_count / total * 100 if total else 0)
    logger.info("Elapsed time:   %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)
    logger.info("Throughput:     %.1f tickets/min", total / (elapsed / 60) if elapsed > 0 else 0)

    # Decision log stats
    stats = get_summary_stats()
    logger.info("-" * 60)
    logger.info("DECISION LOG SUMMARY")
    logger.info("-" * 60)
    for key, value in stats.items():
        logger.info("  %s: %s", key, value)

    # Count final actions from results
    from collections import Counter
    action_counts = Counter(r["final_action"] for r in results)
    logger.info("-" * 60)
    logger.info("FINAL ACTIONS")
    logger.info("-" * 60)
    for action, count in action_counts.most_common():
        logger.info("  %s: %d (%.1f%%)", action, count, count / total * 100)

    # Route accuracy (if ground truth available)
    route_correct = sum(
        1 for r in results
        if r["status"] == "success"
        and r["ground_truth"]["expected_route"]
        and r["route"].get("action") == r["ground_truth"]["expected_route"]
    )
    route_total = sum(
        1 for r in results
        if r["status"] == "success" and r["ground_truth"]["expected_route"]
    )
    if route_total:
        logger.info("-" * 60)
        logger.info("QUICK ACCURACY CHECK")
        logger.info("-" * 60)
        logger.info("  Route accuracy: %d/%d (%.1f%%)", route_correct, route_total, route_correct / route_total * 100)

    # Intent accuracy
    intent_correct = sum(
        1 for r in results
        if r["status"] == "success"
        and r["ground_truth"]["intent"]
        and r["predictions"].get("intent") == r["ground_truth"]["intent"]
    )
    intent_total = sum(
        1 for r in results
        if r["status"] == "success" and r["ground_truth"]["intent"]
    )
    if intent_total:
        logger.info("  Intent accuracy: %d/%d (%.1f%%)", intent_correct, intent_total, intent_correct / intent_total * 100)

    # Must-not-auto-respond compliance
    mnr_tickets = [
        r for r in results
        if r["status"] == "success" and r["ground_truth"]["must_not_auto_respond"]
    ]
    mnr_correct = sum(1 for r in mnr_tickets if r["final_action"] in ("escalated", "blocked_by_guardrails"))
    if mnr_tickets:
        logger.info("  Must-not-auto-respond compliance: %d/%d (%.1f%%)",
                     mnr_correct, len(mnr_tickets), mnr_correct / len(mnr_tickets) * 100)

    return output["metadata"]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CloudServe Support System — Evaluation Harness (LangGraph Pipeline)",
        epilog="Example: python evaluation_harness.py --input data/development_tickets.json --output results/output.json",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input tickets JSON file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the output results JSON file",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Run the harness
    try:
        summary = run_harness(args.input, args.output)
        logger.info("Harness completed successfully.")
        sys.exit(0)
    except Exception as e:
        logger.error("Harness FAILED: %s", e)
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

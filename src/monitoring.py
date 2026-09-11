"""
Monitoring & Metrics module for CloudServe Support System.

Build ID: B-12
Requirement: NFR-03 — Monitoring and observability for the pipeline.

Purpose:
    Analyze evaluation harness output files to produce structured metrics
    including accuracy breakdowns, latency stats, token usage estimates,
    error rates, and per-intent performance. Outputs both a summary to
    stdout and an optional JSON metrics file.

Design decisions:
    1. Post-hoc analysis, not real-time streaming.
       Why? Our pipeline runs in batch mode. Analyzing the output JSON
       after a run gives us the same data without adding overhead to each
       ticket's processing. For a capstone project, this is the right
       trade-off — we get full observability without complicating the
       hot path.
    2. Metrics are computed from the evaluation harness output.
       Why? The output file already contains predictions, ground truth,
       timings, and errors. No need for a separate data collection layer.
    3. JSON output for programmatic consumption.
       Why? The Streamlit dashboard reads this file. Other tools can too.
       Human-readable summary goes to stdout.
    4. Per-intent breakdown is critical.
       Why? Aggregate accuracy hides which intents the classifier struggles
       with. If we're 95% overall but 0% on compliance_request, we need to
       know that.

Interview context:
    "How do you monitor the system?"
    → We have a metrics module that analyzes each pipeline run. It computes
      accuracy by intent, route accuracy, latency distribution, error rates,
      and guardrail pass rates. Results feed into a Streamlit dashboard.
    "What metrics do you track?"
    → Intent accuracy (overall + per-intent), answerable accuracy, route
      accuracy, must_not_auto_respond compliance, average processing time,
      error rate, and guardrail pass/fail breakdown.
"""

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_metrics(results_data: dict) -> dict:
    """
    Compute comprehensive metrics from an evaluation harness output file.

    Args:
        results_data: Parsed JSON from the evaluation harness output

    Returns:
        Dict with all computed metrics
    """
    metadata = results_data.get("metadata", {})
    results = results_data.get("results", [])

    # Separate successful vs failed tickets
    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") != "success"]

    metrics = {
        "run_info": {
            "timestamp": metadata.get(
                "run_timestamp", datetime.now(timezone.utc).isoformat()
            ),
            "input_file": metadata.get("input_file", "unknown"),
            "total_tickets": metadata.get("total_tickets", len(results)),
            "successful": len(successful),
            "failed": len(failed),
            "error_rate": round(len(failed) / max(len(results), 1) * 100, 1),
            "elapsed_seconds": metadata.get("elapsed_seconds", 0),
            "tickets_per_minute": metadata.get("tickets_per_minute", 0),
        },
        "accuracy": _compute_accuracy(successful),
        "per_intent": _compute_per_intent(successful),
        "routing": _compute_routing_metrics(successful),
        "guardrails": _compute_guardrail_metrics(successful),
        "latency": _compute_latency_metrics(successful),
        "retrieval": _compute_retrieval_metrics(successful),
        "calibration": _compute_calibration_table(successful),
        "precision_recall": _compute_precision_recall(successful),
        "errors": _compute_error_breakdown(failed),
    }

    return metrics


def _compute_accuracy(results: list) -> dict:
    """Overall accuracy metrics including urgency."""
    if not results:
        return {"intent": 0, "urgency": 0, "answerable": 0, "route": 0, "sample_size": 0}

    intent_correct = 0
    urgency_correct = 0
    answerable_correct = 0
    route_correct = 0
    intent_total = 0
    urgency_total = 0
    answerable_total = 0
    route_total = 0

    for r in results:
        gt = r.get("ground_truth", {})
        route = r.get("route", {})
        pred = r.get("predictions", {})

        # Intent accuracy
        if gt.get("intent") and pred.get("intent"):
            intent_total += 1
            if gt["intent"] == pred["intent"]:
                intent_correct += 1

        # Urgency accuracy
        if gt.get("urgency") and pred.get("urgency"):
            urgency_total += 1
            if gt["urgency"] == pred["urgency"]:
                urgency_correct += 1

        # Answerable accuracy
        gt_ans = gt.get("answerable_from_docs")
        pred_ans = pred.get("answerable_from_docs")
        if gt_ans is not None and pred_ans is not None:
            answerable_total += 1
            if gt_ans == pred_ans:
                answerable_correct += 1

        # Route accuracy
        gt_route = gt.get("expected_route") or gt.get("expected_action")
        pred_route = route.get("action")
        if gt_route and pred_route:
            route_total += 1
            if gt_route == pred_route:
                route_correct += 1

    return {
        "intent": round(intent_correct / max(intent_total, 1) * 100, 1),
        "intent_correct": intent_correct,
        "intent_total": intent_total,
        "urgency": round(urgency_correct / max(urgency_total, 1) * 100, 1),
        "urgency_correct": urgency_correct,
        "urgency_total": urgency_total,
        "answerable": round(answerable_correct / max(answerable_total, 1) * 100, 1),
        "answerable_correct": answerable_correct,
        "answerable_total": answerable_total,
        "route": round(route_correct / max(route_total, 1) * 100, 1),
        "route_correct": route_correct,
        "route_total": route_total,
        "sample_size": len(results),
    }


def _compute_per_intent(results: list) -> dict:
    """Per-intent accuracy breakdown."""
    intent_stats = defaultdict(
        lambda: {
            "total": 0,
            "correct": 0,
            "answerable_correct": 0,
            "answerable_total": 0,
            "route_correct": 0,
            "route_total": 0,
        }
    )

    for r in results:
        gt = r.get("ground_truth", {})
        route = r.get("route", {})
        pred = r.get("predictions", {})
        gt_intent = gt.get("intent")

        if not gt_intent:
            continue

        stats = intent_stats[gt_intent]
        stats["total"] += 1

        if pred.get("intent") == gt_intent:
            stats["correct"] += 1

        gt_ans = gt.get("answerable_from_docs")
        pred_ans = pred.get("answerable_from_docs")
        if gt_ans is not None and pred_ans is not None:
            stats["answerable_total"] += 1
            if gt_ans == pred_ans:
                stats["answerable_correct"] += 1

        gt_route = gt.get("expected_route") or gt.get("expected_action")
        pred_route = route.get("action")
        if gt_route and pred_route:
            stats["route_total"] += 1
            if gt_route == pred_route:
                stats["route_correct"] += 1

    # Convert to percentages
    per_intent = {}
    for intent, stats in sorted(intent_stats.items()):
        per_intent[intent] = {
            "count": stats["total"],
            "intent_accuracy": round(
                stats["correct"] / max(stats["total"], 1) * 100, 1
            ),
            "answerable_accuracy": round(
                stats["answerable_correct"] / max(stats["answerable_total"], 1) * 100, 1
            ),
            "route_accuracy": round(
                stats["route_correct"] / max(stats["route_total"], 1) * 100, 1
            ),
        }

    return per_intent


def _compute_routing_metrics(results: list) -> dict:
    """Routing decision breakdown."""
    auto_respond = 0
    escalate = 0
    must_not_auto_flags = 0
    must_not_auto_correct = 0

    MUST_NOT_AUTO = {
        "compliance_request",
        "security_incident",
        "feature_request",
        "unclear_request",
    }

    for r in results:
        route = r.get("route", {})
        action = route.get("action")

        if action == "auto_respond":
            auto_respond += 1
        elif action == "escalate":
            escalate += 1

        # Check must_not_auto_respond compliance
        gt_intent = r.get("ground_truth", {}).get("intent", "")
        if gt_intent in MUST_NOT_AUTO:
            must_not_auto_flags += 1
            if action == "escalate":
                must_not_auto_correct += 1

    total = auto_respond + escalate
    return {
        "auto_respond": auto_respond,
        "escalate": escalate,
        "auto_respond_pct": round(auto_respond / max(total, 1) * 100, 1),
        "escalate_pct": round(escalate / max(total, 1) * 100, 1),
        "must_not_auto_total": must_not_auto_flags,
        "must_not_auto_correct": must_not_auto_correct,
        "must_not_auto_compliance": round(
            must_not_auto_correct / max(must_not_auto_flags, 1) * 100, 1
        ),
    }


def _compute_guardrail_metrics(results: list) -> dict:
    """Guardrail pass/fail breakdown."""
    total_checked = 0
    total_passed = 0
    check_stats = defaultdict(lambda: {"passed": 0, "failed": 0})

    for r in results:
        gr = r.get("guardrails")
        if not gr:
            continue

        total_checked += 1
        if gr.get("passed", False):
            total_passed += 1

        for check in gr.get("checks", []):
            name = check.get("name", "unknown")
            if check.get("passed", False):
                check_stats[name]["passed"] += 1
            else:
                check_stats[name]["failed"] += 1

    return {
        "total_checked": total_checked,
        "total_passed": total_passed,
        "pass_rate": round(total_passed / max(total_checked, 1) * 100, 1),
        "per_check": dict(check_stats),
    }


def _compute_latency_metrics(results: list) -> dict:
    """
    Per-ticket latency distribution with percentiles.

    The evaluation harness records start_time and end_time (epoch seconds)
    for each ticket, so we can compute real per-ticket latency.
    """
    latencies = []
    for r in results:
        start = r.get("start_time")
        end = r.get("end_time")
        if start and end and end > start:
            latencies.append(round(end - start, 2))

    if not latencies:
        return {
            "sample_size": 0,
            "note": "No per-ticket timing data available.",
        }

    latencies.sort()
    n = len(latencies)

    return {
        "sample_size": n,
        "min": latencies[0],
        "max": latencies[-1],
        "mean": round(sum(latencies) / n, 2),
        "median_p50": latencies[n // 2],
        "p75": latencies[int(n * 0.75)],
        "p90": latencies[int(n * 0.90)],
        "p95": latencies[min(int(n * 0.95), n - 1)],
        "p99": latencies[min(int(n * 0.99), n - 1)],
    }


def _compute_retrieval_metrics(results: list) -> dict:
    """
    Retrieval quality metrics: doc hit rate and chunk relevance stats.

    Doc hit rate = fraction of tickets where at least one expected doc_id
    appeared in the retrieved chunks. This measures whether the vector store
    is surfacing the right documentation.
    """
    hit = 0
    miss = 0
    total_checked = 0
    all_relevance_scores = []
    chunk_counts = []

    for r in results:
        gt = r.get("ground_truth", {})
        ret = r.get("retrieval", {})
        expected_docs = set(gt.get("expected_doc_ids") or [])
        retrieved_docs = set(ret.get("unique_doc_ids") or [])

        # Collect relevance scores
        for chunk in ret.get("chunks", []):
            score = chunk.get("relevance_score")
            if score is not None:
                all_relevance_scores.append(score)

        chunk_counts.append(ret.get("num_chunks", 0))

        if expected_docs:
            total_checked += 1
            if expected_docs & retrieved_docs:
                hit += 1
            else:
                miss += 1

    result = {
        "doc_hit_rate": round(hit / max(total_checked, 1) * 100, 1),
        "doc_hits": hit,
        "doc_misses": miss,
        "doc_total_checked": total_checked,
    }

    if all_relevance_scores:
        all_relevance_scores.sort()
        n = len(all_relevance_scores)
        result["relevance_mean"] = round(sum(all_relevance_scores) / n, 4)
        result["relevance_median"] = round(all_relevance_scores[n // 2], 4)
        result["relevance_min"] = round(all_relevance_scores[0], 4)
        result["relevance_max"] = round(all_relevance_scores[-1], 4)

    if chunk_counts:
        result["avg_chunks_retrieved"] = round(sum(chunk_counts) / len(chunk_counts), 1)

    return result


def _compute_calibration_table(results: list) -> dict:
    """
    Confidence calibration table for intent classification.

    Groups predictions by confidence bucket (0.5-0.6, 0.6-0.7, ... 0.9-1.0)
    and computes actual accuracy in each bucket. A well-calibrated model
    has actual accuracy ≈ stated confidence in each bucket.

    Also computes Expected Calibration Error (ECE) — the weighted average
    gap between confidence and accuracy across buckets.
    """
    buckets = {}
    bucket_edges = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8),
                    (0.8, 0.9), (0.9, 1.01)]

    for low, high in bucket_edges:
        label = f"{low:.1f}-{high:.1f}" if high <= 1.0 else f"{low:.1f}-1.0"
        buckets[label] = {"total": 0, "correct": 0, "sum_confidence": 0.0}

    for r in results:
        gt = r.get("ground_truth", {})
        pred = r.get("predictions", {})
        gt_intent = gt.get("intent")
        pred_intent = pred.get("intent")
        conf = pred.get("intent_confidence")

        if not gt_intent or not pred_intent or conf is None:
            continue

        for (low, high), label in zip(bucket_edges, buckets.keys()):
            if low <= conf < high:
                buckets[label]["total"] += 1
                buckets[label]["sum_confidence"] += conf
                if gt_intent == pred_intent:
                    buckets[label]["correct"] += 1
                break

    # Build the calibration table and compute ECE
    table = {}
    ece_numerator = 0.0
    total_samples = 0

    for label, stats in buckets.items():
        if stats["total"] == 0:
            continue
        accuracy = round(stats["correct"] / stats["total"] * 100, 1)
        avg_conf = round(stats["sum_confidence"] / stats["total"] * 100, 1)
        table[label] = {
            "count": stats["total"],
            "accuracy_pct": accuracy,
            "avg_confidence_pct": avg_conf,
            "gap": round(abs(accuracy - avg_conf), 1),
        }
        ece_numerator += stats["total"] * abs(
            stats["correct"] / stats["total"] - stats["sum_confidence"] / stats["total"]
        )
        total_samples += stats["total"]

    ece = round(ece_numerator / max(total_samples, 1) * 100, 2)

    return {
        "buckets": table,
        "expected_calibration_error_pct": ece,
        "interpretation": (
            "ECE < 5% = well calibrated, 5-10% = acceptable, > 10% = poorly calibrated"
        ),
    }


def _compute_precision_recall(results: list) -> dict:
    """
    Per-class precision, recall, and F1 for intent classification.

    Precision = of all tickets predicted as intent X, how many truly are X?
    Recall = of all tickets that truly are intent X, how many did we predict?
    F1 = harmonic mean of precision and recall.
    """
    # Build confusion data
    pred_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for r in results:
        gt = r.get("ground_truth", {})
        pred = r.get("predictions", {})
        gt_intent = gt.get("intent")
        pred_intent = pred.get("intent")

        if not gt_intent or not pred_intent:
            continue

        if gt_intent == pred_intent:
            pred_counts[gt_intent]["tp"] += 1
        else:
            pred_counts[pred_intent]["fp"] += 1
            pred_counts[gt_intent]["fn"] += 1

    per_class = {}
    macro_p = 0.0
    macro_r = 0.0
    macro_f1 = 0.0
    n_classes = 0

    for intent in sorted(pred_counts.keys()):
        tp = pred_counts[intent]["tp"]
        fp = pred_counts[intent]["fp"]
        fn = pred_counts[intent]["fn"]

        precision = round(tp / max(tp + fp, 1) * 100, 1)
        recall = round(tp / max(tp + fn, 1) * 100, 1)
        if precision + recall > 0:
            f1 = round(2 * (precision * recall) / (precision + recall), 1)
        else:
            f1 = 0.0

        per_class[intent] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }

        macro_p += precision
        macro_r += recall
        macro_f1 += f1
        n_classes += 1

    return {
        "per_class": per_class,
        "macro_avg": {
            "precision": round(macro_p / max(n_classes, 1), 1),
            "recall": round(macro_r / max(n_classes, 1), 1),
            "f1": round(macro_f1 / max(n_classes, 1), 1),
        },
    }


def _compute_error_breakdown(failed: list) -> dict:
    """Categorize errors from failed tickets."""
    error_types = defaultdict(int)

    for r in failed:
        error = r.get("error", "unknown")
        if "rate_limit" in error.lower() or "429" in error:
            error_types["rate_limit"] += 1
        elif "timeout" in error.lower():
            error_types["timeout"] += 1
        elif "json" in error.lower() or "parse" in error.lower():
            error_types["parse_error"] += 1
        elif "connection" in error.lower():
            error_types["connection_error"] += 1
        else:
            error_types["other"] += 1

    return {
        "total_errors": len(failed),
        "by_type": dict(error_types),
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_metrics(metrics: dict):
    """Pretty-print metrics to stdout."""
    ri = metrics["run_info"]
    acc = metrics["accuracy"]
    route = metrics["routing"]
    gr = metrics["guardrails"]

    print("=" * 65)
    print("  CLOUDSERVE SUPPORT SYSTEM — MONITORING REPORT")
    print("=" * 65)

    print(f"\n  Run: {ri['timestamp']}")
    print(f"  Input: {ri['input_file']}")
    print(
        f"  Tickets: {ri['successful']}/{ri['total_tickets']} successful "
        f"({ri['error_rate']}% error rate)"
    )
    print(
        f"  Duration: {ri['elapsed_seconds']:.1f}s "
        f"({ri['tickets_per_minute']:.1f} tickets/min)"
    )

    print(f"\n{'─' * 65}")
    print("  ACCURACY")
    print(f"{'─' * 65}")
    print(
        f"  Intent:      {acc['intent']:>6.1f}%  ({acc['intent_correct']}/{acc['intent_total']})"
    )
    print(
        f"  Urgency:     {acc['urgency']:>6.1f}%  ({acc['urgency_correct']}/{acc['urgency_total']})"
    )
    print(
        f"  Answerable:  {acc['answerable']:>6.1f}%  ({acc['answerable_correct']}/{acc['answerable_total']})"
    )
    print(
        f"  Route:       {acc['route']:>6.1f}%  ({acc['route_correct']}/{acc['route_total']})"
    )

    print(f"\n{'─' * 65}")
    print("  ROUTING")
    print(f"{'─' * 65}")
    print(
        f"  Auto-respond:  {route['auto_respond']}  ({route['auto_respond_pct']:.1f}%)"
    )
    print(f"  Escalate:      {route['escalate']}  ({route['escalate_pct']:.1f}%)")
    print(
        f"  Must-not-auto: {route['must_not_auto_correct']}/{route['must_not_auto_total']} "
        f"({route['must_not_auto_compliance']:.1f}% compliance)"
    )

    print(f"\n{'─' * 65}")
    print("  GUARDRAILS")
    print(f"{'─' * 65}")
    print(
        f"  Pass rate: {gr['total_passed']}/{gr['total_checked']} "
        f"({gr['pass_rate']:.1f}%)"
    )
    for name, stats in gr.get("per_check", {}).items():
        total = stats["passed"] + stats["failed"]
        pct = round(stats["passed"] / max(total, 1) * 100, 1)
        print(f"    {name}: {stats['passed']}/{total} ({pct}%)")

    # Per-intent breakdown
    pi = metrics.get("per_intent", {})
    if pi:
        print(f"\n{'─' * 65}")
        print("  PER-INTENT BREAKDOWN")
        print(f"{'─' * 65}")
        print(f"  {'Intent':<28} {'Count':>5} {'Intent%':>8} {'Ans%':>8} {'Route%':>8}")
        print(f"  {'─' * 28} {'─' * 5} {'─' * 8} {'─' * 8} {'─' * 8}")
        for intent, stats in pi.items():
            print(
                f"  {intent:<28} {stats['count']:>5} "
                f"{stats['intent_accuracy']:>7.1f}% "
                f"{stats['answerable_accuracy']:>7.1f}% "
                f"{stats['route_accuracy']:>7.1f}%"
            )

    # Latency
    lat = metrics.get("latency", {})
    if lat.get("sample_size", 0) > 0:
        print(f"\n{'─' * 65}")
        print("  LATENCY (per ticket)")
        print(f"{'─' * 65}")
        print(f"  Min:     {lat['min']:>7.1f}s")
        print(f"  Median:  {lat['median_p50']:>7.1f}s")
        print(f"  Mean:    {lat['mean']:>7.1f}s")
        print(f"  P90:     {lat['p90']:>7.1f}s")
        print(f"  P95:     {lat['p95']:>7.1f}s")
        print(f"  Max:     {lat['max']:>7.1f}s")

    # Retrieval
    ret = metrics.get("retrieval", {})
    if ret.get("doc_total_checked", 0) > 0:
        print(f"\n{'─' * 65}")
        print("  RETRIEVAL")
        print(f"{'─' * 65}")
        print(
            f"  Doc hit rate:    {ret['doc_hit_rate']:>5.1f}%  "
            f"({ret['doc_hits']}/{ret['doc_total_checked']})"
        )
        if "relevance_mean" in ret:
            print(f"  Relevance mean:  {ret['relevance_mean']:.4f}")
            print(f"  Relevance median:{ret['relevance_median']:.4f}")
        if "avg_chunks_retrieved" in ret:
            print(f"  Avg chunks:      {ret['avg_chunks_retrieved']:.1f}")

    # Calibration
    cal = metrics.get("calibration", {})
    cal_buckets = cal.get("buckets", {})
    if cal_buckets:
        print(f"\n{'─' * 65}")
        print("  CONFIDENCE CALIBRATION (intent)")
        print(f"{'─' * 65}")
        print(f"  {'Bucket':<12} {'Count':>6} {'Accuracy':>9} {'Avg Conf':>9} {'Gap':>6}")
        print(f"  {'─' * 12} {'─' * 6} {'─' * 9} {'─' * 9} {'─' * 6}")
        for bucket, stats in cal_buckets.items():
            print(
                f"  {bucket:<12} {stats['count']:>6} "
                f"{stats['accuracy_pct']:>8.1f}% "
                f"{stats['avg_confidence_pct']:>8.1f}% "
                f"{stats['gap']:>5.1f}%"
            )
        ece = cal.get("expected_calibration_error_pct", 0)
        print(f"\n  ECE: {ece:.2f}%  ({cal.get('interpretation', '')})")

    # Precision / Recall (show only classes with support > 0 and imperfect scores)
    pr = metrics.get("precision_recall", {})
    pr_classes = pr.get("per_class", {})
    if pr_classes:
        print(f"\n{'─' * 65}")
        print("  PRECISION / RECALL / F1 (per intent)")
        print(f"{'─' * 65}")
        print(f"  {'Intent':<28} {'Prec':>6} {'Recall':>7} {'F1':>6} {'N':>4}")
        print(f"  {'─' * 28} {'─' * 6} {'─' * 7} {'─' * 6} {'─' * 4}")
        for intent, stats in pr_classes.items():
            print(
                f"  {intent:<28} {stats['precision']:>5.1f}% "
                f"{stats['recall']:>6.1f}% "
                f"{stats['f1']:>5.1f}% "
                f"{stats['support']:>3}"
            )
        macro = pr.get("macro_avg", {})
        print(f"  {'─' * 28} {'─' * 6} {'─' * 7} {'─' * 6}")
        print(
            f"  {'MACRO AVG':<28} {macro.get('precision', 0):>5.1f}% "
            f"{macro.get('recall', 0):>6.1f}% "
            f"{macro.get('f1', 0):>5.1f}%"
        )

    # Error breakdown
    errors = metrics.get("errors", {})
    if errors.get("total_errors", 0) > 0:
        print(f"\n{'─' * 65}")
        print("  ERRORS")
        print(f"{'─' * 65}")
        for etype, count in errors.get("by_type", {}).items():
            print(f"  {etype}: {count}")

    print(f"\n{'=' * 65}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compute monitoring metrics from evaluation harness output"
    )
    parser.add_argument(
        "--input", required=True, help="Path to evaluation harness output JSON"
    )
    parser.add_argument(
        "--output", default=None, help="Optional path to save metrics JSON report"
    )
    args = parser.parse_args()

    # Load results
    with open(args.input, "r") as f:
        results_data = json.load(f)

    # Compute metrics
    metrics = compute_metrics(results_data)

    # Print to stdout
    print_metrics(metrics)

    # Save if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {args.output}")


if __name__ == "__main__":
    main()

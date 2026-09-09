#!/usr/bin/env python3
"""
Fairness Audit for CloudServe Intelligent Support System.

B-14: Analyze evaluation results for bias across customer dimensions:
  1. Customer tier (free / starter / professional / enterprise)
  2. Customer region (NA / EU / APAC / LATAM)
  3. Intent category — do some intents get worse treatment?
  4. Response quality — are auto-responded tickets consistent?

Design decisions:
    We audit for OUTCOME FAIRNESS, not just accuracy. Two questions:
    1. "Does the system escalate more for some groups?" (route bias)
    2. "Do some groups get worse auto-responses?" (quality bias)

    A support system that correctly escalates enterprise tickets but
    wrongly auto-responds to free-tier tickets has a fairness problem
    even if overall accuracy looks good.

Interview context:
    "How do you ensure your AI system doesn't discriminate?"
    → We measure outcomes across protected dimensions (tier, region).
      If accuracy or escalation rates differ significantly between
      groups, we investigate whether the difference is justified
      (some groups genuinely have harder problems) or biased.
    "What metrics do you use for fairness?"
    → Equalized accuracy (does the system make equally good decisions
      for all groups?) and demographic parity of escalation rates
      (does the system escalate at similar rates across groups?).

Usage:
    python -m src.fairness_audit results/output_50.json
    python -m src.fairness_audit results/output_50.json results/output_rerun_21.json
"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def load_results(*file_paths):
    """Load and merge evaluation results from one or more output files.

    When merging, later files override earlier ones for the same ticket_id,
    so the rerun results (with classifier fixes) take precedence.
    """
    tickets_by_id = {}
    for fp in file_paths:
        with open(fp) as f:
            data = json.load(f)
        for r in data["results"]:
            if r["status"] == "success":
                tickets_by_id[r["ticket_id"]] = r
    return list(tickets_by_id.values())


def load_ticket_metadata(data_dir="data"):
    """Load customer metadata from the test batch files.

    The evaluation output doesn't include customer_tier and customer_region,
    so we join them from the original ticket data.
    """
    metadata = {}
    for json_file in Path(data_dir).glob("*.json"):
        try:
            with open(json_file) as f:
                tickets = json.load(f)
            if isinstance(tickets, list):
                for t in tickets:
                    if "ticket_id" in t:
                        metadata[t["ticket_id"]] = {
                            "customer_tier": t.get("customer_tier", "unknown"),
                            "customer_region": t.get("customer_region", "unknown"),
                            "customer_name": t.get("customer_name", "unknown"),
                            "language_fluency": t.get("language_fluency", "unknown"),
                        }
        except (json.JSONDecodeError, KeyError):
            continue
    return metadata


def compute_group_metrics(results, metadata, group_key):
    """Compute accuracy and routing metrics per group.

    Returns a dict: group_value → {n, intent_acc, ans_acc, route_acc,
                                    escalation_rate, auto_respond_rate}
    """
    groups = defaultdict(
        lambda: {
            "n": 0,
            "intent_correct": 0,
            "ans_correct": 0,
            "route_correct": 0,
            "escalated": 0,
            "auto_responded": 0,
            "blocked": 0,
            "avg_response_len": [],
        }
    )

    for r in results:
        tid = r["ticket_id"]
        meta = metadata.get(tid, {})
        group = meta.get(group_key, "unknown")

        g = groups[group]
        g["n"] += 1

        if r["predictions"]["intent"] == r["ground_truth"]["intent"]:
            g["intent_correct"] += 1
        if (
            r["predictions"]["answerable_from_docs"]
            == r["ground_truth"]["answerable_from_docs"]
        ):
            g["ans_correct"] += 1
        if r["route"]["action"] == r["ground_truth"]["expected_route"]:
            g["route_correct"] += 1

        action = r.get("final_action", r["route"]["action"])
        if action in ("escalated", "escalate"):
            g["escalated"] += 1
        elif action in ("sent", "auto_respond"):
            g["auto_responded"] += 1

        # Response length (only for auto-responded tickets)
        resp = r.get("generation", {}).get("response_text", "")
        if resp:
            g["avg_response_len"].append(len(resp))

    # Compute rates
    result = {}
    for group, g in sorted(groups.items()):
        n = g["n"]
        result[group] = {
            "n": n,
            "intent_accuracy": g["intent_correct"] / n if n else 0,
            "answerable_accuracy": g["ans_correct"] / n if n else 0,
            "route_accuracy": g["route_correct"] / n if n else 0,
            "escalation_rate": g["escalated"] / n if n else 0,
            "auto_respond_rate": g["auto_responded"] / n if n else 0,
            "avg_response_length": (
                sum(g["avg_response_len"]) / len(g["avg_response_len"])
                if g["avg_response_len"]
                else 0
            ),
        }

    return result


def check_disparity(group_metrics, metric_name, threshold=0.20):
    """Check if the spread between best and worst group exceeds threshold.

    Returns (is_fair, spread, best_group, worst_group).
    """
    if len(group_metrics) < 2:
        return True, 0, None, None

    # Filter out groups with very small n (< 3)
    valid = {k: v for k, v in group_metrics.items() if v["n"] >= 3}
    if len(valid) < 2:
        return True, 0, None, None

    values = {k: v[metric_name] for k, v in valid.items()}
    best_group = max(values, key=values.get)
    worst_group = min(values, key=values.get)
    spread = values[best_group] - values[worst_group]

    return spread <= threshold, spread, best_group, worst_group


def print_group_table(group_metrics, group_name):
    """Pretty-print a group metrics table."""
    print(f"\n{'─' * 78}")
    print(f"  BY {group_name.upper()}")
    print(f"{'─' * 78}")
    print(
        f"  {'Group':<16} {'N':>3} {'Intent':>8} {'Answer':>8} {'Route':>8} {'Esc%':>7} {'Auto%':>7} {'AvgLen':>7}"
    )
    print(
        f"  {'─' * 16} {'─' * 3} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 7} {'─' * 7} {'─' * 7}"
    )

    for group, m in sorted(group_metrics.items()):
        print(
            f"  {group:<16} {m['n']:>3} "
            f"{m['intent_accuracy']:>7.0%} "
            f"{m['answerable_accuracy']:>7.0%} "
            f"{m['route_accuracy']:>7.0%} "
            f"{m['escalation_rate']:>6.0%} "
            f"{m['auto_respond_rate']:>6.0%} "
            f"{m['avg_response_length']:>6.0f}"
        )


def print_disparity_check(group_metrics, group_name, metrics_to_check):
    """Print disparity analysis for a group dimension."""
    print("\n  Disparity checks (threshold: 20%):")
    all_fair = True
    for metric in metrics_to_check:
        is_fair, spread, best, worst = check_disparity(group_metrics, metric)
        status = "✓ FAIR" if is_fair else "⚠ DISPARITY"
        if not is_fair:
            all_fair = False
        detail = f" (spread {spread:.0%}: {best} vs {worst})" if best else ""
        print(f"    {metric:<24} {status}{detail}")

    return all_fair


def run_audit(results, metadata):
    """Run the full fairness audit and print results."""
    print("=" * 78)
    print("  CLOUDSERVE SUPPORT SYSTEM — FAIRNESS AUDIT REPORT")
    print("=" * 78)
    print(f"\n  Tickets analyzed: {len(results)}")
    print(
        f"  Tickets with metadata: {sum(1 for r in results if r['ticket_id'] in metadata)}"
    )

    metrics_to_check = [
        "intent_accuracy",
        "answerable_accuracy",
        "route_accuracy",
        "escalation_rate",
    ]

    all_fair = True

    # ── By Customer Tier ──
    tier_metrics = compute_group_metrics(results, metadata, "customer_tier")
    print_group_table(tier_metrics, "Customer Tier")
    tier_fair = print_disparity_check(tier_metrics, "Customer Tier", metrics_to_check)
    all_fair = all_fair and tier_fair

    # ── By Region ──
    region_metrics = compute_group_metrics(results, metadata, "customer_region")
    print_group_table(region_metrics, "Customer Region")
    region_fair = print_disparity_check(
        region_metrics, "Customer Region", metrics_to_check
    )
    all_fair = all_fair and region_fair

    # ── By Intent ──
    # For intent, we use a synthetic "group" from the ground truth intent
    intent_metadata = {
        r["ticket_id"]: {"intent_group": r["ground_truth"]["intent"]} for r in results
    }
    intent_metrics = compute_group_metrics(results, intent_metadata, "intent_group")
    print_group_table(intent_metrics, "Intent Category")
    # Only check route accuracy disparity for intents (intent accuracy is tautological)
    intent_fair = print_disparity_check(
        intent_metrics,
        "Intent Category",
        ["answerable_accuracy", "route_accuracy"],
    )
    all_fair = all_fair and intent_fair

    # ── By Language Fluency ──
    fluency_metrics = compute_group_metrics(results, metadata, "language_fluency")
    if len(fluency_metrics) > 1:
        print_group_table(fluency_metrics, "Language Fluency")
        fluency_fair = print_disparity_check(
            fluency_metrics, "Language Fluency", metrics_to_check
        )
        all_fair = all_fair and fluency_fair

    # ── Overall Verdict ──
    print(f"\n{'=' * 78}")
    if all_fair:
        print("  VERDICT: ✓ NO SIGNIFICANT DISPARITIES DETECTED")
        print("  The system treats all customer segments with comparable accuracy.")
    else:
        print("  VERDICT: ⚠ DISPARITIES DETECTED — REVIEW RECOMMENDED")
        print("  Some customer segments receive different treatment.")
        print("  Investigate whether differences are justified by problem complexity")
        print("  or indicate systematic bias in the classifier/retriever.")
    print(f"{'=' * 78}")

    return all_fair


def main():
    # Parse --data-dir flag if present
    args = sys.argv[1:]
    data_dir = "data"
    result_files = []

    i = 0
    while i < len(args):
        if args[i] == "--data-dir" and i + 1 < len(args):
            data_dir = args[i + 1]
            i += 2
        else:
            result_files.append(args[i])
            i += 1

    if not result_files:
        print(
            "Usage: python -m src.fairness_audit <output_file.json> [<output_file2.json> ...] [--data-dir <path>]"
        )
        print(
            "       python -m src.fairness_audit results/output_50.json results/output_rerun_21.json"
        )
        sys.exit(1)

    results = load_results(*result_files)

    if not results:
        print("No successful results found in the provided files.")
        sys.exit(1)

    # Look for ticket data in common locations
    if not Path(data_dir).exists():
        data_dir = os.path.join(os.path.dirname(result_files[0]), "..", "data")

    metadata = load_ticket_metadata(data_dir)
    if not metadata:
        print(f"Warning: No ticket metadata found in {data_dir}/")
        print("Audit will proceed without customer tier/region breakdown.")

    is_fair = run_audit(results, metadata)
    sys.exit(0 if is_fair else 1)


if __name__ == "__main__":
    main()

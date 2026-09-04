"""
Discovery Analysis Script
--------------------------
This script computes every metric in the Stage 1 Discovery Workbook
directly from development_tickets.json.

Usage:
    python discovery_analysis.py

Run from: C:\fde-capstone\fde-capstone-guru
Make sure your venv is activated first.
"""

import json
from collections import Counter
from itertools import chain
from pathlib import Path


def main():
    # Load the data
    data_path = Path("data/development_tickets.json")
    with open(data_path) as f:
        tickets = json.load(f)

    print(f"Total tickets: {len(tickets)}")
    print()

    # --- 1. Channel split ---
    channels = Counter(t["channel"] for t in tickets)
    print("=== CHANNEL SPLIT ===")
    for ch, count in channels.most_common():
        print(f"  {ch}: {count} ({count/len(tickets)*100:.1f}%)")

    # --- 2. Intent distribution ---
    intents = Counter(t["labels"]["intent"] for t in tickets)
    print(f"\n=== INTENT DISTRIBUTION ({len(intents)} distinct) ===")
    for intent, count in intents.most_common():
        print(f"  {intent}: {count} ({count/len(tickets)*100:.1f}%)")

    # --- 3. Urgency split ---
    urgency = Counter(t["labels"]["urgency"] for t in tickets)
    print("\n=== URGENCY ===")
    for u, count in urgency.most_common():
        print(f"  {u}: {count} ({count/len(tickets)*100:.1f}%)")

    # --- 4. Answerable from docs ---
    answerable = sum(1 for t in tickets if t["labels"]["answerable_from_docs"])
    print(f"\n=== ANSWERABLE FROM DOCS ===")
    print(f"  {answerable}/{len(tickets)} = {answerable/len(tickets)*100:.1f}%")

    # --- 5. Must-not-auto-respond ---
    must_not = [t for t in tickets if t["labels"]["must_not_auto_respond"]]
    must_not_intents = Counter(t["labels"]["intent"] for t in must_not)
    print(f"\n=== MUST NOT AUTO-RESPOND: {len(must_not)} tickets ===")
    for intent, count in must_not_intents.most_common():
        print(f"  {intent}: {count}")

    # --- 6. FCR (First Contact Resolution) ---
    fcr_tickets = [t for t in tickets if "history" in t]
    fcr = sum(1 for t in fcr_tickets if t["history"]["first_contact_resolution"])
    print(f"\n=== FIRST CONTACT RESOLUTION ===")
    print(f"  {fcr}/{len(fcr_tickets)} = {fcr/len(fcr_tickets)*100:.1f}%")

    # --- 7. CSAT ---
    csat_scores = [t["history"]["csat_rating"] for t in fcr_tickets]
    avg_csat = sum(csat_scores) / len(csat_scores)
    csat_dist = Counter(csat_scores)
    print(f"\n=== CSAT ===")
    print(f"  Average: {avg_csat:.2f}/5.0")
    for star in sorted(csat_dist):
        print(f"  {star}-star: {csat_dist[star]}")

    # --- 8. Escalation rate ---
    escalated = sum(1 for t in fcr_tickets if t["history"]["escalated"])
    print(f"\n=== ESCALATION ===")
    print(f"  {escalated}/{len(fcr_tickets)} = {escalated/len(fcr_tickets)*100:.1f}%")

    # --- 9. Breakdown by customer tier ---
    print("\n=== BY CUSTOMER TIER ===")
    tiers = set(t.get("customer_tier", "unknown") for t in tickets)
    for tier in sorted(tiers):
        tier_tickets = [t for t in tickets if t.get("customer_tier") == tier and "history" in t]
        if not tier_tickets:
            continue
        t_fcr = sum(1 for t in tier_tickets if t["history"]["first_contact_resolution"])
        t_csat = sum(t["history"]["csat_rating"] for t in tier_tickets) / len(tier_tickets)
        t_esc = sum(1 for t in tier_tickets if t["history"]["escalated"])
        t_resolve = sum(t["history"]["resolution_time_minutes"] for t in tier_tickets) / len(tier_tickets)
        print(
            f"  {tier}: FCR {t_fcr/len(tier_tickets)*100:.1f}%, "
            f"CSAT {t_csat:.2f}, "
            f"Escalation {t_esc/len(tier_tickets)*100:.1f}%, "
            f"Avg resolve {t_resolve:.0f} min"
        )

    # --- 10. Doc-answerable but still escalated (the findability gap) ---
    doc_answerable = [t for t in tickets if t["labels"]["answerable_from_docs"] and "history" in t]
    doc_but_escalated = sum(1 for t in doc_answerable if t["history"]["escalated"])
    print(f"\n=== DOC-ANSWERABLE BUT STILL ESCALATED ===")
    print(f"  {doc_but_escalated}/{len(doc_answerable)} = {doc_but_escalated/len(doc_answerable)*100:.1f}%")

    # --- 11. Repeat contacts ---
    repeats = sum(1 for t in fcr_tickets if t["history"]["repeat_contact"])
    print(f"\n=== REPEAT CONTACTS ===")
    print(f"  {repeats}/{len(fcr_tickets)} = {repeats/len(fcr_tickets)*100:.1f}%")

    # --- 12. Most referenced doc IDs ---
    all_doc_ids = list(chain.from_iterable(t["labels"].get("expected_doc_ids", []) for t in tickets))
    doc_counts = Counter(all_doc_ids)
    print(f"\n=== MOST REFERENCED DOCS ===")
    for doc, count in doc_counts.most_common(5):
        print(f"  {doc}: {count}")

    # --- 13. Escalation rate BY INTENT (bonus - which intents escalate most?) ---
    print("\n=== ESCALATION RATE BY INTENT ===")
    for intent, _ in intents.most_common():
        intent_tickets = [t for t in fcr_tickets if t["labels"]["intent"] == intent]
        if not intent_tickets:
            continue
        esc = sum(1 for t in intent_tickets if t["history"]["escalated"])
        print(f"  {intent}: {esc}/{len(intent_tickets)} = {esc/len(intent_tickets)*100:.1f}%")


if __name__ == "__main__":
    main()

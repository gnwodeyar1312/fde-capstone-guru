"""
Classifier Sample Test — run 1 ticket per intent (22 total) through the classifier
and compare predictions against ground truth labels.

Usage:
    cd C:\fde-capstone\fde-capstone-guru
    python scripts/test_classifier_sample.py
"""

import json
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import parse_ticket
from src.classify import classify_ticket, get_groq_client, MUST_NOT_AUTO_RESPOND_INTENTS


def main():
    # Load tickets
    data_path = Path("data/development_tickets.json")
    with open(data_path) as f:
        tickets = json.load(f)

    # Pick one ticket per intent (first occurrence) — ensures all 22 intents tested
    seen_intents = set()
    sample = []
    for t in tickets:
        intent = t["labels"]["intent"]
        if intent not in seen_intents:
            seen_intents.add(intent)
            sample.append(t)

    print(f"Testing {len(sample)} tickets (one per intent)")
    print("=" * 100)

    # Create Groq client (reuse across all calls)
    client = get_groq_client()

    # Track results
    results = []
    intent_correct = 0
    urgency_correct = 0
    answerable_correct = 0
    must_not_correct = 0

    for i, raw_ticket in enumerate(sample):
        ticket_id = raw_ticket["ticket_id"]
        true_intent = raw_ticket["labels"]["intent"]
        true_urgency = raw_ticket["labels"]["urgency"]
        true_answerable = raw_ticket["labels"]["answerable_from_docs"]
        true_must_not = raw_ticket["labels"]["must_not_auto_respond"]

        # Parse through ingest — parse_ticket returns IngestedTicket,
        # but classify_ticket expects StandardTicket (the .ticket field)
        ingested = parse_ticket(raw_ticket)
        std_ticket = ingested.ticket

        print(f"\n[{i+1}/{len(sample)}] {ticket_id}")
        print(f"  Subject: {raw_ticket.get('subject', 'N/A')}")

        try:
            result = classify_ticket(std_ticket, client=client)

            # Compare
            intent_match = result.intent == true_intent
            urgency_match = result.urgency == true_urgency
            answerable_match = result.answerable_from_docs == true_answerable
            must_not_match = result.must_not_auto_respond == true_must_not

            if intent_match:
                intent_correct += 1
            if urgency_match:
                urgency_correct += 1
            if answerable_match:
                answerable_correct += 1
            if must_not_match:
                must_not_correct += 1

            status = "CORRECT" if intent_match else "WRONG"

            print(f"  Intent:      {status}  predicted={result.intent} (conf={result.intent_confidence:.2f})  truth={true_intent}")
            print(f"  Urgency:     {'CORRECT' if urgency_match else 'WRONG'}  predicted={result.urgency} (conf={result.urgency_confidence:.2f})  truth={true_urgency}")
            print(f"  Answerable:  {'CORRECT' if answerable_match else 'WRONG'}  predicted={result.answerable_from_docs}  truth={true_answerable}")
            print(f"  Must-not:    {'CORRECT' if must_not_match else 'WRONG'}  predicted={result.must_not_auto_respond}  truth={true_must_not}")
            print(f"  Reasoning:   {result.reasoning[:120]}")

            results.append({
                "ticket_id": ticket_id,
                "true_intent": true_intent,
                "predicted_intent": result.intent,
                "intent_correct": intent_match,
                "intent_confidence": result.intent_confidence,
                "true_urgency": true_urgency,
                "predicted_urgency": result.urgency,
                "urgency_correct": urgency_match,
                "urgency_confidence": result.urgency_confidence,
                "true_answerable": true_answerable,
                "predicted_answerable": result.answerable_from_docs,
                "answerable_correct": answerable_match,
                "true_must_not": true_must_not,
                "predicted_must_not": result.must_not_auto_respond,
                "must_not_correct": must_not_match,
                "reasoning": result.reasoning,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "ticket_id": ticket_id,
                "true_intent": true_intent,
                "error": str(e),
            })

        # Rate limit: Groq free tier — pause between calls
        time.sleep(2)

    # Summary
    total = len(sample)
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"  Intent accuracy:      {intent_correct}/{total} = {intent_correct/total*100:.1f}%")
    print(f"  Urgency accuracy:     {urgency_correct}/{total} = {urgency_correct/total*100:.1f}%")
    print(f"  Answerable accuracy:  {answerable_correct}/{total} = {answerable_correct/total*100:.1f}%")
    print(f"  Must-not-auto accuracy: {must_not_correct}/{total} = {must_not_correct/total*100:.1f}%")

    # Show misclassifications
    misses = [r for r in results if "error" not in r and not r["intent_correct"]]
    if misses:
        print(f"\n  INTENT MISCLASSIFICATIONS ({len(misses)}):")
        for m in misses:
            print(f"    {m['ticket_id']}: predicted={m['predicted_intent']} (conf={m['intent_confidence']:.2f})  truth={m['true_intent']}")
    else:
        print("\n  No intent misclassifications!")

    # Save detailed results
    output_path = Path("evaluation/classifier_sample_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "summary": {
                "total_tested": total,
                "intent_accuracy": round(intent_correct / total * 100, 1),
                "urgency_accuracy": round(urgency_correct / total * 100, 1),
                "answerable_accuracy": round(answerable_correct / total * 100, 1),
                "must_not_auto_accuracy": round(must_not_correct / total * 100, 1),
            },
            "results": results,
        }, f, indent=2)
    print(f"\n  Detailed results saved to: {output_path}")


if __name__ == "__main__":
    main()

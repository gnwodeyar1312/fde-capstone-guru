"""
Live end-to-end test: Ingest → Classify → Retrieve → Generate → Validate → Log
Runs the full pipeline on the first development ticket.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from src.ingest import parse_ticket
from src.classify import classify_ticket
from src.retrieve import retrieve_for_ticket
from src.generate import generate_response
from src.guardrails import validate_response
from src.logging_store import init_database, log_decision, build_decision_record, get_decision, get_summary_stats
from src.route import route_ticket

# Load a real ticket
with open("data/development_tickets.json") as f:
    tickets = json.load(f)

ingested = parse_ticket(tickets[0])
ticket = ingested.ticket
print("=== TICKET ===")
print(f"Subject: {ticket.subject}")
print(f"Body: {ticket.body[:150]}...")
print()

# Classify it
classification = classify_ticket(ticket)
print("=== CLASSIFICATION ===")
print(f"Intent: {classification.intent} (conf: {classification.intent_confidence})")
print(f"Urgency: {classification.urgency}")
print(f"Answerable: {classification.answerable_from_docs}")
print()

# Route it
route = route_ticket(classification)
print("=== ROUTING ===")
print(f"Action: {route.action.value}")
print(f"Reason: {route.reason}")
if route.escalation_target:
    print(f"Escalation target: {route.escalation_target}")
print(f"Rule triggered: {route.rule_triggered}")
print()

# Retrieve relevant docs
retrieval = retrieve_for_ticket(ticket)
print("=== RETRIEVAL ===")
for chunk in retrieval.chunks:
    print(f"  {chunk.doc_id}/{chunk.section_name} (rel: {chunk.relevance_score:.3f})")
print()

# Generate and validate only if auto_respond
if route.is_auto_respond:
    # Generate the response
    result = generate_response(ticket, classification, retrieval)
    print("=== GENERATED RESPONSE ===")
    print(result.response_text)
    print()
    print("=== METADATA ===")
    print(f"Citations: {result.cited_doc_ids}")
    print(f"Could answer: {result.could_answer}")
    print(f"Model: {result.model_used}")
    print()

    # Validate the response
    guardrail = validate_response(result, retrieval)
    print("=== GUARDRAILS ===")
    print(f"Overall: {'PASSED' if guardrail.passed else 'FAILED'}")
    for check in guardrail.checks:
        status = "PASS" if check.passed else f"FAIL ({check.severity})"
        print(f"  [{status}] {check.name}: {check.reason}")
    print()

    # Determine final action
    if guardrail.passed:
        final_action = "sent"
    else:
        final_action = "blocked_by_guardrails"
else:
    result = None
    guardrail = None
    final_action = "escalated"
    print("=== SKIPPED GENERATION (escalated) ===")
    print()

# Log the decision
init_database()
record = build_decision_record(
    ticket_id=ticket.ticket_id,
    classification=classification,
    route=route,
    retrieval_doc_ids=retrieval.unique_doc_ids,
    num_chunks=retrieval.num_results,
    generation=result,
    guardrails=guardrail,
    final_action=final_action,
)
row_id = log_decision(record)

print("=== DECISION LOG ===")
print(f"Logged as row #{row_id}")
print(f"Final action: {final_action}")

# Show what was stored
stored = get_decision(ticket.ticket_id)
print(f"Stored ticket_id: {stored['ticket_id']}")
print(f"Stored intent: {stored['intent']} (conf: {stored['intent_confidence']})")
print(f"Stored route: {stored['route_action']}")
print(f"Stored final: {stored['final_action']}")
print()

# Show summary stats
stats = get_summary_stats()
print("=== SUMMARY STATS ===")
for key, value in stats.items():
    print(f"  {key}: {value}")

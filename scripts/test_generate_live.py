"""
Live end-to-end test: Ingest → Classify → Retrieve → Generate → Validate
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

# Retrieve relevant docs
retrieval = retrieve_for_ticket(ticket)
print("=== RETRIEVAL ===")
for chunk in retrieval.chunks:
    print(f"  {chunk.doc_id}/{chunk.section_name} (rel: {chunk.relevance_score:.3f})")
print()

# Generate the response
result = generate_response(ticket, classification, retrieval)
print("=== GENERATED RESPONSE ===")
print(result.response_text)
print()
print("=== METADATA ===")
print(f"Citations: {result.cited_doc_ids}")
print(f"Could answer: {result.could_answer}")
print(f"Model: {result.model_used}")
print(f"Reasoning: {result.reasoning}")
print()

# Validate the response
guardrail = validate_response(result, retrieval)
print("=== GUARDRAILS ===")
print(f"Overall: {'PASSED' if guardrail.passed else 'FAILED'}")
for check in guardrail.checks:
    status = "PASS" if check.passed else f"FAIL ({check.severity})"
    print(f"  [{status}] {check.name}: {check.reason}")
if guardrail.warnings:
    print(f"Warnings: {guardrail.warnings}")
print(f"Recommendation: {guardrail.recommendation}")

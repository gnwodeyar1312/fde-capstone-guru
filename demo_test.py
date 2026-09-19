#!/usr/bin/env python3
"""
Quick smoke test — run this BEFORE recording to confirm the pipeline works.
If this passes, demo_auto_respond.py and demo_escalation.py will work.
"""
import sys
print("Checking imports...", end=" ")
try:
    from src.ingest import StandardTicket
    from src.pipeline import run_ticket, get_pipeline
    from src.config import get_llm
    print("OK")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

print("Compiling pipeline...", end=" ")
try:
    pipeline = get_pipeline()
    print("OK")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

print("Running a test ticket...", end=" ", flush=True)
try:
    ticket = StandardTicket(
        ticket_id="TEST-000",
        subject="Test",
        body="How do I reset my password?",
        channel="chat",
        customer_id="C-0000",
        received_at="2026-09-15T00:00:00Z",
    )
    result = run_ticket(ticket)
    fa = result.get("final_action", "unknown")
    print(f"OK  (final_action={fa})")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

print("\n✓ All checks passed — you're ready to record!")

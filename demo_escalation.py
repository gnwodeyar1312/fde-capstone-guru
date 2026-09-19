#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  DEMO 2: Escalation + Guardrail Demo                        ║
║  Part A — A security_incident ticket (MNR) is escalated.     ║
║  Part B — Show the 5 guardrail checks that protect output.   ║
╚══════════════════════════════════════════════════════════════╝

Run during Section 5 of the video (~3 minutes).
Usage:  python demo_escalation.py
"""
import time
from datetime import datetime

# ── pretty printing helpers ──────────────────────────────────
CYAN    = "\033[96m"
GREEN   = "\033[92m"
RED     = "\033[91m"
YELLOW  = "\033[93m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RESET   = "\033[0m"

def banner(text):
    w = len(text) + 4
    print(f"\n{CYAN}{'═' * w}")
    print(f"  {BOLD}{text}{RESET}{CYAN}")
    print(f"{'═' * w}{RESET}\n")

def stage(num, name):
    print(f"  {YELLOW}▶ Stage {num}: {name}{RESET}")

def kv(key, val, indent=4):
    print(f"{' ' * indent}{DIM}{key}:{RESET} {BOLD}{val}{RESET}")

def ok(msg):
    print(f"  {GREEN}✓ {msg}{RESET}")

def warn(msg):
    print(f"  {RED}✗ {msg}{RESET}")

def pause(sec=1.0):
    time.sleep(sec)

# ── PART A: MNR Escalation ───────────────────────────────────
def part_a():
    banner("Part A: Must-Not-Respond Escalation")

    print(f"{BOLD}Incoming ticket — security incident:{RESET}\n")
    ticket_data = {
        "ticket_id": "DEMO-002",
        "subject": "Security incident — unauthorized access detected",
        "body": (
            "We detected unauthorized API calls from IP 203.0.113.42 "
            "accessing our production database endpoints. Multiple admin "
            "accounts appear compromised. Need immediate investigation."
        ),
        "channel": "email",
        "customer_id": "C-7781",
        "customer_name": "Jordan Chen",
        "customer_tier": "enterprise",
        "received_at": datetime.utcnow().isoformat() + "Z",
    }
    for k, v in ticket_data.items():
        kv(k, v)

    pause(1.5)

    print(f"\n{BOLD}Running pipeline...{RESET}\n")

    from src.ingest import StandardTicket
    from src.pipeline import run_ticket

    ticket = StandardTicket(**ticket_data)

    stage(1, "Ingest")
    ok("StandardTicket created")
    pause(0.3)

    t0 = time.time()
    result = run_ticket(ticket)
    elapsed = time.time() - t0

    pause(0.5)

    # Classification
    cls = result.get("classification")
    if cls:
        stage(2, "Classify")
        intent = getattr(cls, "intent", str(cls))
        kv("Intent", intent)
        kv("Confidence", f"{getattr(cls, 'confidence', 'N/A')}")
        kv("Urgency", getattr(cls, "urgency", "N/A"))

    # Route
    rd = result.get("route_decision")
    stage(4, "Route")
    if rd:
        action = getattr(rd, "action", str(rd))
        reason = getattr(rd, "reason", "N/A")
        kv("Decision", action)
        kv("Reason", reason)
    pause(0.3)

    # Check what happened
    fa = result.get("final_action", "unknown")
    gen = result.get("generation")
    gr = result.get("guardrails")

    if gen is None:
        ok("Generate stage SKIPPED — no response generated")
    if gr is None:
        ok("Validate stage SKIPPED — no guardrails needed")
    pause(0.3)

    banner("Outcome")
    kv("Final action", fa)
    kv("Latency", f"{elapsed:.2f}s")

    # Explain why
    print(f"\n  {YELLOW}Why escalated?{RESET}")
    mnr_intents = ["compliance_request", "security_incident",
                   "feature_request", "unclear_request"]
    print(f"    Must-not-auto-respond intents: {', '.join(mnr_intents)}")
    if cls:
        detected = getattr(cls, "intent", "")
        if detected in mnr_intents:
            print(f"    {GREEN}→ \"{detected}\" is in the MNR list{RESET}")
        else:
            print(f"    {DIM}→ Classified as \"{detected}\"{RESET}")
    print(f"    {GREEN}→ Ticket goes straight to a human agent{RESET}")
    print(f"    {GREEN}→ Generate + Validate stages are skipped entirely{RESET}")

    print(f"\n{GREEN}{BOLD}Part A complete.{RESET}")


# ── PART B: Guardrail Walkthrough ────────────────────────────
def part_b():
    banner("Part B: Guardrail Safety Checks")

    print(f"  Every auto-responded ticket passes through {BOLD}5 guardrail checks{RESET}")
    print(f"  {DIM}before{RESET} the response reaches the customer.\n")

    pause(1.0)

    guardrails = [
        {
            "name": "citation_validation",
            "desc": "Response references real KB article IDs that exist in ChromaDB",
            "why":  "Prevents hallucinated sources",
        },
        {
            "name": "no_fabricated_urls",
            "desc": "No made-up URLs appear in the response text",
            "why":  "Prevents sending customers to non-existent pages",
        },
        {
            "name": "no_pii_leakage",
            "desc": "Customer PII (email, phone, account #) is not echoed back",
            "why":  "Privacy protection — don't repeat sensitive data",
        },
        {
            "name": "response_length",
            "desc": "Response is within acceptable bounds (not too short/long)",
            "why":  "Catches degenerate outputs (empty, repeated, or runaway)",
        },
        {
            "name": "automated_footer",
            "desc": "Disclaimer footer is present on every auto-response",
            "why":  "Transparency — customer knows this is AI-generated",
        },
    ]

    for i, g in enumerate(guardrails, 1):
        print(f"  {CYAN}{BOLD}{i}. {g['name']}{RESET}")
        print(f"     {g['desc']}")
        print(f"     {DIM}Why: {g['why']}{RESET}")
        pause(0.8)
        print()

    print(f"  {YELLOW}If ANY check fails → response is blocked → ticket escalates{RESET}")
    print(f"  {DIM}This is the last safety net before a response reaches a customer.{RESET}")

    print(f"\n{GREEN}{BOLD}Part B complete.{RESET}")


# ── Main ─────────────────────────────────────────────────────
def main():
    banner("CloudServe Demo: Escalation + Guardrails")

    part_a()
    pause(1.5)
    part_b()

    print(f"\n{GREEN}{BOLD}Demo 2 complete.{RESET}\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  DEMO 1: Auto-Respond Flow                                  ║
║  Shows a billing ticket processed end-to-end through the     ║
║  six-stage pipeline, resulting in an auto-generated response. ║
╚══════════════════════════════════════════════════════════════╝

Run during Section 4 of the video (~3 minutes).
Usage:  python demo_auto_respond.py
"""
import sys
import time
from datetime import datetime

# ── pretty printing helpers ──────────────────────────────────
CYAN    = "\033[96m"
GREEN   = "\033[92m"
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

def pause(sec=1.0):
    time.sleep(sec)

# ── main demo ────────────────────────────────────────────────
def main():
    banner("CloudServe Demo: Auto-Respond Flow")

    # 1 — Show the raw ticket
    print(f"{BOLD}Incoming support ticket:{RESET}\n")
    ticket_data = {
        "ticket_id": "DEMO-001",
        "subject": "How do I upgrade my plan?",
        "body": (
            "Hi, I'm currently on the Starter plan and would like to "
            "upgrade to Professional. Can you walk me through the steps? "
            "I also want to know if there's a prorated charge."
        ),
        "channel": "chat",
        "customer_id": "C-9042",
        "customer_name": "Alex Rivera",
        "customer_tier": "standard",
        "received_at": datetime.utcnow().isoformat() + "Z",
    }
    for k, v in ticket_data.items():
        kv(k, v)

    pause(1.5)

    # 2 — Import & run
    print(f"\n{BOLD}Running through the six-stage pipeline...{RESET}\n")

    from src.ingest import StandardTicket
    from src.pipeline import run_ticket

    ticket = StandardTicket(**ticket_data)

    stage(1, "Ingest — normalise raw ticket")
    ok(f"StandardTicket created  (channel={ticket.channel})")
    pause(0.5)

    stage(2, "Classify → Route → Generate → Validate")
    print(f"    {DIM}(stages 2-6 run inside LangGraph){RESET}")
    pause(0.3)

    t0 = time.time()
    result = run_ticket(ticket)
    elapsed = time.time() - t0

    pause(0.5)

    # 3 — Show results stage by stage
    banner("Pipeline Results")

    # Classification
    cls = result.get("classification")
    if cls:
        stage(2, "Classify")
        kv("Intent", getattr(cls, "intent", str(cls)))
        kv("Confidence", f"{getattr(cls, 'confidence', 'N/A')}")
        kv("Urgency", getattr(cls, "urgency", "N/A"))
        ok("Intent classified")
    pause(0.5)

    # Retrieval
    ret = result.get("retrieval")
    if ret:
        stage(3, "Retrieve")
        chunks = getattr(ret, "chunks", [])
        kv("Chunks returned", len(chunks))
        if chunks:
            top = chunks[0]
            kv("Top chunk score", f"{getattr(top, 'relevance_score', getattr(top, 'score', 'N/A'))}")
        ok("Knowledge base searched")
    pause(0.5)

    # Route
    rd = result.get("route_decision")
    stage(4, "Route")
    if rd:
        action = getattr(rd, "action", str(rd))
        kv("Decision", action)
        kv("Reason", getattr(rd, "reason", "N/A"))
    else:
        kv("Decision", result.get("final_action", "unknown"))
    ok("Routing complete")
    pause(0.5)

    # Generate
    gen = result.get("generation")
    if gen:
        stage(5, "Generate")
        resp_text = getattr(gen, "response", getattr(gen, "text", str(gen)))
        ok("Cited response generated")
    pause(0.5)

    # Guardrails
    gr = result.get("guardrails")
    if gr:
        stage(6, "Validate")
        passed = getattr(gr, "passed", None)
        kv("Guardrails passed", passed)
        checks = getattr(gr, "checks", [])
        for c in checks:
            name = getattr(c, "name", c.get("name", "?")) if isinstance(c, dict) else getattr(c, "name", "?")
            cpassed = getattr(c, "passed", c.get("passed", "?")) if isinstance(c, dict) else getattr(c, "passed", "?")
            symbol = f"{GREEN}✓{RESET}" if cpassed else f"\033[91m✗{RESET}"
            print(f"      {symbol} {name}")
        ok("All guardrail checks evaluated")
    pause(0.5)

    # Final outcome
    banner("Final Outcome")
    fa = result.get("final_action", "unknown")
    kv("Action", fa)
    kv("Latency", f"{elapsed:.2f}s")

    if gen:
        resp_text = getattr(gen, "response", getattr(gen, "text", str(gen)))
        print(f"\n{BOLD}Generated Response:{RESET}")
        print(f"{DIM}{'─' * 56}{RESET}")
        # wrap at ~70 chars
        words = resp_text.split()
        line = ""
        for w in words:
            if len(line) + len(w) + 1 > 70:
                print(f"  {line}")
                line = w
            else:
                line = f"{line} {w}" if line else w
        if line:
            print(f"  {line}")
        print(f"{DIM}{'─' * 56}{RESET}")

    print(f"\n{GREEN}{BOLD}Demo 1 complete.{RESET}\n")


if __name__ == "__main__":
    main()

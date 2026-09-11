"""
Live Traffic Simulator for CloudServe Support System.

Purpose:
    Streams support tickets into the system with configurable delay
    and exposes live Prometheus metrics at http://localhost:8000/metrics.
    Enables instant visualization on Prometheus and Grafana dashboards.

Usage:
    # Run 10 tickets from development dataset with 2s delay
    python scripts/simulate_traffic.py --count 10 --delay 2.0

    # Run continuously in a loop for continuous Grafana demo
    python scripts/simulate_traffic.py --loop --delay 3.0 --port 8000
"""

import argparse
import logging
import random
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import validate_config, get_llm
from src.ingest import ingest_tickets
from src.metrics import (
    start_metrics_server,
    record_routing_metrics,
)
from src.pipeline import run_ticket, get_pipeline
from src.retrieve import build_vector_store
from src.logging_store import init_database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TrafficSimulator")


def main():
    parser = argparse.ArgumentParser(description="CloudServe Live Traffic Simulator for Prometheus/Grafana")
    parser.add_argument(
        "--input",
        default="data/test_batch_20.json",
        help="Input tickets dataset file (default: data/test_batch_20.json)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of tickets to process (default: 10)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay in seconds between tickets (default: 2.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Prometheus metrics HTTP port (default: 8000)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously in a loop",
    )

    args = parser.parse_args()

    # 1. Start Prometheus metrics server
    logger.info("Starting Prometheus metrics server on port %d...", args.port)
    start_metrics_server(port=args.port)
    logger.info("Metrics endpoint ready: http://localhost:%d/metrics", args.port)
    logger.info("Grafana dashboard URL : http://localhost:3000")

    # 2. Initialize system resources
    logger.info("Initializing vector store & pipeline...")
    init_database()
    build_vector_store()
    get_pipeline()
    llm = get_llm()

    # 3. Load tickets
    input_file = Path(args.input)
    if not input_file.exists():
        input_file = Path("data/development_tickets.json")
    
    logger.info("Loading tickets from %s...", input_file)
    ticket_items = ingest_tickets(str(input_file))
    logger.info("Loaded %d tickets for simulation.", len(ticket_items))

    tickets_to_run = ticket_items[:args.count] if not args.loop else ticket_items

    # 4. Stream tickets
    logger.info("Starting traffic stream (delay: %.1fs)... Press Ctrl+C to stop.", args.delay)
    total_processed = 0

    try:
        while True:
            for item in tickets_to_run:
                ticket = item.ticket
                labels = item.labels
                logger.info(
                    "[%d] Processing Ticket %s (Channel: %s, Tier: %s, Subject: '%s')...",
                    total_processed + 1,
                    ticket.ticket_id,
                    ticket.channel,
                    ticket.customer_tier,
                    ticket.subject[:40],
                )

                start_time = time.time()
                try:
                    result = run_ticket(ticket, llm=llm)
                    elapsed = time.time() - start_time
                    action = result.get("final_action", "unknown")
                    intent = result.get("classification").intent if result.get("classification") else "n/a"
                    logger.info(
                        "--> Ticket %s: Action='%s', Intent='%s' (%.2fs)",
                        ticket.ticket_id, action, intent, elapsed
                    )

                    # Record policy compliance if ground truth exists
                    if labels and result.get("route_decision"):
                        gt_dict = {
                            "must_not_auto_respond": labels.must_not_auto_respond,
                        }
                        record_routing_metrics(result.get("route_decision"), ground_truth=gt_dict)

                except Exception as e:
                    logger.error("--> Ticket %s error: %s", ticket.ticket_id, e)

                total_processed += 1
                time.sleep(args.delay)

            if not args.loop:
                break
            logger.info("Loop completed. Restarting ticket stream...")

    except KeyboardInterrupt:
        logger.info("\nSimulation stopped by user.")

    logger.info("Processed %d tickets. Prometheus metrics remain available at http://localhost:%d/metrics", total_processed, args.port)
    logger.info("Keeping metrics server active for 30 seconds for Prometheus scraping...")
    time.sleep(30)


if __name__ == "__main__":
    main()

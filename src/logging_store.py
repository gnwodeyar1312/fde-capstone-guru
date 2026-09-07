"""
Decision Logging module for CloudServe Support System.

Purpose:
    Record every pipeline decision in a SQLite database for auditability,
    debugging, and evaluation. Every ticket that passes through the pipeline
    gets exactly one row in the decision log.

Design decisions:
    1. SQLite, not PostgreSQL or a cloud DB.
       Why? Embedded, zero config, file-based. For a capstone project
       processing 500 tickets, SQLite is the right tool. It also means
       the evaluation harness can query results without a running server.
    2. One row per ticket, not one row per stage.
       Why? The decision log answers "what happened to ticket X?" — a
       single query returns the complete story. Splitting across tables
       would require joins for every lookup.
    3. Complex fields (lists, dicts) are stored as JSON strings.
       Why? SQLite doesn't have native JSON arrays. Storing as JSON
       strings keeps the schema flat while preserving structured data.
       Python's json module handles serialization/deserialization.
    4. The log is APPEND-ONLY in normal operation.
       Why? Decision records are historical facts. Updating them would
       break the audit trail. The only exception is re-running a ticket
       through the pipeline (which creates a new row with a new timestamp).
    5. We use raw SQL, not an ORM.
       Why? The schema is one table with simple queries. SQLAlchemy would
       add complexity with zero benefit. Raw SQL is transparent — you can
       read the queries and know exactly what they do.

Interview context:
    "How do you debug a bad response?"
    → Pull the decision log row. It shows the classification (was the
      intent correct?), the retrieval (were the right docs found?), the
      routing (should it have been escalated?), and the guardrail results
      (what checks passed/failed?). Every stage's output is recorded.
    "What if you need to analyze patterns?"
    → The SQLite DB is queryable. "Show me all tickets where guardrails
      failed" is one SQL query. "What's the average confidence for
      escalated tickets?" is another. This feeds continuous improvement.
    "How does this connect to the evaluation harness?"
    → The evaluation harness (B-05) reads the decision log after processing
      all tickets. It compares logged decisions against ground-truth labels
      to compute accuracy metrics.
"""

import json
import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from src.config import DATABASE_URL
from src.classify import ClassificationResult
from src.route import RouteDecision
from src.generate import GeneratedResponse
from src.guardrails import GuardrailResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decision record model
# ---------------------------------------------------------------------------

class DecisionRecord(BaseModel):
    """
    A complete record of all pipeline decisions for one ticket.

    This is what gets stored in the database. Every field traces back
    to a specific pipeline stage.
    """
    # Ticket identity
    ticket_id: str
    timestamp: str = ""

    # Classification stage
    intent: str = ""
    intent_confidence: float = 0.0
    urgency: str = ""
    answerable_from_docs: bool = True
    classification_reasoning: str = ""

    # Routing stage
    route_action: str = ""  # "auto_respond" or "escalate"
    route_reason: str = ""
    escalation_target: Optional[str] = None
    rule_triggered: str = ""

    # Retrieval stage
    retrieved_doc_ids: list[str] = []
    num_chunks_retrieved: int = 0

    # Generation stage (only for auto_respond)
    response_text: str = ""
    cited_doc_ids: list[str] = []
    could_answer: bool = True
    model_used: str = ""

    # Guardrails stage (only for auto_respond)
    guardrails_passed: bool = True
    guardrail_checks: list[dict] = []
    guardrail_failed_checks: list[str] = []

    # Final outcome
    final_action: str = ""  # "sent", "escalated", "blocked_by_guardrails"

    def model_post_init(self, __context):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Database initialization
# ---------------------------------------------------------------------------

def _get_db_path() -> str:
    """Extract the file path from the DATABASE_URL config."""
    # DATABASE_URL is like "sqlite:///./storage/decisions.db"
    path = DATABASE_URL.replace("sqlite:///", "")
    return path


def init_database(db_path: Optional[str] = None) -> str:
    """
    Initialize the SQLite database and create the decisions table.

    This is IDEMPOTENT — safe to call multiple times. Uses CREATE TABLE
    IF NOT EXISTS so it won't destroy existing data.

    Args:
        db_path: Optional override for the database path

    Returns:
        The database file path
    """
    if db_path is None:
        db_path = _get_db_path()

    # Ensure the directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,

                -- Classification
                intent TEXT,
                intent_confidence REAL,
                urgency TEXT,
                answerable_from_docs INTEGER,
                classification_reasoning TEXT,

                -- Routing
                route_action TEXT,
                route_reason TEXT,
                escalation_target TEXT,
                rule_triggered TEXT,

                -- Retrieval
                retrieved_doc_ids TEXT,
                num_chunks_retrieved INTEGER,

                -- Generation
                response_text TEXT,
                cited_doc_ids TEXT,
                could_answer INTEGER,
                model_used TEXT,

                -- Guardrails
                guardrails_passed INTEGER,
                guardrail_checks TEXT,
                guardrail_failed_checks TEXT,

                -- Final
                final_action TEXT
            )
        """)

        # Index on ticket_id for fast lookups
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ticket_id ON decisions(ticket_id)
        """)

        conn.commit()
        logger.info("Database initialized at %s", db_path)
    finally:
        conn.close()

    return db_path


# ---------------------------------------------------------------------------
# Core logging function
# ---------------------------------------------------------------------------

def log_decision(record: DecisionRecord, db_path: Optional[str] = None) -> int:
    """
    Write a decision record to the database.

    Args:
        record: The DecisionRecord to store
        db_path: Optional override for the database path

    Returns:
        The row ID of the inserted record
    """
    if db_path is None:
        db_path = _get_db_path()

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO decisions (
                ticket_id, timestamp,
                intent, intent_confidence, urgency,
                answerable_from_docs, classification_reasoning,
                route_action, route_reason, escalation_target, rule_triggered,
                retrieved_doc_ids, num_chunks_retrieved,
                response_text, cited_doc_ids, could_answer, model_used,
                guardrails_passed, guardrail_checks, guardrail_failed_checks,
                final_action
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.ticket_id,
                record.timestamp,
                record.intent,
                record.intent_confidence,
                record.urgency,
                int(record.answerable_from_docs),
                record.classification_reasoning,
                record.route_action,
                record.route_reason,
                record.escalation_target,
                record.rule_triggered,
                json.dumps(record.retrieved_doc_ids),
                record.num_chunks_retrieved,
                record.response_text,
                json.dumps(record.cited_doc_ids),
                int(record.could_answer),
                record.model_used,
                int(record.guardrails_passed),
                json.dumps(record.guardrail_checks),
                json.dumps(record.guardrail_failed_checks),
                record.final_action,
            ),
        )
        conn.commit()
        row_id = cursor.lastrowid
        logger.info("Logged decision for ticket %s (row_id=%d)", record.ticket_id, row_id)
        return row_id
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Builder: construct a DecisionRecord from pipeline stage outputs
# ---------------------------------------------------------------------------

def build_decision_record(
    ticket_id: str,
    classification: Optional[ClassificationResult] = None,
    route: Optional[RouteDecision] = None,
    retrieval_doc_ids: Optional[list[str]] = None,
    num_chunks: int = 0,
    generation: Optional[GeneratedResponse] = None,
    guardrails: Optional[GuardrailResult] = None,
    final_action: str = "",
) -> DecisionRecord:
    """
    Build a DecisionRecord from the outputs of each pipeline stage.

    This is a convenience function that maps pipeline objects to the
    flat record structure. Each argument is optional because escalated
    tickets skip generation and guardrails.

    Args:
        ticket_id: The ticket identifier
        classification: Output of classify stage
        route: Output of route stage
        retrieval_doc_ids: List of retrieved doc_ids
        num_chunks: Number of chunks retrieved
        generation: Output of generate stage (None if escalated)
        guardrails: Output of guardrails stage (None if escalated)
        final_action: "sent", "escalated", or "blocked_by_guardrails"

    Returns:
        A complete DecisionRecord ready for logging
    """
    record = DecisionRecord(ticket_id=ticket_id)

    if classification:
        record.intent = classification.intent
        record.intent_confidence = classification.intent_confidence
        record.urgency = classification.urgency
        record.answerable_from_docs = classification.answerable_from_docs
        record.classification_reasoning = classification.reasoning

    if route:
        record.route_action = route.action.value
        record.route_reason = route.reason
        record.escalation_target = route.escalation_target
        record.rule_triggered = route.rule_triggered

    if retrieval_doc_ids is not None:
        record.retrieved_doc_ids = retrieval_doc_ids
    record.num_chunks_retrieved = num_chunks

    if generation:
        record.response_text = generation.response_text
        record.cited_doc_ids = generation.cited_doc_ids
        record.could_answer = generation.could_answer
        record.model_used = generation.model_used

    if guardrails:
        record.guardrails_passed = guardrails.passed
        record.guardrail_checks = [
            {"name": c.name, "passed": c.passed, "reason": c.reason, "severity": c.severity}
            for c in guardrails.checks
        ]
        record.guardrail_failed_checks = guardrails.failed_checks

    record.final_action = final_action

    return record


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_decision(ticket_id: str, db_path: Optional[str] = None) -> Optional[dict]:
    """
    Retrieve the most recent decision record for a ticket.

    Returns the raw row as a dict, or None if not found.
    """
    if db_path is None:
        db_path = _get_db_path()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            "SELECT * FROM decisions WHERE ticket_id = ? ORDER BY id DESC LIMIT 1",
            (ticket_id,),
        )
        row = cursor.fetchone()
        if row:
            result = dict(row)
            # Deserialize JSON fields
            for field in ["retrieved_doc_ids", "cited_doc_ids", "guardrail_checks", "guardrail_failed_checks"]:
                if result.get(field):
                    result[field] = json.loads(result[field])
            return result
        return None
    finally:
        conn.close()


def get_all_decisions(db_path: Optional[str] = None) -> list[dict]:
    """
    Retrieve all decision records.

    Returns a list of dicts, one per row.
    """
    if db_path is None:
        db_path = _get_db_path()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute("SELECT * FROM decisions ORDER BY id")
        rows = cursor.fetchall()
        results = []
        for row in rows:
            result = dict(row)
            for field in ["retrieved_doc_ids", "cited_doc_ids", "guardrail_checks", "guardrail_failed_checks"]:
                if result.get(field):
                    result[field] = json.loads(result[field])
            results.append(result)
        return results
    finally:
        conn.close()


def get_summary_stats(db_path: Optional[str] = None) -> dict:
    """
    Get summary statistics from the decision log.

    Returns counts of auto-responded, escalated, blocked, etc.
    Useful for the evaluation report.
    """
    if db_path is None:
        db_path = _get_db_path()

    conn = sqlite3.connect(db_path)
    try:
        stats = {}

        # Total decisions
        stats["total"] = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]

        # By final action
        for action in ["sent", "escalated", "blocked_by_guardrails"]:
            count = conn.execute(
                "SELECT COUNT(*) FROM decisions WHERE final_action = ?", (action,)
            ).fetchone()[0]
            stats[f"final_{action}"] = count

        # By route action
        for action in ["auto_respond", "escalate"]:
            count = conn.execute(
                "SELECT COUNT(*) FROM decisions WHERE route_action = ?", (action,)
            ).fetchone()[0]
            stats[f"route_{action}"] = count

        # Average confidence
        row = conn.execute(
            "SELECT AVG(intent_confidence) FROM decisions WHERE intent_confidence > 0"
        ).fetchone()
        stats["avg_intent_confidence"] = round(row[0], 3) if row[0] else 0.0

        # Guardrail pass rate
        total_with_guardrails = conn.execute(
            "SELECT COUNT(*) FROM decisions WHERE route_action = 'auto_respond'"
        ).fetchone()[0]
        passed_guardrails = conn.execute(
            "SELECT COUNT(*) FROM decisions WHERE guardrails_passed = 1 AND route_action = 'auto_respond'"
        ).fetchone()[0]
        stats["guardrail_pass_rate"] = (
            round(passed_guardrails / total_with_guardrails, 3)
            if total_with_guardrails > 0
            else 0.0
        )

        return stats
    finally:
        conn.close()

"""
Unit tests for the decision logging module.

Tests database initialization, record insertion, queries, and stats.
Uses a temporary database for each test (no side effects).
"""

import json
import os
import pytest
import tempfile

from src.logging_store import (
    init_database,
    log_decision,
    get_decision,
    get_all_decisions,
    get_summary_stats,
    build_decision_record,
    DecisionRecord,
)
from src.classify import ClassificationResult
from src.route import RouteDecision, RouteAction
from src.generate import GeneratedResponse
from src.guardrails import GuardrailResult, GuardrailCheck


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path(tmp_path):
    """Create a temporary database for each test."""
    path = str(tmp_path / "test_decisions.db")
    init_database(path)
    return path


# ---------------------------------------------------------------------------
# Test data factories
# ---------------------------------------------------------------------------

def make_decision_record(**overrides) -> DecisionRecord:
    defaults = dict(
        ticket_id="TICKET-001",
        intent="billing_query",
        intent_confidence=0.92,
        urgency="low",
        answerable_from_docs=True,
        classification_reasoning="Customer asks about billing.",
        route_action="auto_respond",
        route_reason="All checks passed.",
        rule_triggered="all_checks_passed",
        retrieved_doc_ids=["DOC-BILL-001"],
        num_chunks_retrieved=3,
        response_text="Here is your billing info [DOC-BILL-001].",
        cited_doc_ids=["DOC-BILL-001"],
        could_answer=True,
        model_used="llama-3.1-8b-instant",
        guardrails_passed=True,
        guardrail_checks=[{"name": "citation_validation", "passed": True, "reason": "ok", "severity": "high"}],
        guardrail_failed_checks=[],
        final_action="sent",
    )
    defaults.update(overrides)
    return DecisionRecord(**defaults)


# ---------------------------------------------------------------------------
# Database initialization tests
# ---------------------------------------------------------------------------

class TestInitDatabase:
    """Tests for database initialization."""

    def test_creates_database_file(self, tmp_path):
        path = str(tmp_path / "new.db")
        assert not os.path.exists(path)
        init_database(path)
        assert os.path.exists(path)

    def test_creates_parent_directories(self, tmp_path):
        path = str(tmp_path / "deep" / "nested" / "decisions.db")
        init_database(path)
        assert os.path.exists(path)

    def test_idempotent(self, tmp_path):
        path = str(tmp_path / "test.db")
        init_database(path)
        # Insert a record
        record = make_decision_record()
        log_decision(record, path)
        # Re-initialize — should not destroy data
        init_database(path)
        result = get_decision("TICKET-001", path)
        assert result is not None


# ---------------------------------------------------------------------------
# Log decision tests
# ---------------------------------------------------------------------------

class TestLogDecision:
    """Tests for inserting decision records."""

    def test_insert_returns_row_id(self, db_path):
        record = make_decision_record()
        row_id = log_decision(record, db_path)
        assert row_id >= 1

    def test_insert_multiple_records(self, db_path):
        for i in range(5):
            record = make_decision_record(ticket_id=f"TICKET-{i:03d}")
            log_decision(record, db_path)
        decisions = get_all_decisions(db_path)
        assert len(decisions) == 5

    def test_same_ticket_creates_new_row(self, db_path):
        """Re-processing a ticket creates a new row, not an update."""
        record1 = make_decision_record(final_action="sent")
        record2 = make_decision_record(final_action="escalated")
        log_decision(record1, db_path)
        log_decision(record2, db_path)
        decisions = get_all_decisions(db_path)
        assert len(decisions) == 2


# ---------------------------------------------------------------------------
# Query tests
# ---------------------------------------------------------------------------

class TestGetDecision:
    """Tests for querying decision records."""

    def test_get_existing_ticket(self, db_path):
        record = make_decision_record(ticket_id="TICKET-042")
        log_decision(record, db_path)
        result = get_decision("TICKET-042", db_path)
        assert result is not None
        assert result["ticket_id"] == "TICKET-042"
        assert result["intent"] == "billing_query"

    def test_get_nonexistent_ticket(self, db_path):
        result = get_decision("NONEXISTENT", db_path)
        assert result is None

    def test_get_returns_most_recent(self, db_path):
        """If a ticket was processed twice, return the latest."""
        record1 = make_decision_record(ticket_id="TICKET-001", final_action="sent")
        record2 = make_decision_record(ticket_id="TICKET-001", final_action="escalated")
        log_decision(record1, db_path)
        log_decision(record2, db_path)
        result = get_decision("TICKET-001", db_path)
        assert result["final_action"] == "escalated"

    def test_json_fields_deserialized(self, db_path):
        record = make_decision_record(
            retrieved_doc_ids=["DOC-A", "DOC-B"],
            cited_doc_ids=["DOC-A"],
        )
        log_decision(record, db_path)
        result = get_decision("TICKET-001", db_path)
        assert result["retrieved_doc_ids"] == ["DOC-A", "DOC-B"]
        assert result["cited_doc_ids"] == ["DOC-A"]


# ---------------------------------------------------------------------------
# Summary stats tests
# ---------------------------------------------------------------------------

class TestSummaryStats:
    """Tests for summary statistics."""

    def test_empty_database(self, db_path):
        stats = get_summary_stats(db_path)
        assert stats["total"] == 0

    def test_counts_by_final_action(self, db_path):
        for i in range(3):
            log_decision(make_decision_record(ticket_id=f"T-{i}", final_action="sent"), db_path)
        for i in range(2):
            log_decision(make_decision_record(ticket_id=f"E-{i}", final_action="escalated"), db_path)
        stats = get_summary_stats(db_path)
        assert stats["total"] == 5
        assert stats["final_sent"] == 3
        assert stats["final_escalated"] == 2

    def test_average_confidence(self, db_path):
        log_decision(make_decision_record(ticket_id="T-1", intent_confidence=0.90), db_path)
        log_decision(make_decision_record(ticket_id="T-2", intent_confidence=0.80), db_path)
        stats = get_summary_stats(db_path)
        assert stats["avg_intent_confidence"] == 0.85

    def test_guardrail_pass_rate(self, db_path):
        # 2 auto_respond with guardrails passed, 1 auto_respond with guardrails failed
        log_decision(make_decision_record(
            ticket_id="T-1", route_action="auto_respond", guardrails_passed=True,
        ), db_path)
        log_decision(make_decision_record(
            ticket_id="T-2", route_action="auto_respond", guardrails_passed=True,
        ), db_path)
        log_decision(make_decision_record(
            ticket_id="T-3", route_action="auto_respond", guardrails_passed=False,
        ), db_path)
        stats = get_summary_stats(db_path)
        assert abs(stats["guardrail_pass_rate"] - 0.667) < 0.01


# ---------------------------------------------------------------------------
# Build decision record tests
# ---------------------------------------------------------------------------

class TestBuildDecisionRecord:
    """Tests for the builder function."""

    def test_from_classification(self):
        classification = ClassificationResult(
            intent="billing_query",
            intent_confidence=0.92,
            urgency="low",
            urgency_confidence=0.85,
            answerable_from_docs=True,
            answerable_confidence=0.88,
            reasoning="Billing question.",
        )
        record = build_decision_record(
            ticket_id="T-1",
            classification=classification,
            final_action="escalated",
        )
        assert record.intent == "billing_query"
        assert record.intent_confidence == 0.92
        assert record.urgency == "low"

    def test_from_route(self):
        route = RouteDecision(
            action=RouteAction.ESCALATE,
            reason="Low confidence.",
            escalation_target="tier1_support",
            rule_triggered="low_confidence",
        )
        record = build_decision_record(
            ticket_id="T-1",
            route=route,
            final_action="escalated",
        )
        assert record.route_action == "escalate"
        assert record.escalation_target == "tier1_support"

    def test_from_all_stages(self):
        classification = ClassificationResult(
            intent="billing_query",
            intent_confidence=0.92,
            urgency="low",
            urgency_confidence=0.85,
            answerable_from_docs=True,
            answerable_confidence=0.88,
            reasoning="Billing.",
        )
        route = RouteDecision(
            action=RouteAction.AUTO_RESPOND,
            reason="All checks passed.",
            rule_triggered="all_checks_passed",
        )
        generation = GeneratedResponse(
            response_text="Your answer [DOC-BILL-001].",
            cited_doc_ids=["DOC-BILL-001"],
            model_used="llama-3.1-8b-instant",
        )
        guardrails = GuardrailResult(checks=[
            GuardrailCheck(name="citation_validation", passed=True, reason="ok"),
        ])

        record = build_decision_record(
            ticket_id="T-1",
            classification=classification,
            route=route,
            retrieval_doc_ids=["DOC-BILL-001"],
            num_chunks=3,
            generation=generation,
            guardrails=guardrails,
            final_action="sent",
        )
        assert record.ticket_id == "T-1"
        assert record.intent == "billing_query"
        assert record.route_action == "auto_respond"
        assert record.response_text == "Your answer [DOC-BILL-001]."
        assert record.guardrails_passed is True
        assert record.final_action == "sent"

    def test_escalated_ticket_skips_generation(self):
        """Escalated tickets have no generation or guardrails data."""
        record = build_decision_record(
            ticket_id="T-1",
            final_action="escalated",
        )
        assert record.response_text == ""
        assert record.cited_doc_ids == []
        assert record.guardrails_passed is True  # default


# ---------------------------------------------------------------------------
# DecisionRecord model tests
# ---------------------------------------------------------------------------

class TestDecisionRecordModel:
    """Tests for the DecisionRecord Pydantic model."""

    def test_timestamp_auto_generated(self):
        record = DecisionRecord(ticket_id="T-1")
        assert record.timestamp != ""
        assert "T" in record.timestamp  # ISO format

    def test_defaults(self):
        record = DecisionRecord(ticket_id="T-1")
        assert record.intent == ""
        assert record.retrieved_doc_ids == []
        assert record.guardrails_passed is True
        assert record.final_action == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

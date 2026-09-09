"""
Tests for the ingest module.

Testing strategy:
    - Happy path: all 4 channels parse correctly
    - Edge cases: empty subject (chat), missing fields
    - Ground truth separation: labels must NOT appear in StandardTicket
    - File loading: valid path, missing path, invalid JSON
"""

import json

import pytest

from src.ingest import (
    StandardTicket,
    ingest_tickets,
    parse_ticket,
)

# ---------------------------------------------------------------------------
# Fixtures — reusable test data
# ---------------------------------------------------------------------------


def make_raw_ticket(**overrides):
    """Factory for raw ticket dicts. Override any field you want to test."""
    base = {
        "ticket_id": "TEST-001",
        "channel": "email",
        "subject": "Test subject",
        "body": "I need help with my deployment.",
        "received_at": "2026-05-01T10:00:00Z",
        "customer_id": "CUST-9999",
        "customer_name": "Test User",
        "customer_tier": "standard",
        "customer_region": "north_america",
        "language_fluency": "fluent",
        "labels": {
            "intent": "deployment_failure",
            "urgency": "high",
            "expected_route": "auto_respond",
            "answerable_from_docs": True,
            "expected_doc_ids": ["DOC-DEPLOY-001"],
            "must_not_auto_respond": False,
        },
        "history": {
            "first_contact_resolution": True,
            "resolution_time_minutes": 30,
            "csat_rating": 5,
            "escalated": False,
            "repeat_contact": False,
        },
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# StandardTicket validation
# ---------------------------------------------------------------------------


class TestStandardTicket:
    def test_valid_email_ticket(self):
        t = StandardTicket(
            ticket_id="T-1",
            channel="email",
            subject="Help",
            body="Need help",
            received_at="2026-01-01T00:00:00Z",
            customer_id="C-1",
        )
        assert t.channel == "email"
        assert t.ticket_id == "T-1"

    def test_valid_chat_ticket_empty_subject(self):
        """Chat tickets have empty subjects — this must be allowed."""
        t = StandardTicket(
            ticket_id="T-2",
            channel="chat",
            subject="",
            body="My build is failing",
            received_at="2026-01-01T00:00:00Z",
            customer_id="C-2",
        )
        assert t.subject == ""
        assert t.combined_text() == "My build is failing"

    def test_all_four_channels_accepted(self):
        for ch in ["email", "chat", "docs_comment", "forum"]:
            t = StandardTicket(
                ticket_id="T-1",
                channel=ch,
                body="test",
                received_at="2026-01-01T00:00:00Z",
                customer_id="C-1",
            )
            assert t.channel == ch

    def test_invalid_channel_rejected(self):
        with pytest.raises(ValueError, match="Unknown channel"):
            StandardTicket(
                ticket_id="T-1",
                channel="twitter",
                body="test",
                received_at="2026-01-01T00:00:00Z",
                customer_id="C-1",
            )

    def test_empty_body_rejected(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            StandardTicket(
                ticket_id="T-1",
                channel="email",
                body="   ",
                received_at="2026-01-01T00:00:00Z",
                customer_id="C-1",
            )

    def test_combined_text_with_subject(self):
        t = StandardTicket(
            ticket_id="T-1",
            channel="email",
            subject="SSO broken",
            body="Cannot log in via SAML",
            received_at="2026-01-01T00:00:00Z",
            customer_id="C-1",
        )
        combined = t.combined_text()
        assert "SSO broken" in combined
        assert "Cannot log in" in combined
        assert combined == "SSO broken\n\nCannot log in via SAML"

    def test_combined_text_without_subject(self):
        t = StandardTicket(
            ticket_id="T-1",
            channel="chat",
            subject="",
            body="Build failing",
            received_at="2026-01-01T00:00:00Z",
            customer_id="C-1",
        )
        assert t.combined_text() == "Build failing"


# ---------------------------------------------------------------------------
# Ground truth separation — THE critical test
# ---------------------------------------------------------------------------


class TestGroundTruthSeparation:
    def test_labels_extracted_from_ticket(self):
        """The pipeline ticket must NOT contain labels."""
        raw = make_raw_ticket()
        result = parse_ticket(raw)

        # Labels exist in the IngestedTicket wrapper
        assert result.labels is not None
        assert result.labels.intent == "deployment_failure"

        # But NOT in the StandardTicket itself
        ticket_dict = result.ticket.model_dump()
        assert "labels" not in ticket_dict
        assert "intent" not in ticket_dict

    def test_history_extracted_from_ticket(self):
        raw = make_raw_ticket()
        result = parse_ticket(raw)
        assert result.history is not None
        assert result.history.csat_rating == 5

    def test_ticket_without_labels(self):
        """Hidden evaluation set won't have labels — must still parse."""
        raw = make_raw_ticket()
        del raw["labels"]
        del raw["history"]
        result = parse_ticket(raw)
        assert result.labels is None
        assert result.history is None
        assert result.ticket.ticket_id == "TEST-001"


# ---------------------------------------------------------------------------
# File ingestion
# ---------------------------------------------------------------------------


class TestIngestTickets:
    def test_load_development_tickets(self):
        """Smoke test: load the actual development dataset."""
        results = ingest_tickets("data/development_tickets.json")
        assert len(results) == 500
        channels = {r.ticket.channel for r in results}
        assert channels == {"email", "chat", "docs_comment", "forum"}

    def test_all_tickets_have_labels(self):
        """Every dev ticket should have ground-truth labels."""
        results = ingest_tickets("data/development_tickets.json")
        for r in results:
            assert r.labels is not None, f"{r.ticket.ticket_id} missing labels"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            ingest_tickets("data/nonexistent.json")

    def test_invalid_json(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json {{{")
        with pytest.raises(json.JSONDecodeError):
            ingest_tickets(bad_file)

    def test_non_array_json(self, tmp_path):
        bad_file = tmp_path / "obj.json"
        bad_file.write_text('{"ticket_id": "T-1"}')
        with pytest.raises(ValueError, match="JSON array"):
            ingest_tickets(bad_file)

"""
Ingest module for CloudServe Support System.

This is the FIRST stage of the pipeline: Ingest → Classify → Retrieve → Route → Generate → Validate

Purpose:
    Accept raw support tickets from any channel (email, chat, docs_comment, forum),
    validate required fields, normalize into a standard structure, and separate
    ground-truth labels from the data the system is allowed to see.

Design decisions:
    1. We use Pydantic models (not plain dicts) for runtime validation and clear contracts.
    2. Ground-truth labels are stripped during ingestion — the pipeline never sees them.
       They're stored separately for evaluation only.
    3. The module accepts a file path (not hardcoded) to satisfy acceptance criterion A9:
       the system must run unattended with --input and --output arguments.

Interview context:
    "Why Pydantic instead of dataclasses?"
    → Pydantic validates at construction time. If a ticket arrives with urgency=999 or
      a missing customer_id, Pydantic raises immediately rather than letting bad data
      propagate through 5 more pipeline stages where the error message becomes useless.
      Fail fast, fail loud, fail at the boundary.
"""

import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models — these define the CONTRACT for data flowing through the system
# ---------------------------------------------------------------------------


class GroundTruthLabels(BaseModel):
    """
    Labels attached to tickets for EVALUATION ONLY.

    These represent what the correct answer should be. Our system must
    predict these — it must never read them during processing.

    Stored separately so we can compare system predictions against truth.
    """

    intent: str = ""
    urgency: str = ""
    expected_route: str = ""
    answerable_from_docs: bool = False
    expected_doc_ids: list[str] = Field(default_factory=list)
    must_not_auto_respond: bool = False


class TicketHistory(BaseModel):
    """Historical outcome data — also evaluation-only."""

    first_contact_resolution: bool | None = None
    resolution_time_minutes: int | None = None
    csat_rating: int | None = None
    escalated: bool | None = None
    repeat_contact: bool | None = None


class StandardTicket(BaseModel):
    """
    The normalized ticket that flows through the pipeline.

    Every component downstream (classify, retrieve, route, generate)
    receives this exact shape. No component ever needs to check
    which channel the ticket came from to parse it — that's the
    whole point of normalization.

    Fields:
        ticket_id:       Unique identifier (e.g., "DEV-0001")
        channel:         Origin channel — email, chat, docs_comment, forum
        subject:         Subject line (empty string for chat tickets)
        body:            The actual support request text
        received_at:     ISO 8601 timestamp
        customer_id:     Customer identifier
        customer_name:   Customer's display name
        customer_tier:   standard, business, or enterprise
        customer_region: Geographic region
        language_fluency: fluent or non_fluent — affects response generation
    """

    ticket_id: str
    channel: str
    subject: str = ""
    body: str
    received_at: str
    customer_id: str
    customer_name: str = ""
    customer_tier: str = "standard"
    customer_region: str = ""
    language_fluency: str = "fluent"

    @field_validator("channel")
    @classmethod
    def validate_channel(cls, v: str) -> str:
        allowed = {"email", "chat", "docs_comment", "forum"}
        if v not in allowed:
            raise ValueError(f"Unknown channel '{v}'. Expected one of: {allowed}")
        return v

    @field_validator("body")
    @classmethod
    def validate_body_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Ticket body cannot be empty.")
        return v

    def combined_text(self) -> str:
        """
        Merge subject + body into a single string for downstream NLP.

        Why? The classifier and retriever need one text input. Some tickets
        have important context in the subject ("SSO login fails for SAML users")
        that isn't repeated in the body. Concatenating ensures nothing is lost.
        """
        parts = []
        if self.subject.strip():
            parts.append(self.subject.strip())
        parts.append(self.body.strip())
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Ingestion result — pairs each ticket with its ground truth for evaluation
# ---------------------------------------------------------------------------


class IngestedTicket(BaseModel):
    """
    Container pairing a StandardTicket with its ground-truth labels.

    The pipeline processes only the `ticket` field.
    The `labels` and `history` fields are used ONLY by the evaluation harness
    to score the system's predictions.
    """

    ticket: StandardTicket
    labels: GroundTruthLabels | None = None
    history: TicketHistory | None = None


# ---------------------------------------------------------------------------
# Core ingestion functions
# ---------------------------------------------------------------------------


def parse_ticket(raw: dict) -> IngestedTicket:
    """
    Parse a single raw ticket dict into an IngestedTicket.

    This function does three things:
    1. Extracts and removes ground-truth labels (the system must not see them)
    2. Extracts and removes history (evaluation-only)
    3. Validates and constructs a StandardTicket from remaining fields

    Args:
        raw: A dictionary from the JSON dataset

    Returns:
        IngestedTicket with .ticket (for pipeline) and .labels (for eval)

    Raises:
        ValueError: If required fields are missing or invalid
    """
    # Step 1: Extract ground truth BEFORE building the ticket
    # This separation is critical — it's what makes the evaluation honest
    labels_raw = raw.pop("labels", None)
    labels = GroundTruthLabels(**labels_raw) if labels_raw else None

    history_raw = raw.pop("history", None)
    history = TicketHistory(**history_raw) if history_raw else None

    # Step 2: Build the StandardTicket from remaining fields
    ticket = StandardTicket(**raw)

    return IngestedTicket(ticket=ticket, labels=labels, history=history)


def ingest_tickets(input_path: str | Path) -> list[IngestedTicket]:
    """
    Load and parse all tickets from a JSON file.

    This is the main entry point for the ingest stage. It:
    1. Reads the JSON file from the given path (never hardcoded!)
    2. Parses each ticket, separating data from ground truth
    3. Logs statistics for monitoring
    4. Returns a list of IngestedTickets ready for the pipeline

    Args:
        input_path: Path to the JSON file containing tickets.
                    Accepts string or Path object for flexibility.

    Returns:
        List of IngestedTicket objects

    Raises:
        FileNotFoundError: If input_path doesn't exist
        json.JSONDecodeError: If file isn't valid JSON
        ValueError: If any ticket fails validation
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info("Loading tickets from %s", input_path)

    with open(input_path, "r", encoding="utf-8") as f:
        raw_tickets = json.load(f)

    if not isinstance(raw_tickets, list):
        raise ValueError(  # noqa: TRY004
            f"Expected a JSON array of tickets, got {type(raw_tickets).__name__}"
        )

    results: list[IngestedTicket] = []
    errors: list[str] = []

    for i, raw in enumerate(raw_tickets):
        try:
            # Make a copy so we don't mutate the original
            ticket_data = dict(raw)
            ingested = parse_ticket(ticket_data)
            results.append(ingested)
        except Exception as e:  # noqa: BLE001
            error_msg = f"Ticket {i} (id={raw.get('ticket_id', '?')}): {e}"
            errors.append(error_msg)
            logger.warning("Skipping invalid ticket: %s", error_msg)

    # Log summary statistics
    from collections import Counter

    channels = Counter(r.ticket.channel for r in results)

    logger.info(
        "Ingested %d/%d tickets (%d errors). Channels: %s",
        len(results),
        len(raw_tickets),
        len(errors),
        dict(channels),
    )

    if errors:
        logger.warning("Ingestion errors:\n  %s", "\n  ".join(errors))

    return results

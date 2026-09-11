"""
FastAPI Application for CloudServe Support System.

Build ID: B-12-API
Requirement: Expose REST endpoints and real-time Prometheus metrics.

Endpoints:
    - GET  /health          : Service health check
    - GET  /metrics         : Prometheus telemetry & metrics scraping endpoint
    - POST /api/v1/tickets  : Process a single incoming support ticket
    - POST /api/v1/batch    : Process a batch of tickets
    - GET  /api/v1/stats    : Summary stats from the decision database
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST

from src.ingest import StandardTicket
from src.metrics import get_latest_metrics, record_error
from src.pipeline import run_ticket, get_pipeline
from src.retrieve import build_vector_store
from src.logging_store import init_database, log_decision, build_decision_record, get_summary_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------

class TicketRequest(BaseModel):
    ticket_id: Optional[str] = Field(default=None, description="Unique ticket ID")
    channel: str = Field(default="portal", description="Source channel (email, chat, portal, api)")
    customer_tier: str = Field(default="standard", description="Customer tier (free, standard, premium, enterprise)")
    subject: str = Field(default="", description="Ticket subject line")
    body: str = Field(..., description="Ticket body content")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional ticket metadata")


class BatchTicketRequest(BaseModel):
    tickets: List[TicketRequest] = Field(..., description="List of tickets to process")


class TicketResponse(BaseModel):
    ticket_id: str
    status: str
    final_action: str
    intent: Optional[str] = None
    urgency: Optional[str] = None
    confidence: Optional[float] = None
    answerable_from_docs: Optional[bool] = None
    routing_action: Optional[str] = None
    escalation_target: Optional[str] = None
    routing_reason: Optional[str] = None
    generated_response: Optional[str] = None
    citations: List[str] = Field(default_factory=list)
    guardrail_passed: Optional[bool] = None
    duration_seconds: float
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Lifespan Management
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize vector store, pipeline, and decision log on startup."""
    logger.info("Initializing CloudServe Support System API...")
    try:
        init_database()
        build_vector_store()
        get_pipeline()
        logger.info("CloudServe Support System API initialization complete.")
    except Exception as e:
        logger.warning("Startup initialization warning: %s", e)
    yield


# Initialize FastAPI app
app = FastAPI(
    title="CloudServe Support System API",
    description="Real-time intelligent customer support pipeline with Prometheus telemetry",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Core Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint for container orchestrators and load balancers."""
    return {
        "status": "healthy",
        "service": "cloudserve-support-system",
        "timestamp": time.time(),
    }


@app.get("/metrics", tags=["Observability"])
def prometheus_metrics():
    """
    Expose Prometheus metrics for scraping.
    Standard endpoint scraped by Prometheus server.
    """
    return Response(
        content=get_latest_metrics(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/api/v1/tickets", response_model=TicketResponse, tags=["Pipeline"])
def process_single_ticket(request: TicketRequest):
    """
    Process a single support ticket through the LangGraph pipeline in real time.
    Emits live Prometheus metrics across every stage.
    """
    start_time = time.time()
    ticket_id = request.ticket_id or f"API-{int(time.time() * 1000)}"

    ticket = StandardTicket(
        ticket_id=ticket_id,
        channel=request.channel,
        customer_tier=request.customer_tier,
        subject=request.subject,
        body=request.body,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        metadata=request.metadata or {},
    )

    try:
        pipeline_state = run_ticket(ticket)
        duration = time.time() - start_time

        classification = pipeline_state.get("classification")
        retrieval = pipeline_state.get("retrieval")
        route = pipeline_state.get("route_decision")
        generation = pipeline_state.get("generation")
        guardrails = pipeline_state.get("guardrails")
        final_action = pipeline_state.get("final_action", "unknown")

        # Log decision to SQLite
        try:
            record = build_decision_record(
                ticket_id=ticket.ticket_id,
                classification=classification,
                route=route,
                retrieval_doc_ids=retrieval.unique_doc_ids if retrieval else [],
                num_chunks=retrieval.num_results if retrieval else 0,
                generation=generation,
                guardrails=guardrails,
                final_action=final_action,
            )
            log_decision(record)
        except Exception as log_err:
            logger.warning("Failed to log decision: %s", log_err)

        return TicketResponse(
            ticket_id=ticket_id,
            status="success",
            final_action=final_action,
            intent=classification.intent if classification else None,
            urgency=classification.urgency if classification else None,
            confidence=classification.intent_confidence if classification else None,
            answerable_from_docs=classification.answerable_from_docs if classification else None,
            routing_action=route.action.value if route else None,
            escalation_target=route.escalation_target if route else None,
            routing_reason=route.reason if route else None,
            generated_response=generation.response_text if generation else None,
            citations=generation.cited_doc_ids if generation else [],
            guardrail_passed=guardrails.passed if guardrails else None,
            duration_seconds=round(duration, 3),
        )

    except Exception as e:
        duration = time.time() - start_time
        record_error(type(e).__name__, "api")
        logger.error("Error processing ticket %s: %s", ticket_id, e)
        return TicketResponse(
            ticket_id=ticket_id,
            status="error",
            final_action="error",
            duration_seconds=round(duration, 3),
            error=str(e),
        )


@app.post("/api/v1/batch", response_model=List[TicketResponse], tags=["Pipeline"])
def process_batch_tickets(request: BatchTicketRequest):
    """Process a batch of support tickets sequentially."""
    results = []
    for item in request.tickets:
        res = process_single_ticket(item)
        results.append(res)
    return results


@app.get("/api/v1/stats", tags=["Analytics"])
def get_stats():
    """Retrieve aggregate statistics from the decision store."""
    try:
        return get_summary_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

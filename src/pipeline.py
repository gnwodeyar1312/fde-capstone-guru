"""
LangGraph Pipeline Orchestration for CloudServe Support System.

Purpose:
    Orchestrate the 6-stage pipeline using LangGraph's StateGraph:
    Ingest → Classify → Retrieve → Route → Generate → Validate

Design decisions:
    1. We use LangGraph StateGraph for pipeline orchestration.
       Why? LangGraph provides a graph-based workflow engine that makes
       the pipeline stages explicit, composable, and traceable. Each node
       in the graph corresponds to a pipeline stage, and edges define the
       flow including conditional branching (e.g., skip generation for
       escalated tickets).
    2. The state is a TypedDict that flows through all stages.
       Why? Typed state ensures each stage gets the right inputs and
       produces the right outputs. It's the contract between stages.
    3. Conditional routing after the Route stage.
       Why? Escalated tickets skip generation and validation entirely.
       LangGraph's conditional edges make this explicit in the graph
       structure rather than buried in if-statements.
    4. The pipeline is compiled once and reused for all tickets.
       Why? Compiling the graph has overhead. Reusing the compiled graph
       for each ticket amortizes that cost.

Interview context:
    "Why LangGraph instead of a simple function chain?"
    → LangGraph makes the pipeline structure visible and modifiable.
      Adding a new stage (e.g., a reranker between retrieve and route)
      is adding a node and two edges, not refactoring a 200-line function.
      It also supports conditional branching natively — escalated tickets
      follow a different path than auto-respond tickets.
    "Could you add parallel stages?"
    → Yes. LangGraph supports parallel execution. For example, retrieval
      and classification could run in parallel since they're independent.
      We keep them sequential for simplicity in the MVP, but the graph
      structure makes parallelism a one-line change.
"""

import logging
import time
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

from src.classify import ClassificationResult, classify_ticket
from src.config import get_llm
from src.generate import GeneratedResponse, generate_response
from src.guardrails import GuardrailResult, validate_response
from src.ingest import StandardTicket
from src.metrics import (
    track_stage_duration,
    track_in_flight,
    record_classification_metrics,
    record_retrieval_metrics,
    record_routing_metrics,
    record_generation_metrics,
    record_guardrail_metrics,
    record_ticket_completion,
    record_error,
)
from src.retrieve import RetrievalResult, retrieve_for_ticket
from src.route import RouteDecision, route_ticket

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline state — the typed container that flows through all stages
# ---------------------------------------------------------------------------


class PipelineState(TypedDict, total=False):
    """
    The state that flows through the LangGraph pipeline.

    Each stage reads what it needs and writes its output.
    Using TypedDict ensures type safety at each stage boundary.

    Note: State keys must not collide with node names in LangGraph.
    The routing stage output is named 'route_decision' (not 'route')
    to avoid conflict with the 'route' node name.
    """
    # Input
    ticket: Any  # StandardTicket
    llm: Any  # ChatOpenAI instance

    # Stage outputs
    classification: Any  # ClassificationResult
    retrieval: Any  # RetrievalResult
    route_decision: Any  # RouteDecision
    generation: Any  # GeneratedResponse (None if escalated)
    guardrails: Any  # GuardrailResult (None if escalated)

    # Final outcome
    final_action: str  # "sent", "escalated", "blocked_by_guardrails"
    error: str  # Error message if any stage fails


# ---------------------------------------------------------------------------
# Stage nodes — each one is a function that takes state and returns updates
# ---------------------------------------------------------------------------


def classify_node(state: PipelineState) -> dict:
    """Stage 2: Classify the ticket using LangChain ChatOpenAI."""
    ticket = state["ticket"]
    llm = state.get("llm")
    with track_stage_duration("classify"):
        classification = classify_ticket(ticket, llm=llm)
    record_classification_metrics(classification)
    return {"classification": classification}


def retrieve_node(state: PipelineState) -> dict:
    """Stage 3: Retrieve relevant documentation chunks via LangChain Chroma."""
    ticket = state["ticket"]
    with track_stage_duration("retrieve"):
        retrieval = retrieve_for_ticket(ticket)
    record_retrieval_metrics(retrieval)
    return {"retrieval": retrieval}


def route_node(state: PipelineState) -> dict:
    """Stage 4: Route the ticket (deterministic rules, no LLM)."""
    classification = state["classification"]
    with track_stage_duration("route"):
        decision = route_ticket(classification)
    record_routing_metrics(decision)
    return {"route_decision": decision}


def generate_node(state: PipelineState) -> dict:
    """Stage 5: Generate a response using LangChain ChatOpenAI (auto_respond only)."""
    ticket = state["ticket"]
    classification = state["classification"]
    retrieval = state["retrieval"]
    llm = state.get("llm")
    with track_stage_duration("generate"):
        generation = generate_response(ticket, classification, retrieval, llm=llm)
    record_generation_metrics(generation)
    return {"generation": generation}


def validate_node(state: PipelineState) -> dict:
    """Stage 6: Validate the generated response through guardrails."""
    generation = state["generation"]
    retrieval = state["retrieval"]
    with track_stage_duration("validate"):
        guardrail_result = validate_response(generation, retrieval)

    if guardrail_result.passed:
        final_action = "sent"
    else:
        final_action = "blocked_by_guardrails"

    record_guardrail_metrics(guardrail_result)
    return {"guardrails": guardrail_result, "final_action": final_action}


def escalate_node(state: PipelineState) -> dict:
    """Terminal node for escalated tickets — no generation or validation needed."""
    return {"final_action": "escalated"}


# ---------------------------------------------------------------------------
# Conditional edge: should we generate or escalate?
# ---------------------------------------------------------------------------


def should_generate(state: PipelineState) -> str:
    """
    Conditional edge after routing.

    If the route decision is auto_respond, we proceed to generation.
    If escalated, we skip directly to the escalate terminal node.
    """
    decision = state["route_decision"]
    if decision.is_auto_respond:
        return "generate"
    return "escalate"


# ---------------------------------------------------------------------------
# Build and compile the pipeline graph
# ---------------------------------------------------------------------------


def build_pipeline() -> StateGraph:
    """
    Build the LangGraph StateGraph for the support pipeline.

    Graph structure:
        classify → retrieve → route → [conditional]
                                          ├── generate → validate → END
                                          └── escalate → END

    Returns:
        A compiled StateGraph ready to invoke.
    """
    graph = StateGraph(PipelineState)

    # Add nodes (stages)
    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("route", route_node)
    graph.add_node("generate", generate_node)
    graph.add_node("validate", validate_node)
    graph.add_node("escalate", escalate_node)

    # Set entry point
    graph.set_entry_point("classify")

    # Add edges (stage transitions)
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "route")

    # Conditional edge after routing
    graph.add_conditional_edges(
        "route",
        should_generate,
        {
            "generate": "generate",
            "escalate": "escalate",
        },
    )

    graph.add_edge("generate", "validate")
    graph.add_edge("validate", END)
    graph.add_edge("escalate", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Convenience: run a single ticket through the pipeline
# ---------------------------------------------------------------------------

# Cache the compiled pipeline (compiled once, reused for all tickets)
_compiled_pipeline = None


def get_pipeline():
    """Get or create the compiled pipeline (singleton)."""
    global _compiled_pipeline
    if _compiled_pipeline is None:
        _compiled_pipeline = build_pipeline()
    return _compiled_pipeline


def run_ticket(ticket: StandardTicket, llm=None) -> PipelineState:
    """
    Run a single ticket through the full LangGraph pipeline.

    Args:
        ticket: A StandardTicket from the ingest stage
        llm: Optional pre-configured ChatOpenAI instance

    Returns:
        The final PipelineState with all stage outputs
    """
    pipeline = get_pipeline()

    initial_state = {
        "ticket": ticket,
        "llm": llm,
    }

    start_time = time.time()
    channel = getattr(ticket, "channel", "unknown") or "unknown"
    tier = getattr(ticket, "customer_tier", "unknown") or "unknown"

    with track_in_flight(channel):
        try:
            result = pipeline.invoke(initial_state)
            duration = time.time() - start_time
            final_action = result.get("final_action", "unknown")
            record_ticket_completion(channel, tier, "success", final_action, duration)
            return result
        except Exception as e:
            duration = time.time() - start_time
            record_error(type(e).__name__, "pipeline")
            record_ticket_completion(channel, tier, "error", "error", duration)
            raise

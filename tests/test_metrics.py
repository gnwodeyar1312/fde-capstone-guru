"""
Tests for Prometheus Metrics Instrumentation and API Endpoints.

Verifies:
1. Metric registrations and updates across all pipeline stages.
2. In-flight gauge increments and decrements.
3. Stage timing context managers.
4. Correct label propagation.
5. FastAPI /health and /metrics endpoint outputs.
"""

import time
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.classify import ClassificationResult
from src.guardrails import GuardrailCheck, GuardrailResult
from src.generate import GeneratedResponse
from src.metrics import (
    get_latest_metrics,
    record_classification_metrics,
    record_error,
    record_generation_metrics,
    record_guardrail_metrics,
    record_rate_limit,
    record_retrieval_metrics,
    record_routing_metrics,
    record_ticket_completion,
    track_in_flight,
    track_stage_duration,
    TICKETS_IN_FLIGHT,
)
from src.retrieve import RetrievedChunk, RetrievalResult
from src.route import RouteAction, RouteDecision


@pytest.fixture
def client():
    return TestClient(app)


class TestMetricsInstrumentation:
    def test_in_flight_tracking(self):
        gauge = TICKETS_IN_FLIGHT.labels(channel="test_channel")
        initial_val = gauge._value.get()
        with track_in_flight("test_channel"):
            assert gauge._value.get() == initial_val + 1
        assert gauge._value.get() == initial_val

    def test_stage_duration_tracking(self):
        with track_stage_duration("test_stage"):
            time.sleep(0.01)

        metrics_text = get_latest_metrics().decode("utf-8")
        assert "cloudserve_stage_duration_seconds" in metrics_text
        assert 'stage="test_stage"' in metrics_text

    def test_record_classification_metrics(self):
        res = ClassificationResult(
            intent="deployment_failure",
            intent_confidence=0.95,
            urgency="high",
            urgency_confidence=0.88,
            answerable_from_docs=True,
            answerable_confidence=0.91,
            must_not_auto_respond=False,
            reasoning="Kubernetes crashloop",
        )
        record_classification_metrics(res)
        metrics_text = get_latest_metrics().decode("utf-8")
        assert 'intent="deployment_failure"' in metrics_text
        assert "cloudserve_classification_confidence" in metrics_text

    def test_record_retrieval_metrics(self):
        chunks = [
            RetrievedChunk(
                doc_id="KB-001",
                title="Deployment Guide",
                category="Deployment",
                section_name="Troubleshooting",
                content="Check logs...",
                similarity_score=0.15,
            )
        ]
        ret = RetrievalResult(chunks=chunks, query="deploy error")
        record_retrieval_metrics(ret)
        metrics_text = get_latest_metrics().decode("utf-8")
        assert "cloudserve_retrieval_chunks_count" in metrics_text
        assert "cloudserve_retrieval_relevance_score" in metrics_text

    def test_record_routing_metrics(self):
        decision = RouteDecision(
            action=RouteAction.AUTO_RESPOND,
            reason="High confidence answerable",
            rule_triggered="rule_1_standard_auto",
            escalation_target=None,
        )
        gt = {"must_not_auto_respond": False}
        record_routing_metrics(decision, ground_truth=gt)

        metrics_text = get_latest_metrics().decode("utf-8")
        assert 'action="auto_respond"' in metrics_text
        assert 'rule_triggered="rule_1_standard_auto"' in metrics_text

    def test_record_policy_compliance(self):
        decision = RouteDecision(
            action=RouteAction.ESCALATE,
            reason="Security policy",
            rule_triggered="rule_must_not_auto",
            escalation_target="security_team",
        )
        gt = {"must_not_auto_respond": True}
        record_routing_metrics(decision, ground_truth=gt)

        metrics_text = get_latest_metrics().decode("utf-8")
        assert 'policy="must_not_auto_respond"' in metrics_text
        assert 'compliant="true"' in metrics_text

    def test_record_generation_metrics(self):
        gen = GeneratedResponse(
            response_text="Here are the deployment fix steps...",
            cited_doc_ids=["KB-001", "KB-002"],
            could_answer=True,
            model_used="llama3-8b",
        )
        record_generation_metrics(gen)
        metrics_text = get_latest_metrics().decode("utf-8")
        assert 'could_answer="true"' in metrics_text
        assert "cloudserve_generation_citations_count" in metrics_text

    def test_record_guardrail_metrics(self):
        checks = [
            GuardrailCheck(name="no_pii_leakage", passed=True, reason="Clean", severity="block"),
            GuardrailCheck(name="citation_validation", passed=True, reason="Valid citations", severity="block"),
        ]
        gr = GuardrailResult(passed=True, checks=checks, failed_checks=[], warnings=[])
        record_guardrail_metrics(gr)

        metrics_text = get_latest_metrics().decode("utf-8")
        assert 'passed="true"' in metrics_text
        assert 'check_name="no_pii_leakage"' in metrics_text

    def test_record_ticket_completion_and_error(self):
        record_ticket_completion(
            channel="email",
            tier="enterprise",
            status="success",
            final_action="sent",
            duration=1.23,
        )
        record_error("RateLimitError", "classify")
        record_rate_limit("groq", 5.0)

        metrics_text = get_latest_metrics().decode("utf-8")
        assert 'channel="email"' in metrics_text
        assert 'tier="enterprise"' in metrics_text
        assert 'error_type="RateLimitError"' in metrics_text
        assert 'provider="groq"' in metrics_text


class TestApiEndpoints:
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "cloudserve-support-system"

    def test_metrics_endpoint(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "cloudserve_tickets_total" in response.text
        assert response.headers["content-type"].startswith("text/plain")

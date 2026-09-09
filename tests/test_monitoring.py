"""
Tests for the monitoring module.

Three test classes:
1. TestComputeMetrics — verifies metric computation from harness output
2. TestPerIntentBreakdown — verifies per-intent accuracy calculation
3. TestErrorBreakdown — verifies error categorization
"""

import pytest

from src.monitoring import compute_metrics

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_results():
    """Minimal harness output for testing."""
    return {
        "metadata": {
            "input_file": "test.json",
            "run_timestamp": "2026-09-08T00:00:00Z",
            "total_tickets": 3,
            "successful": 2,
            "errors": 1,
            "elapsed_seconds": 30.0,
            "tickets_per_minute": 4.0,
        },
        "results": [
            {
                "ticket_id": "DEV-0001",
                "status": "success",
                "ground_truth": {
                    "intent": "deployment_failure",
                    "urgency": "high",
                    "expected_route": "auto_respond",
                    "answerable_from_docs": True,
                },
                "predictions": {
                    "intent": "deployment_failure",
                    "intent_confidence": 0.92,
                    "answerable_from_docs": True,
                },
                "route": {"action": "auto_respond", "reason": "all checks passed"},
                "guardrails": {
                    "passed": True,
                    "checks": [
                        {"name": "citation_validation", "passed": True},
                        {"name": "no_pii_leakage", "passed": True},
                    ],
                },
                "generation": {"response": "Here is help..."},
                "final_action": "sent",
            },
            {
                "ticket_id": "DEV-0002",
                "status": "success",
                "ground_truth": {
                    "intent": "feature_request",
                    "urgency": "low",
                    "expected_route": "escalate",
                    "answerable_from_docs": False,
                },
                "predictions": {
                    "intent": "feature_request",
                    "intent_confidence": 0.88,
                    "answerable_from_docs": False,
                },
                "route": {"action": "escalate", "reason": "must_not_auto_respond"},
                "guardrails": None,
                "final_action": "escalated",
            },
            {
                "ticket_id": "DEV-0003",
                "status": "error",
                "error": "Rate limit exceeded (429)",
                "ground_truth": {"intent": "billing_query"},
                "predictions": {},
                "route": {},
            },
        ],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_run_info(self, sample_results):
        metrics = compute_metrics(sample_results)
        ri = metrics["run_info"]
        assert ri["total_tickets"] == 3
        assert ri["successful"] == 2
        assert ri["failed"] == 1
        assert ri["error_rate"] == 33.3

    def test_accuracy_from_successful(self, sample_results):
        metrics = compute_metrics(sample_results)
        acc = metrics["accuracy"]
        assert acc["intent"] == 100.0
        assert acc["intent_correct"] == 2
        assert acc["answerable"] == 100.0
        assert acc["route"] == 100.0

    def test_routing_metrics(self, sample_results):
        metrics = compute_metrics(sample_results)
        route = metrics["routing"]
        assert route["auto_respond"] == 1
        assert route["escalate"] == 1
        assert route["must_not_auto_correct"] == 1
        assert route["must_not_auto_total"] == 1
        assert route["must_not_auto_compliance"] == 100.0

    def test_guardrail_metrics(self, sample_results):
        metrics = compute_metrics(sample_results)
        gr = metrics["guardrails"]
        assert gr["total_checked"] == 1
        assert gr["total_passed"] == 1
        assert gr["pass_rate"] == 100.0


class TestPerIntentBreakdown:
    def test_per_intent_counts(self, sample_results):
        metrics = compute_metrics(sample_results)
        pi = metrics["per_intent"]
        assert "deployment_failure" in pi
        assert "feature_request" in pi
        assert pi["deployment_failure"]["count"] == 1
        assert pi["feature_request"]["intent_accuracy"] == 100.0

    def test_wrong_intent_lowers_accuracy(self, sample_results):
        # Flip the prediction to be wrong
        sample_results["results"][0]["predictions"]["intent"] = "billing_query"
        metrics = compute_metrics(sample_results)
        pi = metrics["per_intent"]
        assert pi["deployment_failure"]["intent_accuracy"] == 0.0


class TestErrorBreakdown:
    def test_rate_limit_categorized(self, sample_results):
        metrics = compute_metrics(sample_results)
        errors = metrics["errors"]
        assert errors["total_errors"] == 1
        assert errors["by_type"]["rate_limit"] == 1

    def test_no_errors(self):
        data = {
            "metadata": {
                "total_tickets": 1,
                "successful": 1,
                "errors": 0,
                "elapsed_seconds": 10,
                "tickets_per_minute": 6,
            },
            "results": [
                {
                    "ticket_id": "DEV-0001",
                    "status": "success",
                    "ground_truth": {
                        "intent": "billing_query",
                        "expected_route": "auto_respond",
                        "answerable_from_docs": True,
                    },
                    "predictions": {
                        "intent": "billing_query",
                        "answerable_from_docs": True,
                    },
                    "route": {"action": "auto_respond"},
                    "guardrails": {"passed": True, "checks": []},
                }
            ],
        }
        metrics = compute_metrics(data)
        assert metrics["errors"]["total_errors"] == 0

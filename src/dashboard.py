"""
Monitoring Dashboard for CloudServe Support System.

Build ID: B-12
Requirement: NFR-03 — Monitoring and observability.

Purpose:
    Interactive Streamlit dashboard that visualizes pipeline performance
    metrics. Reads evaluation harness output files and displays accuracy
    trends, per-intent breakdowns, routing distributions, guardrail
    compliance, and error analysis.

Usage:
    streamlit run src/dashboard.py

Design decisions:
    1. Streamlit, not Flask or custom HTML.
       Why? Streamlit gives us interactive charts, file upload, and
       responsive layout with zero frontend code. Perfect for a data-heavy
       monitoring dashboard in a capstone project.
    2. Load from results/ directory by default.
       Why? The evaluation harness writes there. The dashboard auto-discovers
       all JSON result files so you can compare runs.
    3. Charts use Streamlit's built-in plotly integration.
       Why? Interactive hover, zoom, and export with no extra dependencies.

Interview context:
    "How do you monitor pipeline health?"
    → We have a Streamlit dashboard that reads evaluation results and shows
      accuracy by intent, routing distribution, guardrail compliance, and
      error trends. You can compare multiple runs to see how prompt changes
      affect performance.
"""

import streamlit as st
import json
import os
import sys
from pathlib import Path

# Add project root to path so we can import monitoring
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.monitoring import compute_metrics


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CloudServe Monitoring",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_results(file_path: str) -> dict:
    """Load and compute metrics from a results file."""
    with open(file_path) as f:
        data = json.load(f)
    return compute_metrics(data)


def discover_result_files() -> list[str]:
    """Find all JSON result files in results/ directory."""
    results_dir = Path(__file__).parent.parent / "results"
    if not results_dir.exists():
        return []
    files = sorted(results_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
    return [str(f) for f in files]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("📊 CloudServe Monitor")
st.sidebar.markdown("---")

# File selection
result_files = discover_result_files()
if not result_files:
    st.error("No result files found in results/ directory. Run the evaluation harness first.")
    st.stop()

selected_file = st.sidebar.selectbox(
    "Select result file",
    result_files,
    format_func=lambda x: Path(x).name,
)

# Load metrics
metrics = load_results(selected_file)
ri = metrics["run_info"]
acc = metrics["accuracy"]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Run:** {Path(selected_file).name}")
st.sidebar.markdown(f"**Tickets:** {ri['successful']}/{ri['total_tickets']}")
st.sidebar.markdown(f"**Time:** {ri['elapsed_seconds']:.0f}s")


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("CloudServe Support System — Monitoring Dashboard")
st.caption(f"Run: {ri['timestamp']}  |  Input: {ri['input_file']}")

# ---- KPI Row ----
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Intent Accuracy", f"{acc['intent']:.1f}%",
            f"{acc['intent_correct']}/{acc['intent_total']}")
col2.metric("Answerable Accuracy", f"{acc['answerable']:.1f}%",
            f"{acc['answerable_correct']}/{acc['answerable_total']}")
col3.metric("Route Accuracy", f"{acc['route']:.1f}%",
            f"{acc['route_correct']}/{acc['route_total']}")
col4.metric("Error Rate", f"{ri['error_rate']:.1f}%",
            f"{ri['failed']} failed")
col5.metric("Throughput", f"{ri['tickets_per_minute']:.1f}/min",
            f"{ri['elapsed_seconds']:.0f}s total")

st.markdown("---")

# ---- Per-Intent Breakdown ----
st.subheader("Per-Intent Performance")

pi = metrics.get("per_intent", {})
if pi:
    import pandas as pd

    intent_data = []
    for intent, stats in pi.items():
        intent_data.append({
            "Intent": intent,
            "Count": stats["count"],
            "Intent Acc %": stats["intent_accuracy"],
            "Answerable Acc %": stats["answerable_accuracy"],
            "Route Acc %": stats["route_accuracy"],
        })

    df_intent = pd.DataFrame(intent_data)

    # Bar chart — intent accuracy by intent
    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        st.bar_chart(
            df_intent.set_index("Intent")[["Intent Acc %", "Answerable Acc %", "Route Acc %"]],
            height=400,
        )

    with col_table:
        st.dataframe(
            df_intent.style.format({
                "Intent Acc %": "{:.1f}",
                "Answerable Acc %": "{:.1f}",
                "Route Acc %": "{:.1f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
else:
    st.info("No per-intent data available.")

st.markdown("---")

# ---- Routing & Guardrails side-by-side ----
col_route, col_guard = st.columns(2)

with col_route:
    st.subheader("Routing Distribution")
    route = metrics["routing"]

    route_data = {
        "Decision": ["Auto-Respond", "Escalate"],
        "Count": [route["auto_respond"], route["escalate"]],
    }
    import pandas as pd
    df_route = pd.DataFrame(route_data)
    st.bar_chart(df_route.set_index("Decision"), height=250)

    st.markdown(f"**Must-not-auto compliance:** "
                f"{route['must_not_auto_correct']}/{route['must_not_auto_total']} "
                f"({route['must_not_auto_compliance']:.1f}%)")

with col_guard:
    st.subheader("Guardrail Results")
    gr = metrics["guardrails"]

    st.metric("Overall Pass Rate", f"{gr['pass_rate']:.1f}%",
              f"{gr['total_passed']}/{gr['total_checked']}")

    per_check = gr.get("per_check", {})
    if per_check:
        check_data = []
        for name, stats in per_check.items():
            total = stats["passed"] + stats["failed"]
            check_data.append({
                "Check": name,
                "Passed": stats["passed"],
                "Failed": stats["failed"],
                "Rate": f"{stats['passed'] / max(total, 1) * 100:.0f}%",
            })
        st.dataframe(pd.DataFrame(check_data), use_container_width=True, hide_index=True)

st.markdown("---")

# ---- Error Analysis ----
errors = metrics.get("errors", {})
if errors.get("total_errors", 0) > 0:
    st.subheader("Error Analysis")
    st.warning(f"**{errors['total_errors']}** tickets failed during this run.")

    error_types = errors.get("by_type", {})
    if error_types:
        err_data = [{"Type": k, "Count": v} for k, v in error_types.items()]
        st.dataframe(pd.DataFrame(err_data), use_container_width=True, hide_index=True)

# ---- Raw Metrics (collapsible) ----
with st.expander("Raw Metrics JSON"):
    st.json(metrics)

# ---- Compare Runs ----
st.markdown("---")
st.subheader("Compare Runs")

if len(result_files) >= 2:
    compare_files = st.multiselect(
        "Select files to compare",
        result_files,
        default=result_files[:min(3, len(result_files))],
        format_func=lambda x: Path(x).name,
    )

    if compare_files:
        compare_data = []
        for f in compare_files:
            m = load_results(f)
            compare_data.append({
                "Run": Path(f).name,
                "Tickets": m["run_info"]["successful"],
                "Intent %": m["accuracy"]["intent"],
                "Answerable %": m["accuracy"]["answerable"],
                "Route %": m["accuracy"]["route"],
                "Error Rate %": m["run_info"]["error_rate"],
                "Speed (t/min)": m["run_info"]["tickets_per_minute"],
            })
        df_compare = pd.DataFrame(compare_data)
        st.dataframe(df_compare, use_container_width=True, hide_index=True)

        # Comparison chart
        st.bar_chart(
            df_compare.set_index("Run")[["Intent %", "Answerable %", "Route %"]],
            height=300,
        )
else:
    st.info("Run the evaluation harness multiple times to compare results across runs.")

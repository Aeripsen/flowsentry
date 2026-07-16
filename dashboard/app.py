"""
FlowSentry live dashboard.

The centerpiece is the reject-threshold slider: move it and the coverage-vs-
reliability tradeoff recomputes LIVE from the trained model on a real slice of the
BCCC-UDP-QUIC test flows. Below it: a live alert feed (the same logic as
flowsentry.stream), a per-attack-family bar chart, and the real measured per-flow
latency/throughput.

Run:  streamlit run dashboard/app.py
The sys.path shim below lets a bare `streamlit run dashboard/app.py` work.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Make the src/ package importable whether or not it is installed / on PYTHONPATH.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flowsentry.attack_map import ATTACK_MAP  # noqa: E402
from flowsentry.stream import classify_stream, load_bundle, load_stream  # noqa: E402

st.set_page_config(page_title="FlowSentry", layout="wide")


@st.cache_resource
def get_bundle() -> dict:
    return load_bundle()


@st.cache_data(show_spinner=False)
def get_sample(n: int):
    """(X_stage2, truth) for the first n sample flows."""
    return load_stream(n)


@st.cache_data(show_spinner=False)
def imputed_sample(n: int):
    """Impute once (with the trained imputer) so curve calls skip re-imputation."""
    bundle = get_bundle()
    X, y = get_sample(n)
    return bundle["imputer"].transform(X), y


@st.cache_data(show_spinner=False)
def full_curve(n: int) -> pd.DataFrame:
    bundle = get_bundle()
    X, y = imputed_sample(n)
    grid = [round(float(t), 3) for t in np.linspace(0.0, 0.99, 34)]
    return pd.DataFrame(bundle["model"].coverage_reliability_curve(X, y, grid))


def operating_point(n: int, threshold: float) -> dict:
    bundle = get_bundle()
    X, y = imputed_sample(n)
    return bundle["model"].coverage_reliability_curve(X, y, [threshold])[0]


@st.cache_data(show_spinner=False)
def run_stream(n: int, reject_threshold: float):
    bundle = get_bundle()
    X, truth = get_sample(n)
    t0 = time.perf_counter()
    alerts, latencies, summary = classify_stream(bundle, X, truth, reject_threshold)
    wall = time.perf_counter() - t0
    summary["throughput"] = summary["n_flows"] / wall if wall > 0 else float("nan")
    summary["wall_s"] = wall
    return alerts, latencies.tolist(), summary


st.title("FlowSentry - hierarchical UDP/QUIC intrusion detection")
st.caption(
    "Two-stage hierarchical classifier with a tunable reject option, on the real "
    "BCCC-UDP-QUIC-IDS-2025 dataset. The MITRE mapping is class-level (a triage hint)."
)

try:
    get_bundle()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    st.header("Controls")
    n = st.select_slider(
        "Test flows sampled", options=[1000, 2000, 3000, 5000, 8000], value=3000
    )
    threshold = st.slider(
        "Reject threshold (abstain below this confidence)",
        min_value=0.0,
        max_value=0.99,
        value=0.0,
        step=0.01,
    )
    st.caption(
        "Raise it and the model answers fewer flows (lower coverage) but is more "
        "reliable on the ones it does answer. That tradeoff is the whole point of "
        "the reject option."
    )

# --- Coverage vs reliability (recomputed live at the slider value) ---
st.subheader("Coverage vs reliability")
point = operating_point(n, threshold)
rel = point["reliability"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Reject threshold", f"{threshold:.2f}")
c2.metric("Coverage", f"{point['coverage'] * 100:.1f}%", help="fraction of flows answered")
c3.metric(
    "Reliability",
    f"{rel * 100:.1f}%" if rel is not None else "n/a",
    help="accuracy on the answered subset",
)
c4.metric("Escalated to stage 2", f"{point['escalation_rate'] * 100:.1f}%")

curve_df = full_curve(n)
chart_df = (
    curve_df.dropna(subset=["reliability"]).set_index("threshold")[["coverage", "reliability"]]
)
st.line_chart(chart_df, height=320)
st.caption(
    f"Operating point at threshold={threshold:.2f}: "
    f"coverage={point['coverage'] * 100:.1f}%, reliability={rel * 100:.1f}%"
    if rel is not None
    else f"Operating point at threshold={threshold:.2f}: no flows answered (coverage 0%)."
)

# --- Live inference: real measured latency/throughput + alert feed ---
alerts, _latencies, summary = run_stream(n, threshold)

st.subheader("Live inference")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Throughput", f"{summary['throughput']:,.0f} flows/s")
m2.metric("Mean latency", f"{summary['mean_ms']:.2f} ms")
m3.metric("p95 latency", f"{summary['p95_ms']:.2f} ms")
m4.metric("Alerts", f"{summary['counts']['attack']}")
st.caption(
    f"Measured on this machine: {summary['n_flows']} flows, single-thread batch replay, "
    f"preprocess+classify per flow, {summary['wall_s']:.2f}s wall."
)

col_bar, col_feed = st.columns([1, 2])

with col_bar:
    st.subheader("Alerts by attack family")
    fam = summary["family_counts"]
    if fam:
        fam_df = (
            pd.DataFrame({"family": list(fam.keys()), "alerts": list(fam.values())})
            .set_index("family")
            .sort_values("alerts", ascending=False)
        )
        st.bar_chart(fam_df, height=280)
    else:
        st.info("No attacks in this sample at the current threshold.")

with col_feed:
    st.subheader("Alert feed")
    if alerts:
        feed = pd.DataFrame(alerts)[
            [
                "timestamp",
                "flow_index",
                "predicted_class",
                "confidence",
                "escalated",
                "mitre_id",
                "mitre_technique",
                "true_label",
                "playbook",
            ]
        ]
        st.dataframe(feed, use_container_width=True, hide_index=True, height=280)
    else:
        st.info("No attack alerts at this threshold.")

with st.expander("MITRE ATT&CK mapping (class-level, not per-signature)"):
    st.write(
        "Every UDP DDoS family in this dataset is a volumetric flood, so they map to the "
        "same technique (T1498.001 Direct Network Flood). A real SOC refines this per "
        "signature; treat it as a triage hint."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "class": k,
                    "mitre_id": v["technique_id"],
                    "technique": v["technique_name"],
                    "playbook": v["playbook"],
                }
                for k, v in ATTACK_MAP.items()
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

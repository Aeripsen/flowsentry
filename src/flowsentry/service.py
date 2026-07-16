"""
FastAPI serving layer for FlowSentry.

GET  /health   -> liveness + whether a trained model is loaded (503 if missing)
POST /predict  -> classify one flow, with the reject/abstain knob as a request param
GET  /curve    -> the measured coverage-vs-reliability curve (from the last train run)

A flow is a dict of BCCC-UDP-QUIC feature name -> value. Missing UDP features are
imputed with the training median; missing QUIC features default to 0 (i.e. no QUIC
subflow observed), which is the honest default for a UDP-only flow. Scoring goes
through FlowScorer (scoring.py), the same path the replay and dashboard use.
"""
from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .scoring import ARTIFACT, ARTIFACT_DIR, FlowScorer

app = FastAPI(
    title="FlowSentry",
    version=__version__,
    description="Per-flow hierarchical UDP/QUIC intrusion detection with a tunable reject option.",
)

_scorer: FlowScorer | None = None


def _load() -> FlowScorer:
    global _scorer
    if _scorer is None:
        _scorer = FlowScorer.from_artifact(ARTIFACT)
    return _scorer


class Flow(BaseModel):
    features: dict = Field(
        ...,
        description=(
            "BCCC-UDP-QUIC feature name -> value. UDP flow-stat features "
            "(UDPFlowLyzer) and optional QUIC features (QUICFlowLyzer). Missing "
            "UDP features are median-imputed; missing QUIC features default to 0."
        ),
        examples=[
            {
                "pkt_count": 1240.0, "byte_count": 1785600.0, "pps": 41300.0,
                "bps": 4.76e8, "avg_pkt_size": 1440.0, "mean_iat": 2.4e-5,
                "iat_std": 8.0e-6, "directional_asymmetry": 1.0, "idle_ratio": 0.0,
                "has_quic_subflow": 0,
            }
        ],
    )
    reject_threshold: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Abstain ('unknown') when final confidence is below this.",
    )


@app.get("/health")
def health():
    try:
        _load()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return {"status": "ok", "model": "loaded", "version": __version__}


@app.post("/predict")
def predict(req: Flow):
    try:
        scorer = _load()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return scorer.score_one(req.features, reject_threshold=req.reject_threshold)


@app.get("/curve")
def curve():
    path = ARTIFACT_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(status_code=503, detail="no metrics; run training first")
    return json.loads(path.read_text())

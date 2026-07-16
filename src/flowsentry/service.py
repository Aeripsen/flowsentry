"""
FastAPI serving layer for FlowSentry.

GET  /health         -> liveness: the process is up (always 200; never touches the model)
GET  /ready          -> readiness: a trained model is loaded (503 until it is)
POST /predict        -> classify one flow, reject/abstain knob as a request field
POST /predict/batch  -> classify up to max_batch_rows flows in one vectorized call
GET  /curve          -> the measured coverage-vs-reliability curve from the last train run

A flow is a dict of BCCC-UDP-QUIC feature name -> numeric value. Missing UDP
features are imputed with the training median; missing QUIC features default to 0
(no QUIC subflow observed). Values must be finite numbers: strings and NaN/inf are
rejected with 422 instead of poisoning the feature row. Scoring goes through
FlowScorer (scoring.py), the same path the replay and dashboard use.

Logs are structured: one JSON line per scored request (label, confidence,
escalation, abstention, measured latency), so a log pipeline can consume them
without regex archaeology.
"""
from __future__ import annotations

import json
import logging
import math
import time
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .config import get_settings
from .scoring import ARTIFACT, ARTIFACT_DIR, FlowScorer

MAX_BATCH_ROWS = get_settings().serving.max_batch_rows

app = FastAPI(
    title="FlowSentry",
    version=__version__,
    description="Per-flow hierarchical UDP/QUIC intrusion detection with a tunable reject option.",
)

_scorer: FlowScorer | None = None


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload)


logger = logging.getLogger("flowsentry.service")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(_JsonFormatter())
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _load() -> FlowScorer:
    global _scorer
    if _scorer is None:
        _scorer = FlowScorer.from_artifact(ARTIFACT)
        logger.info("model_loaded", extra={"fields": {"artifact": str(ARTIFACT)}})
    return _scorer


def _require_scorer() -> FlowScorer:
    try:
        return _load()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


def _require_finite(features: dict[str, float]) -> None:
    """422 on NaN/Infinity. Done here with a plain-string detail rather than a
    pydantic validator: the default validation error echoes the offending input
    back, and a NaN/inf in that echo crashes the JSON encoder on the way out
    (a 500 on the error path; the test for this found exactly that)."""
    for name, value in features.items():
        if not math.isfinite(value):
            raise HTTPException(
                status_code=422, detail=f"feature {name!r} is not a finite number"
            )


class Flow(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description=(
            "BCCC-UDP-QUIC feature name -> numeric value. UDP flow-stat features "
            "(UDPFlowLyzer) and optional QUIC features (QUICFlowLyzer). Missing "
            "UDP features are median-imputed; missing QUIC features default to 0. "
            "Values must be finite; unknown names are ignored."
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


class FlowBatch(BaseModel):
    flows: list[dict[str, float]] = Field(
        ..., min_length=1, max_length=MAX_BATCH_ROWS,
        description=f"Feature dicts, one per flow, at most {MAX_BATCH_ROWS} per request.",
    )
    reject_threshold: float = Field(0.0, ge=0.0, le=1.0)


@app.get("/health")
def health():
    """Liveness only. Deliberately never touches the model: a hung or missing
    artifact must not make the process look dead to a restart loop."""
    return {"status": "ok", "version": __version__}


@app.get("/ready")
def ready():
    """Readiness: 200 iff a trained model is loaded and scoring can serve."""
    _require_scorer()
    return {"status": "ready", "model": "loaded", "version": __version__}


@app.post("/predict")
def predict(req: Flow):
    scorer = _require_scorer()
    _require_finite(req.features)
    t0 = time.perf_counter()
    verdict = scorer.score_one(req.features, reject_threshold=req.reject_threshold)
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    logger.info(
        "predict",
        extra={
            "fields": {
                **verdict,
                "latency_ms": latency_ms,
                "n_features": len(req.features),
                "reject_threshold": req.reject_threshold,
            }
        },
    )
    return verdict


@app.post("/predict/batch")
def predict_batch(req: FlowBatch):
    import numpy as np

    scorer = _require_scorer()
    for features in req.flows:
        _require_finite(features)
    t0 = time.perf_counter()
    rows = np.vstack([scorer.row_from_features(f) for f in req.flows])
    labels, conf, escalated, abstained = scorer.score_batch(
        rows, reject_threshold=req.reject_threshold
    )
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    results = [
        {
            "label": str(labels[i]),
            "confidence": round(float(conf[i]), 4),
            "escalated_to_stage2": bool(escalated[i]),
            "abstained": bool(abstained[i]),
        }
        for i in range(len(req.flows))
    ]
    logger.info(
        "predict_batch",
        extra={
            "fields": {
                "n_flows": len(results),
                "latency_ms": latency_ms,
                "reject_threshold": req.reject_threshold,
            }
        },
    )
    return {"n": len(results), "results": results}


@app.get("/curve")
def curve():
    path = ARTIFACT_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(status_code=503, detail="no metrics; run training first")
    return json.loads(path.read_text())

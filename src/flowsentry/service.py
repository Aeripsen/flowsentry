"""
FastAPI serving layer for FlowSentry.

GET  /health   -> liveness + whether a trained model is loaded
POST /predict  -> classify one flow, with the reject/abstain knob as a request param
GET  /curve    -> the measured coverage-vs-reliability curve (from the last train run)
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .data import CATEGORICAL, COLUMNS

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"
ARTIFACT = ARTIFACT_DIR / "flowsentry.joblib"

app = FastAPI(
    title="FlowSentry",
    version=__version__,
    description="Real-time hierarchical intrusion detection with a tunable reject option.",
)

_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        if not ARTIFACT.exists():
            raise FileNotFoundError(
                "model artifact missing; run `python -m flowsentry.train` first"
            )
        _bundle = joblib.load(ARTIFACT)
    return _bundle


class Flow(BaseModel):
    features: dict = Field(
        ...,
        description="NSL-KDD feature name -> value; missing values are defaulted.",
        examples=[{"protocol_type": "tcp", "service": "http", "flag": "SF", "src_bytes": 181,
                   "dst_bytes": 5450, "count": 8, "srv_count": 8, "same_srv_rate": 1.0}],
    )
    reject_threshold: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Abstain ('unknown') when final confidence is below this.",
    )


def _row(features: dict) -> pd.DataFrame:
    row = {c: features.get(c, 0) for c in COLUMNS}
    for c in CATEGORICAL:
        row[c] = features.get(c, "other")
    return pd.DataFrame([row])


@app.get("/health")
def health():
    try:
        _load()
        return {"status": "ok", "model": "loaded", "version": __version__}
    except FileNotFoundError:
        return {"status": "degraded", "model": "missing", "version": __version__}


@app.post("/predict")
def predict(req: Flow):
    try:
        bundle = _load()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    X = bundle["preprocessor"].transform(_row(req.features))
    labels, conf, escalated, abstained = bundle["model"].predict_detail(
        X, reject_threshold=req.reject_threshold
    )
    return {
        "label": str(labels[0]),
        "confidence": round(float(conf[0]), 4),
        "escalated_to_stage2": bool(escalated[0]),
        "abstained": bool(abstained[0]),
    }


@app.get("/curve")
def curve():
    path = ARTIFACT_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(status_code=503, detail="no metrics; run training first")
    return json.loads(path.read_text())

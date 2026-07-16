"""
Serving-path scorer: raw flow features in, verdict out, at serving speed.

Why this module exists (all measured, see artifacts/benchmark.json):

  The fitted forests carry n_jobs=-1, and sklearn's predict_proba spins up and
  tears down a joblib thread pool on every call. On a single-row request that
  overhead is ~30-60 ms, which capped the serving path at ~23 flows/s and was the
  README's embarrassing "20 flows/s" number. Scoring the trees sequentially
  (model.forest_proba) removes it: bit-identical probabilities at ~2-6 ms per
  flow. For large batches the thread pool amortizes and wins, so score_batch
  switches strategy at a measured cutoff: on the dev machine sequential is faster
  up to ~1k rows, the threaded path is faster from ~4k rows; the default cutoff
  sits between at 2048.

  Imputation uses the fitted SimpleImputer's medians directly
  (np.where(isnan, medians, X)), which a test asserts is exactly equal to
  imputer.transform, without its per-call validation.

FlowScorer is what the FastAPI service, the replay pipeline, and the dashboard
share, so single-row and batch scoring cannot drift apart.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .data import QUIC_FEATURES, STAGE2_FEATURES
from .model import UNKNOWN, TwoStageRejectClassifier

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"
ARTIFACT = ARTIFACT_DIR / "flowsentry.joblib"

# Batch size at which score_batch hands off from sequential tree scoring to the
# forests' native threaded path. Measured on the dev machine: sequential wins at
# <=1024 rows, threaded wins at >=4096 (see artifacts/benchmark.json); 2048 splits
# the bracket. Getting it somewhat wrong costs milliseconds, not correctness.
SEQUENTIAL_CUTOFF = 2048

_QUIC_SET = frozenset(QUIC_FEATURES)


def load_bundle(path: Path = ARTIFACT) -> dict:
    """Load the serving artifact dict: {imputer, model, ...}.

    Trust boundary: this is joblib (pickle) and must only ever load an artifact
    this repo trained locally (train.py or the Docker build). Never point it at a
    downloaded file; see docs/THREAT_MODEL.md.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"model artifact missing at {path}; run `python -m flowsentry.train` first"
        )
    import joblib

    return joblib.load(path)


class FlowScorer:
    """Scores flows with a trained (imputer, two-stage model) bundle.

    Two entry points, one semantics:
      score_one(features_dict)  the per-request path: build the feature row,
                                impute, sequential two-stage predict.
      score_batch(X)            whole-matrix path for replay/eval; picks the
                                faster scoring strategy from the batch size.
    A test asserts both return the same labels and confidences on the same rows.
    """

    def __init__(
        self,
        imputer: Any,
        model: TwoStageRejectClassifier,
        feature_names: list[str] | None = None,
        sequential_cutoff: int = SEQUENTIAL_CUTOFF,
    ) -> None:
        self.imputer = imputer
        self.model = model
        self.feature_names = list(feature_names or STAGE2_FEATURES)
        self.sequential_cutoff = sequential_cutoff
        self._medians = np.asarray(imputer.statistics_, dtype=np.float64)
        if self._medians.shape[0] != len(self.feature_names):
            raise ValueError(
                f"imputer has {self._medians.shape[0]} statistics for "
                f"{len(self.feature_names)} features; artifact and schema disagree"
            )

    @classmethod
    def from_bundle(cls, bundle: Mapping[str, Any]) -> FlowScorer:
        return cls(
            imputer=bundle["imputer"],
            model=bundle["model"],
            feature_names=bundle.get("stage2_features"),
        )

    @classmethod
    def from_artifact(cls, path: Path = ARTIFACT) -> FlowScorer:
        return cls.from_bundle(load_bundle(path))

    def row_from_features(self, features: Mapping[str, float]) -> np.ndarray:
        """One raw feature row in schema order. Missing UDP features become NaN
        (median-imputed next); missing QUIC features become 0, the honest default
        for a flow with no observed QUIC subflow. Unknown keys are ignored."""
        vals = [
            float(features[name])
            if name in features
            else (0.0 if name in _QUIC_SET else np.nan)
            for name in self.feature_names
        ]
        return np.asarray([vals], dtype=np.float64)

    def impute(self, X: np.ndarray) -> np.ndarray:
        """Median-impute NaNs; exactly equal to the fitted imputer.transform
        (asserted by a test) without its per-call validation overhead."""
        return np.where(np.isnan(X), self._medians, X)

    def score_one(self, features: Mapping[str, float], reject_threshold: float = 0.0) -> dict:
        """The /predict path: one flow dict -> verdict dict."""
        X = self.impute(self.row_from_features(features))
        labels, conf, escalated, abstained = self.model.predict_detail(
            X, reject_threshold=reject_threshold, sequential=True
        )
        return {
            "label": str(labels[0]),
            "confidence": round(float(conf[0]), 4),
            "escalated_to_stage2": bool(escalated[0]),
            "abstained": bool(abstained[0]),
        }

    def score_batch(self, X: np.ndarray, reject_threshold: float = 0.0):
        """Score a whole raw feature matrix (NaN = missing) in one call.
        Returns (labels, confidence, escalated_mask, abstained_mask)."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"expected a 2-D feature matrix, got shape {X.shape}")
        sequential = X.shape[0] <= self.sequential_cutoff
        return self.model.predict_detail(
            self.impute(X), reject_threshold=reject_threshold, sequential=sequential
        )


__all__ = [
    "ARTIFACT",
    "ARTIFACT_DIR",
    "SEQUENTIAL_CUTOFF",
    "UNKNOWN",
    "FlowScorer",
    "load_bundle",
]

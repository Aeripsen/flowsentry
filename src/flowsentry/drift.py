"""
Drift detection: population stability index (PSI) per feature, measured against
the training distribution.

Why this exists: the model card is explicit that the headline numbers mean
"recognize the same campaign", not "detect whatever comes next". When the traffic
feeding the model shifts (new campaign, new network, new capture point), the
first observable symptom is the feature distributions moving, long before anyone
has labels to re-score accuracy with. PSI is the cheap standard way to watch
that per feature.

Mechanics:
  * At training time, train.py computes a reference from the IMPUTED training
    matrix (the space the model actually consumes; imputation is part of the
    model's view of the world) and persists it in the artifact: per feature, the
    interior decile edges and the training proportion of each bin.
  * At scoring time, psi() bins a window of scored (imputed) rows with the same
    edges and compares proportions: psi_f = sum((p_win - p_ref) * ln(p_win/p_ref)).
  * `python -m flowsentry.stream --drift` reports the most drifted features of
    the replayed window.

Reading PSI, by the widely used industry convention (a convention, not a law):
  < 0.10 stable, 0.10-0.25 moderate shift worth a look, > 0.25 major shift.

Known blind spot, stated honestly: measuring post-imputation masks drift in
missingness itself (a flood of flows missing a feature arrives as a spike of
medians, which PSI usually still sees, but as the wrong story). Tracking
missingness rates alongside is the natural next step if this graduates from
replay tooling to a service loop.
"""
from __future__ import annotations

import numpy as np

N_BINS = 10
_EPS = 1e-4  # floor for empty bins so the log term stays finite (standard practice)

STABLE, MODERATE, MAJOR = "stable", "moderate", "major"


def band(psi_value: float) -> str:
    if psi_value < 0.10:
        return STABLE
    if psi_value <= 0.25:
        return MODERATE
    return MAJOR


def _proportions(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin proportions of x for interior edges (open outer bins on both sides)."""
    counts = np.bincount(np.searchsorted(edges, x, side="right"), minlength=len(edges) + 1)
    props = counts / max(len(x), 1)
    return np.maximum(props, _EPS)


def reference_from_matrix(
    X: np.ndarray, feature_names: list[str], n_bins: int = N_BINS
) -> dict:
    """Per-feature decile edges + training bin proportions, as plain lists so the
    reference survives any serialization. Constant features get no edges and are
    compared degenerately (all mass in one bin)."""
    X = np.asarray(X, dtype=np.float64)
    if X.shape[1] != len(feature_names):
        raise ValueError(f"{X.shape[1]} columns for {len(feature_names)} feature names")
    features: dict[str, dict] = {}
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    for j, name in enumerate(feature_names):
        col = X[:, j]
        edges = np.unique(np.quantile(col, quantiles))
        features[name] = {
            "edges": [float(e) for e in edges],
            "props": [float(p) for p in _proportions(col, edges)],
        }
    return {"n_bins": n_bins, "n_rows": int(X.shape[0]), "features": features}


def psi(reference: dict, X: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    """PSI per feature of window X (imputed rows, same column order as the
    reference) against the training reference."""
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] == 0:
        raise ValueError("empty window: PSI needs at least one row")
    out: dict[str, float] = {}
    for j, name in enumerate(feature_names):
        ref = reference["features"].get(name)
        if ref is None:
            raise KeyError(f"feature {name!r} missing from the drift reference")
        edges = np.asarray(ref["edges"], dtype=np.float64)
        p_ref = np.asarray(ref["props"], dtype=np.float64)
        p_win = _proportions(X[:, j], edges)
        out[name] = float(np.sum((p_win - p_ref) * np.log(p_win / p_ref)))
    return out


def drift_report(
    reference: dict, X: np.ndarray, feature_names: list[str], top: int = 10
) -> dict:
    """PSI for every feature plus the ranked top drifters and band counts."""
    scores = psi(reference, X, feature_names)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    bands = {STABLE: 0, MODERATE: 0, MAJOR: 0}
    for _, v in ranked:
        bands[band(v)] += 1
    return {
        "n_rows": int(np.asarray(X).shape[0]),
        "bands": bands,
        "top": [
            {"feature": k, "psi": round(v, 4), "band": band(v)} for k, v in ranked[:top]
        ],
    }

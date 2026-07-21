"""
Gradient-boosted arms for the model comparison: LightGBM and XGBoost.

The shipped model is bagged trees, the architecture from the paper. Boosted
trees are the other strong tabular family and the obvious reviewer question
("did you try a GBDT?"), so this module makes them first-class arms: same
features, same grouped split, same metrics, and a small printed grid instead of
a silent search, so the comparison in artifacts/gbdt_comparison.json is one a
reader can re-run and audit line by line.

Both estimators are single-stage joint models (all 132 UDP+QUIC features). That
is deliberate: the hierarchy benchmark already showed single joint models match
the two-stage design on this sample, so the honest comparison is model family
against model family, not model family entangled with an architecture change.

Both are optional dependencies (pip install -e ".[gbdt]"); the core package and
the shipped serving path do not grow a hard dependency for an experiment arm.

Imbalance is handled the way the shipped forest handles it, with balanced class
weights, so a macro-F1 difference means the model family differs, not the
weighting policy.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from .calibration import ece

# The grids are small on purpose: the point is a fair head to head against the
# committed baseline, not a leaderboard squeeze. Every entry is printed with its
# validation score by scripts/gbdt_comparison.py, so nothing is searched silently.
# n_jobs is fixed (not machine-dependent -1) because both libraries only promise
# repeatable results under a fixed thread count.
N_JOBS = 4

LIGHTGBM_GRID: list[dict[str, Any]] = [
    {"learning_rate": lr, "n_estimators": n, "num_leaves": leaves}
    for lr in (0.05, 0.1)
    for n in (200, 400)
    for leaves in (31, 63)
]

XGBOOST_GRID: list[dict[str, Any]] = [
    {"learning_rate": lr, "n_estimators": n, "max_depth": depth}
    for lr in (0.05, 0.1)
    for n in (200, 400)
    for depth in (6, 8)
]


def make_lightgbm(seed: int = 42, **params: Any) -> Any:
    """LGBMClassifier with the repeatability and imbalance defaults pinned.

    deterministic + force_row_wise + a fixed thread count is the combination
    LightGBM documents for run-to-run identical models; without it the same
    grid entry can score differently across machines and the printed grid
    stops being auditable.
    """
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError(
            "lightgbm is not installed; this is an optional arm: pip install -e '.[gbdt]'"
        ) from exc
    defaults: dict[str, Any] = {
        "random_state": seed,
        "deterministic": True,
        "force_row_wise": True,
        "n_jobs": N_JOBS,
        "class_weight": "balanced",
        "verbosity": -1,
    }
    return LGBMClassifier(**{**defaults, **params})


class EncodedXGBClassifier:
    """XGBoost behind the repo's string labels.

    XGBClassifier wants integer class ids at fit time; everything else in this
    repo passes family names. The wrapper owns the encoding so the estimator
    keeps the sklearn contract the rest of the code relies on: classes_ holds
    the original names and predict_proba columns follow them. Balanced sample
    weights stand in for LightGBM's class_weight, which XGBoost does not take.
    """

    def __init__(self, seed: int = 42, **params: Any) -> None:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed; this is an optional arm: pip install -e '.[gbdt]'"
            ) from exc
        defaults: dict[str, Any] = {
            "random_state": seed,
            "tree_method": "hist",
            "n_jobs": N_JOBS,
            "verbosity": 0,
        }
        self.params = {**defaults, **params}
        self.inner = XGBClassifier(**self.params)
        self._encoder = LabelEncoder()

    def fit(self, X: np.ndarray, y: np.ndarray) -> EncodedXGBClassifier:
        y_enc = self._encoder.fit_transform(np.asarray(y))
        weights = compute_sample_weight("balanced", y_enc)
        self.inner.fit(X, y_enc, sample_weight=weights)
        self.classes_: np.ndarray = self._encoder.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.inner.predict_proba(X)


def make_xgboost(seed: int = 42, **params: Any) -> EncodedXGBClassifier:
    return EncodedXGBClassifier(seed=seed, **params)


def top_label(proba: np.ndarray, classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(predicted labels, top-label confidence) from a probability matrix.

    This is the same quantity the two-stage model serves as `confidence`, so the
    reject curves of the arms are thresholding the same kind of number.
    """
    idx = proba.argmax(axis=1)
    labels = np.asarray(classes, dtype=object)[idx]
    conf = proba.max(axis=1)
    return labels, conf


def binary_attack_pr_auc(y_true: np.ndarray, proba: np.ndarray, classes: list[str]) -> float:
    """The repo's headline number: rank flows by 1 - P(benign), score with
    average precision against attack-or-not. Matches train.py exactly so the
    committed 0.9767 baseline is comparable."""
    is_attack = (np.asarray(y_true) != "benign").astype(int)
    p_benign = proba[:, classes.index("benign")]
    return float(average_precision_score(is_attack, 1.0 - p_benign))


def reject_curve(
    conf: np.ndarray, correct: np.ndarray, thresholds: list[float]
) -> list[dict[str, Any]]:
    """Coverage and reliability at each reject threshold, single-stage form of
    TwoStageRejectClassifier.coverage_reliability_curve (no escalation rate,
    because these arms have no second stage to escalate to)."""
    rows: list[dict[str, Any]] = []
    for t in thresholds:
        covered = conf >= t
        n = int(covered.sum())
        rows.append(
            {
                "threshold": round(float(t), 4),
                "coverage": round(float(covered.mean()), 4),
                "reliability": round(float(correct[covered].mean()), 4) if n else None,
                "n_covered": n,
            }
        )
    return rows


Calibrator = Callable[[np.ndarray], np.ndarray]


def fit_confidence_calibrators(conf: np.ndarray, correct: np.ndarray) -> dict[str, Calibrator]:
    """Fit both candidate maps confidence -> P(correct) on the same data.

    Isotonic is the flexible monotone map the repo's earlier calibration
    experiment used; Platt is a two-parameter sigmoid, which can win when the
    calibration split is small enough for isotonic to overfit its steps. C is
    large because Platt scaling is a curve fit, not a regularized classifier.
    """
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(conf, correct)
    platt = LogisticRegression(C=1e6).fit(conf.reshape(-1, 1), correct.astype(int))

    def apply_platt(c: np.ndarray) -> np.ndarray:
        return platt.predict_proba(np.asarray(c, dtype=float).reshape(-1, 1))[:, 1]

    def apply_iso(c: np.ndarray) -> np.ndarray:
        return np.asarray(iso.predict(np.asarray(c, dtype=float)))

    return {"isotonic": apply_iso, "platt": apply_platt}


def pick_calibrator(
    calibrators: dict[str, Calibrator], conf: np.ndarray, correct: np.ndarray
) -> tuple[str, dict[str, float]]:
    """Choose by ECE on data neither calibrator was fit on. Fitting and picking
    on the same split would always pick isotonic, because the more flexible map
    wins in-sample by construction."""
    scores = {name: round(ece(cal(conf), correct), 4) for name, cal in calibrators.items()}
    best = min(scores, key=lambda name: scores[name])
    return best, scores

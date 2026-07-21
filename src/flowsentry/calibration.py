"""
Calibration measurement for the served confidence number.

The reject knob is a threshold on confidence, and every operating guide in this
repo tells people to read the coverage-reliability curve instead of the
threshold's face value. That is the right advice precisely because the raw forest
vote fraction is not a probability. This module measures the gap rather than
leaving it as a caveat.

  reliability_bins  the curve itself: mean confidence against observed accuracy
                    inside each equal-width confidence bin, with the bin counts,
                    so a bin holding nine flows is not read as a trend.
  ece / mce         the summary: coverage-weighted mean gap, and the worst
                    single bin. ECE hides a bad bin that holds few flows, MCE is
                    the number that bin shows up in, so both are reported.
  brier             a proper scoring rule, which ECE is not: ECE can be driven
                    to zero by a model that always says 0.83 and is right 83% of
                    the time, and such a model is useless per flow. Brier only
                    goes down when the confidence is both well calibrated and
                    actually discriminating.

Everything here takes the top-label confidence and a 0/1 correctness vector,
which is what the reject knob thresholds, plus a multiclass Brier over the full
probability matrix for the score that does not throw the other classes away.
"""
from __future__ import annotations

import numpy as np

N_BINS = 10


def reliability_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> list[dict]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (conf >= lo) & (conf < hi) if hi < 1.0 else (conf >= lo) & (conf <= hi)
        n = int(mask.sum())
        rows.append(
            {
                "bin_low": round(float(lo), 2),
                "bin_high": round(float(hi), 2),
                "n": n,
                "share": round(float(mask.mean()), 4),
                "mean_confidence": round(float(conf[mask].mean()), 4) if n else None,
                "accuracy": round(float(correct[mask].mean()), 4) if n else None,
                "gap": round(float(conf[mask].mean() - correct[mask].mean()), 4) if n else None,
            }
        )
    return rows


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected calibration error: coverage-weighted |confidence - accuracy| over
    equal-width confidence bins."""
    return float(
        sum(r["share"] * abs(r["gap"]) for r in reliability_bins(conf, correct, n_bins) if r["n"])
    )


def mce(conf: np.ndarray, correct: np.ndarray, n_bins: int = N_BINS) -> float:
    """Worst single bin, ignoring how few flows it holds."""
    gaps = [abs(r["gap"]) for r in reliability_bins(conf, correct, n_bins) if r["n"]]
    return float(max(gaps)) if gaps else 0.0


def brier_top_label(conf: np.ndarray, correct: np.ndarray) -> float:
    return float(np.mean((conf - correct) ** 2))


def brier_multiclass(proba: np.ndarray, y_true: np.ndarray, classes: list[str]) -> float:
    """Mean squared error against the one-hot truth, summed over classes.

    Range is 0 to 2 (the usual multiclass convention: a confidently wrong flow
    costs 1 on its own class and 1 on the true one)."""
    onehot = np.zeros_like(proba)
    index = {c: j for j, c in enumerate(classes)}
    for i, label in enumerate(y_true):
        j = index.get(str(label))
        if j is not None:
            onehot[i, j] = 1.0
    return float(np.mean(((proba - onehot) ** 2).sum(axis=1)))

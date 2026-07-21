"""Measure whether calibrating the final confidence would buy the reject knob
anything. Run:  python scripts/calibration_experiment.py

Writes artifacts/calibration_experiment.json and prints the same numbers.

Design, leakage-safe end to end:
  * the held-out TEST split is carved exactly as train.py does (grouped, seed 42),
    and is never touched by fitting or calibration;
  * a CALIBRATION split is then carved out of the remaining training connections
    (grouped again, seed 43), the two-stage model is fit on what is left, and an
    isotonic mapping confidence -> P(correct) is fit on the calibration split.

Two measurements on the untouched test split:
  1. ECE (expected calibration error, 10 equal-width bins, from
     flowsentry.calibration) of the raw two-stage max-probability confidence,
     before and after the isotonic mapping. This asks: does "confidence 0.9"
     actually mean 90% correct?
  2. The coverage-reliability curve under raw vs calibrated confidence at matched
     coverage. Isotonic regression is monotone, so it cannot re-rank flows, so the
     curve should not improve; this measures that expectation instead of assuming
     it (ties introduced by isotonic's flat segments could still coarsen the
     curve, which would show up here).

The model fit here trains on ~75% of the usual training split (the calibration
data has to come from somewhere), so its absolute numbers differ slightly from
the shipped model's. That is inherent to answering the question honestly and is
why this is an experiment script, not part of train.py; see the model card for
the conclusion and what shipping a calibrator would cost.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flowsentry.calibration import ece  # noqa: E402
from flowsentry.data import (  # noqa: E402
    STAGE1_INDICES,
    build_matrices,
    leakage_safe_split,
    load_sample,
)
from flowsentry.model import TwoStageRejectClassifier  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "calibration_experiment.json"
THRESHOLDS = np.round(np.linspace(0.0, 0.99, 34), 3)


def curve(conf: np.ndarray, correct: np.ndarray, thresholds) -> list[dict]:
    rows = []
    for t in thresholds:
        covered = conf >= t
        n = int(covered.sum())
        rows.append(
            {
                "threshold": float(t),
                "coverage": round(float(covered.mean()), 4),
                "reliability": round(float(correct[covered].mean()), 4) if n else None,
            }
        )
    return rows


def reliability_at_coverage(rows: list[dict], target: float) -> float | None:
    """Reliability of the tightest operating point that still covers >= target."""
    feasible = [r for r in rows if r["reliability"] is not None and r["coverage"] >= target]
    if not feasible:
        return None
    return min(feasible, key=lambda r: r["coverage"])["reliability"]


def main() -> dict:
    df = load_sample()
    X, y, groups = build_matrices(df)
    tr, te = leakage_safe_split(groups, test_size=0.25, seed=42)

    # carve a grouped calibration split out of the training connections
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=43)
    sub_i, cal_i = next(gss.split(np.zeros(len(tr)), groups=groups[tr]))
    sub, cal = tr[sub_i], tr[cal_i]
    assert not (set(groups[sub]) & set(groups[cal]))
    assert not (set(groups[cal]) & set(groups[te]))

    imputer = SimpleImputer(strategy="median").fit(X[sub])
    model = TwoStageRejectClassifier(stage1_features=STAGE1_INDICES).fit(
        imputer.transform(X[sub]), y[sub]
    )

    def conf_correct(idx):
        labels, conf, _, _ = model.predict_detail(imputer.transform(X[idx]))
        return conf, (labels == y[idx]).astype(float)

    conf_cal, correct_cal = conf_correct(cal)
    conf_te, correct_te = conf_correct(te)

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
        conf_cal, correct_cal
    )
    conf_te_cal = iso.predict(conf_te)

    raw_curve = curve(conf_te, correct_te, THRESHOLDS)
    cal_curve = curve(conf_te_cal, correct_te, THRESHOLDS)
    matched = []
    for target in (0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65):
        matched.append(
            {
                "coverage_target": target,
                "reliability_raw": reliability_at_coverage(raw_curve, target),
                "reliability_calibrated": reliability_at_coverage(cal_curve, target),
            }
        )

    report = {
        "what": (
            "isotonic confidence calibration experiment; model fit on train minus a "
            "grouped calibration split, evaluated on the untouched test split"
        ),
        "n_fit": int(len(sub)),
        "n_calibration": int(len(cal)),
        "n_test": int(len(te)),
        "ece_raw": round(ece(conf_te, correct_te), 4),
        "ece_calibrated": round(ece(conf_te_cal, correct_te), 4),
        "mean_confidence_raw": round(float(conf_te.mean()), 4),
        "mean_confidence_calibrated": round(float(conf_te_cal.mean()), 4),
        "test_accuracy_full_coverage": round(float(correct_te.mean()), 4),
        "reliability_at_matched_coverage": matched,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"[save] {OUT}")
    return report


if __name__ == "__main__":
    main()

"""Calibration of the SHIPPED model's confidence on the held-out test split.
Run:  python scripts/calibration_report.py

Writes artifacts/calibration.json and prints the same numbers.

This is not the same measurement as scripts/calibration_experiment.py. That one
asks whether calibrating would help the reject knob, and to answer it honestly it
has to refit the forests on ~75% of the training split so an isotonic mapping can
be fit on held-out data. Its ECE therefore belongs to a model this repo does not
ship. This script measures the model people actually get: same config, same seed,
same full training split as train.py, scored on the same leakage-safe test split.

What it reports:

  * the reliability curve, binned by confidence, with bin counts, so the shape is
    visible and thin bins are not read as trends;
  * ECE and MCE, the average and worst-bin gap;
  * Brier, top-label and multiclass, because ECE is not a proper scoring rule and
    a constant-confidence model can score a perfect ECE while being useless;
  * the same numbers split by scoring path, Stage 1 answered against escalated to
    Stage 2. The escalated flows are by construction the ones Stage 1 was unsure
    about, so if the confidence number means anything the two paths should not be
    equally trustworthy, and the reject knob thresholds them with one number.

No model artifact is written; train.py stays the only thing that produces
artifacts/flowsentry.joblib.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flowsentry.calibration import (  # noqa: E402
    brier_multiclass,
    brier_top_label,
    ece,
    mce,
    reliability_bins,
)
from flowsentry.config import get_settings  # noqa: E402
from flowsentry.data import (  # noqa: E402
    STAGE1_INDICES,
    build_matrices,
    leakage_safe_split,
    load_sample,
)
from flowsentry.model import TwoStageRejectClassifier  # noqa: E402
from flowsentry.registry import make_stage_estimator  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "calibration.json"


def summary(conf: np.ndarray, correct: np.ndarray) -> dict:
    return {
        "n": int(len(conf)),
        "mean_confidence": round(float(conf.mean()), 4),
        "accuracy": round(float(correct.mean()), 4),
        "overconfidence": round(float(conf.mean() - correct.mean()), 4),
        "ece": round(ece(conf, correct), 4),
        "mce": round(mce(conf, correct), 4),
        "brier_top_label": round(brier_top_label(conf, correct), 4),
    }


def main() -> dict:
    cfg = get_settings().training
    df = load_sample()
    X, y, groups = build_matrices(df)
    tr, te = leakage_safe_split(groups, test_size=cfg.test_size, seed=cfg.seed)

    imputer = SimpleImputer(strategy="median").fit(X[tr])
    Xtr, Xte = imputer.transform(X[tr]), imputer.transform(X[te])
    yte = y[te]

    model = TwoStageRejectClassifier(
        stage1_features=STAGE1_INDICES,
        escalate_threshold=cfg.escalate_threshold,
        stage1_estimator=make_stage_estimator(cfg.stage_estimator, **cfg.stage1_params),
        stage2_estimator=make_stage_estimator(cfg.stage_estimator, **cfg.stage2_params),
    ).fit(Xtr, y[tr])

    classes = [str(c) for c in model.classes_]
    labels, conf, escalated, _ = model.predict_detail(Xte)
    proba = model.predict_proba(Xte)
    correct = (labels == yte).astype(float)

    report = {
        "what": (
            "calibration of the shipped model's confidence on the connection-grouped "
            "leakage-safe held-out test split; same config and training split as "
            "train.py, no calibrator fitted"
        ),
        "config": {
            "test_size": cfg.test_size,
            "seed": cfg.seed,
            "stage_estimator": cfg.stage_estimator,
            "escalate_threshold": cfg.escalate_threshold,
        },
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "n_bins": 10,
        "overall": summary(conf, correct),
        "brier_multiclass": round(brier_multiclass(proba, yte, classes), 4),
        "reliability_curve": reliability_bins(conf, correct),
        "by_scoring_path": {
            "stage1_answered": summary(conf[~escalated], correct[~escalated]),
            "escalated_to_stage2": summary(conf[escalated], correct[escalated]),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"[save] {OUT}")
    return report


if __name__ == "__main__":
    main()

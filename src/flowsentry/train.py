"""
Train + evaluate FlowSentry on the NSL-KDD public benchmark, then save the
serving artifact (preprocessor + model) and a metrics report.

Run:  python -m flowsentry.train
Reports only real measured numbers (macro-F1, per-class F1, and the
coverage-vs-reliability curve). No invented impact figures.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, f1_score

from .data import load
from .model import TwoStageRejectClassifier

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"
THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def main() -> dict:
    print("[data] loading NSL-KDD (benchmark) ...")
    X_train, y_train, X_test, y_test, pre, feature_names = load()
    print(f"[data] train={X_train.shape} test={X_test.shape} features={len(feature_names)}")

    print("[fit ] two-stage reject classifier ...")
    model = TwoStageRejectClassifier()
    model.fit(X_train, y_train)

    # Headline metric: full-coverage (no abstain) macro-F1 on the held-out test set.
    y_pred = model.predict(X_test, reject_threshold=0.0)
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    curve = model.coverage_reliability_curve(X_test, y_test, THRESHOLDS)

    metrics = {
        "dataset": "NSL-KDD (public benchmark; KDDTrain+ / KDDTest+)",
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": len(feature_names),
        "classes": sorted(set(map(str, y_test))),
        "macro_f1_full_coverage": round(macro_f1, 4),
        "per_class_f1": {k: round(v["f1-score"], 4) for k, v in report.items()
                         if k in set(map(str, np.unique(y_test)))},
        "coverage_reliability_curve": curve,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"preprocessor": pre, "model": model, "feature_names": feature_names},
                ARTIFACT_DIR / "flowsentry.joblib")
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))
    print(f"[save] {ARTIFACT_DIR / 'flowsentry.joblib'}")
    return metrics


if __name__ == "__main__":
    main()

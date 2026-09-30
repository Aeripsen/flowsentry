"""
The per-flow export behind the live demo page
(https://aeripsen.github.io/flowsentry/), and the check that keeps the page
from drifting away from the committed artifacts.

What it writes: artifacts/demo_flows.json, every flow of the connection-grouped
held-out test split with the shipped two-stage model's predicted class, its
confidence, whether Stage 1 escalated it, P(benign), and the true class. The
page applies the reject knob to these rows in the browser; it never runs the
model and never invents a flow. The zero-day panel reads the committed
artifacts/zero_day_lofo.json directly.

Why it can be trusted: check_export() rebuilds, from the exported rows alone,
  1. the committed coverage-reliability curve and the binary attack-detection
     PR-AUC in metrics.json, and
  2. the full-coverage confusion counts in per_family.json.
CI runs it offline through tests/test_demo_data.py before pages.yml deploys.

Confidences are exported at full float precision: forest confidences sit on
multiples of 1/n_trees, right on the committed thresholds (0.9, 0.95, 0.99),
so rounding them could silently flip a flow across the knob.

Run: python scripts/demo_data.py   (make demo-data; ~1 min, trains if the
local artifact is missing)
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import average_precision_score

from .config import get_settings
from .data import build_matrices, load_sample


def curve(pred: np.ndarray, conf: np.ndarray, y: np.ndarray, escalated: np.ndarray,
          thresholds: list[float]) -> list[dict[str, Any]]:
    """TwoStageRejectClassifier.coverage_reliability_curve, from exported rows."""
    rows = []
    for t in thresholds:
        covered = conf >= t
        n_cov = int(covered.sum())
        rel = float((pred[covered] == y[covered]).mean()) if n_cov else float("nan")
        rows.append(
            {
                "threshold": round(float(t), 4),
                "coverage": round(float(covered.mean()), 4),
                "reliability": round(rel, 4) if n_cov else None,
                "n_covered": n_cov,
                "escalation_rate": round(float(escalated.mean()), 4),
            }
        )
    return rows


def rows_from_export(demo: dict[str, Any]) -> tuple[np.ndarray, ...]:
    classes = np.asarray(demo["classes"], dtype=object)
    y = classes[np.asarray(demo["true"], dtype=int)]
    pred = classes[np.asarray(demo["pred"], dtype=int)]
    conf = np.asarray(demo["confidence"], dtype=float)
    escalated = np.asarray(demo["escalated"], dtype=bool)
    p_benign = np.asarray(demo["p_benign"], dtype=float)
    return y, pred, conf, escalated, p_benign


def check_export(
    demo: dict[str, Any], metrics: dict[str, Any], per_family: dict[str, Any] | None = None
) -> list[str]:
    errors: list[str] = []
    y, pred, conf, escalated, p_benign = rows_from_export(demo)
    if len(y) != metrics["n_test"]:
        errors.append("export size differs from metrics.json n_test")
    thresholds = [r["threshold"] for r in metrics["coverage_reliability_curve"]]
    if curve(pred, conf, y, escalated, thresholds) != metrics["coverage_reliability_curve"]:
        errors.append("export does not regenerate the committed coverage-reliability curve")
    is_attack = (y != "benign").astype(int)
    pr_auc = round(float(average_precision_score(is_attack, 1.0 - p_benign)), 4)
    if pr_auc != metrics["binary_attack_detection_pr_auc"]:
        errors.append(f"export binary PR-AUC {pr_auc} != committed "
                      f"{metrics['binary_attack_detection_pr_auc']}")
    if per_family is not None:
        for fam, block in per_family["confusion_full_coverage"].items():
            called = Counter(pred[y == fam].tolist())
            if int((y == fam).sum()) != block["n_flows"] or dict(called) != block["called"]:
                errors.append(f"export does not regenerate per_family confusion for {fam}")
    return errors


def main() -> dict[str, Any]:
    cfg = get_settings()
    art_path = Path(cfg.artifact_dir) / "flowsentry.joblib"
    if not art_path.exists():
        from .train import main as train_main

        train_main()
    art = joblib.load(art_path)
    X, y_all, _ = build_matrices(load_sample())
    te = np.asarray(art["test_indices"], dtype=int)
    Xte = art["imputer"].transform(X[te])
    model = art["model"]
    pred, conf, escalated, _ = model.predict_detail(Xte, reject_threshold=0.0)
    classes = list(model.classes_)
    p_benign = model.predict_proba(Xte)[:, classes.index("benign")]
    index = {c: i for i, c in enumerate(classes)}
    y = np.asarray(y_all)[te]

    metrics = json.loads((Path(cfg.artifact_dir) / "metrics.json").read_text())
    demo = {
        "dataset": metrics["dataset"],
        "what": (
            "Every flow of the connection-grouped held-out test split: the shipped "
            "two-stage model's predicted class, its confidence (the reject knob's input), "
            "whether Stage 1 escalated it to Stage 2, P(benign), and the true class. The "
            "demo page applies the knob to these rows; tests/test_demo_data.py rebuilds "
            "the committed curve, binary PR-AUC and per-family confusion from them."
        ),
        "rebuild": "make train && make demo-data",
        "n": int(len(te)),
        "classes": classes,
        "committed_thresholds": [r["threshold"] for r in metrics["coverage_reliability_curve"]],
        "true": [index[c] for c in y],
        "pred": [index[c] for c in pred],
        "confidence": [float(c) for c in conf],
        "escalated": [int(e) for e in escalated],
        "p_benign": [float(v) for v in p_benign],
    }
    per_family_path = Path(cfg.artifact_dir) / "per_family.json"
    per_family = json.loads(per_family_path.read_text()) if per_family_path.exists() else None
    errors = check_export(demo, metrics, per_family)
    if errors:
        raise SystemExit(
            "FAIL: " + "; ".join(errors) + ". The local artifact is not the committed "
            "model; run `make train` (or `make reproduce`) and retry."
        )
    out = Path(cfg.artifact_dir) / "demo_flows.json"
    out.write_text(json.dumps(demo, separators=(",", ":")), newline="\n")
    print(f"[save ] {out} ({out.stat().st_size / 1e6:.2f} MB); rebuilds metrics.json "
          f"curve, binary PR-AUC and per_family confusion")
    return demo


if __name__ == "__main__":
    main()

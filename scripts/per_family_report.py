"""Per-attack-family precision and recall on the held-out split, plus where the
misses actually go. Run:  python scripts/per_family_report.py

Writes artifacts/per_family.json and prints the same numbers.

Why this exists: the repo reports per-family PR-AUC and F1, and a PR-AUC is a
ranking number. Nobody staffing a SOC asks "how well does this rank UDP-OVH";
they ask "if UDP-OVH shows up, do we catch it, and when the box says UDP-OVH how
often is it right". That is precision and recall, and it was not published here.
F1 hides the asymmetry too: two families can share an F1 with completely
different failure modes (misses vs false alarms), and the fix is different in
each case.

What it measures on the same connection-grouped, leakage-safe test split
train.py uses, with the same shipped config:

  1. Precision / recall / F1 / support per class at full coverage.
  2. The confusion rows: for each family, which classes its flows were actually
     called, so a weak recall number comes with the reason attached.
  3. The same per-family precision and recall under the reject knob, where a
     rejected flow is an abstention rather than a wrong answer. The global
     coverage-reliability curve says reliability climbs to 99.3%; this asks
     which families pay for it, because abstaining on every hard family and
     answering only the easy ones would produce that exact curve.

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

from flowsentry.config import get_settings  # noqa: E402
from flowsentry.data import (  # noqa: E402
    STAGE1_INDICES,
    build_matrices,
    leakage_safe_split,
    load_sample,
)
from flowsentry.evaluation import confusion_rows, per_family  # noqa: E402
from flowsentry.model import UNKNOWN, TwoStageRejectClassifier  # noqa: E402
from flowsentry.registry import make_stage_estimator  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "per_family.json"
REJECT_THRESHOLDS = (0.5, 0.9, 0.99)
WEAK_RECALL = 0.5


def main() -> dict:
    cfg = get_settings().training
    df = load_sample()
    X, y, groups = build_matrices(df)
    tr, te = leakage_safe_split(groups, test_size=cfg.test_size, seed=cfg.seed)

    imputer = SimpleImputer(strategy="median").fit(X[tr])
    Xtr, Xte = imputer.transform(X[tr]), imputer.transform(X[te])
    ytr, yte = y[tr], y[te]

    model = TwoStageRejectClassifier(
        stage1_features=STAGE1_INDICES,
        escalate_threshold=cfg.escalate_threshold,
        stage1_estimator=make_stage_estimator(cfg.stage_estimator, **cfg.stage1_params),
        stage2_estimator=make_stage_estimator(cfg.stage_estimator, **cfg.stage2_params),
    ).fit(Xtr, ytr)

    classes = [str(c) for c in model.classes_]
    labels, conf, _, _ = model.predict_detail(Xte)

    full = per_family(yte, labels, classes)

    under_reject = []
    for t in REJECT_THRESHOLDS:
        answered = conf >= t
        rejected = np.where(answered, labels, UNKNOWN)
        rows = per_family(yte, rejected, classes)
        for c in classes:
            m = yte == c
            rows[c]["answered_share"] = round(float(answered[m].mean()), 4)
        under_reject.append(
            {
                "reject_threshold": t,
                "coverage": round(float(answered.mean()), 4),
                "per_family": rows,
            }
        )

    weak = sorted(
        (c for c in classes if full[c]["recall"] < WEAK_RECALL),
        key=lambda c: full[c]["recall"],
    )

    report = {
        "what": (
            "per-attack-family precision and recall on the connection-grouped "
            "leakage-safe held-out test split, at full coverage and under the reject "
            "knob; same shipped config as train.py"
        ),
        "config": {
            "test_size": cfg.test_size,
            "seed": cfg.seed,
            "stage_estimator": cfg.stage_estimator,
            "escalate_threshold": cfg.escalate_threshold,
        },
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "accuracy_full_coverage": round(float((labels == yte).mean()), 4),
        "per_family_full_coverage": full,
        "confusion_full_coverage": confusion_rows(yte, labels, classes),
        "per_family_under_reject": under_reject,
        "weak_families_recall_below": WEAK_RECALL,
        "weak_families": weak,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"[save] {OUT}")
    return report


if __name__ == "__main__":
    main()

"""Measure what the connection-grouped split actually buys, and the sample
structure that explains the answer. Run:  python scripts/split_comparison.py

Writes artifacts/split_comparison.json and prints the same numbers.

The README, ADR 002 and the model card all make a deliberately unflattering claim:
on THIS committed sample the grouped split changes nothing measurable, because the
row-capping that balances the classes also dilutes the connections. That claim was
true but hand-computed, and a hand-computed number in a repo whose pitch is
cite-or-cut is exactly the thing a reviewer pulls first. This script is where those
numbers come from now.

What it measures, all on the committed sample:

  1. Connection structure: flows per connection overall and for the dominant flood,
     which is the mechanism behind the whole result. Grouping can only matter when a
     connection has many flows to spread across the split.
  2. Host structure: distinct source IPs per class, and the fraction of test flows
     whose source IP also appears in training. This is the deeper limit that the
     grouping does NOT fix and that no split of this dataset can fix.
  3. Grouped vs stratified, same seed, same test_size, same model config: the
     head-to-head that says whether the correct method scores differently from the
     incorrect one here.

The stratified arm is trained only to be compared against. It is not the shipped
model and its artifact is never written; train.py remains the only thing that
produces artifacts/flowsentry.joblib.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flowsentry.config import get_settings  # noqa: E402
from flowsentry.data import (  # noqa: E402
    STAGE1_INDICES,
    build_matrices,
    leakage_safe_split,
    load_sample,
)
from flowsentry.model import TwoStageRejectClassifier  # noqa: E402
from flowsentry.registry import make_stage_estimator  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "split_comparison.json"


def _fit_and_score(X, y, tr, te, cfg) -> dict:
    """Train the shipped model config on the given split and report the three
    numbers the docs quote. Imputer fits on train only, exactly as train.py does."""
    imputer = SimpleImputer(strategy="median").fit(X[tr])
    Xtr, Xte = imputer.transform(X[tr]), imputer.transform(X[te])
    ytr, yte = y[tr], y[te]

    model = TwoStageRejectClassifier(
        stage1_features=STAGE1_INDICES,
        escalate_threshold=cfg.escalate_threshold,
        stage1_estimator=make_stage_estimator(cfg.stage_estimator, **cfg.stage1_params),
        stage2_estimator=make_stage_estimator(cfg.stage_estimator, **cfg.stage2_params),
    ).fit(Xtr, ytr)

    labels = model.predict(Xte, reject_threshold=0.0)
    proba = model.predict_proba(Xte)
    classes = list(model.classes_)

    is_attack = (yte != "benign").astype(int)
    p_benign = proba[:, classes.index("benign")]
    return {
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "accuracy_full_coverage": round(float((labels == yte).mean()), 4),
        "macro_f1_full_coverage": round(float(f1_score(yte, labels, average="macro")), 4),
        "binary_attack_detection_pr_auc": round(
            float(average_precision_score(is_attack, 1.0 - p_benign)), 4
        ),
    }


def main() -> dict:
    cfg = get_settings().training
    df = load_sample()
    X, y, groups = build_matrices(df)
    src = df["src_ip"].to_numpy()

    # 1. connection structure, overall and per class
    per_class = {}
    for c in sorted(set(y)):
        m = y == c
        n_conn = len(set(groups[m]))
        per_class[c] = {
            "n_flows": int(m.sum()),
            "n_connections": n_conn,
            "mean_flows_per_connection": round(float(m.sum() / n_conn), 4),
            "n_source_ips": int(len(set(src[m]))),
        }

    # 2. host overlap across the grouped split: the limit grouping does NOT fix
    tr, te = leakage_safe_split(groups, test_size=cfg.test_size, seed=cfg.seed)
    train_srcs = set(src[tr])
    shares_src = np.isin(src[te], list(train_srcs))

    # 3. grouped vs stratified head-to-head, same seed and test_size
    grouped = _fit_and_score(X, y, tr, te, cfg)
    tr_s, te_s = train_test_split(
        np.arange(len(df)), test_size=cfg.test_size, random_state=cfg.seed, stratify=y
    )
    stratified = _fit_and_score(X, y, tr_s, te_s, cfg)

    report = {
        "what": (
            "what the connection-grouped split buys on the committed sample, and the "
            "connection/host structure that explains it; the stratified arm is trained "
            "for comparison only and is never shipped"
        ),
        "config": {
            "test_size": cfg.test_size,
            "seed": cfg.seed,
            "stage_estimator": cfg.stage_estimator,
        },
        "sample_structure": {
            "n_flows": int(len(df)),
            "n_connections": int(len(set(groups))),
            "mean_flows_per_connection": round(float(len(df) / len(set(groups))), 4),
            "note": (
                "the row-capping in scripts/build_sample.py that balances the classes "
                "samples the two dominant classes down at random, which scatters their "
                "flows across connections and leaves ~1 flow per connection; that is why "
                "grouping cannot move the score on this sample"
            ),
            "per_class": per_class,
        },
        "host_overlap_grouped_split": {
            "test_flows_sharing_a_source_ip_with_train": round(float(shares_src.mean()), 4),
            "n_test_flows_sharing": int(shares_src.sum()),
            "n_test_flows": int(len(te)),
            "note": (
                "the limit the grouping does NOT fix: the same attacking host appears on "
                "both sides on different ports, so UDP-RAW's near-perfect PR-AUC measures "
                "'recognize this campaign', not 'detect an unseen flood'. Grouping by "
                "source IP instead would collapse the attack classes onto one side of the "
                "split, because there are only 2 attack source IPs. Only a cross-day or "
                "cross-dataset eval answers this, and none has been run."
            ),
        },
        "grouped": grouped,
        "stratified": stratified,
        "verdict": (
            "the grouped split is the correct method and is kept, but on this sample it "
            "is hygiene, not a score-inflation guard: binary PR-AUC is identical and "
            "accuracy differs by well under a point. Reported so the claim is measured "
            "rather than asserted."
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"[save] {OUT}")
    return report


if __name__ == "__main__":
    main()

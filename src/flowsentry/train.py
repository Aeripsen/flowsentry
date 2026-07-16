"""
Train + evaluate FlowSentry on the real BCCC-UDP-QUIC-IDS-2025 sample, then save the
serving artifact (imputer + two-stage model) and a metrics report.

Run:  python -m flowsentry.train

Everything reported here is a REAL measured number on a connection-grouped,
leakage-safe held-out test split of the public dataset. No invented figures.

Pipeline:
  1. load the committed BCCC sample (benign + 7 UDP DDoS families)
  2. leakage-safe split: no 5-tuple connection in both train and test
  3. fit a median imputer on the TRAIN split only, apply to both
  4. fit the two-stage model: Stage 1 on UDP-only features, Stage 2 on UDP+QUIC
  5. report macro-F1, per-class F1, PR-AUC (binary attack detection + macro OVR +
     per-class), the coverage-reliability curve, and a single-RF ablation.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, classification_report, f1_score
from sklearn.preprocessing import label_binarize

from .data import (
    STAGE1_INDICES,
    STAGE2_FEATURES,
    UDP_FEATURES,
    build_matrices,
    leakage_safe_split,
    load_sample,
)
from .model import TwoStageRejectClassifier

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts"
THRESHOLDS = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
ESCALATE_THRESHOLD = 0.90


def _pr_auc(y_true_bin, scores) -> float:
    return float(average_precision_score(y_true_bin, scores))


def main() -> dict:
    print("[data] loading BCCC-UDP-QUIC-IDS-2025 sample ...")
    df = load_sample()
    X, y, groups = build_matrices(df)
    tr, te = leakage_safe_split(groups, test_size=0.25, seed=42)
    print(
        f"[data] rows={len(df)} features={X.shape[1]} "
        f"(udp={len(UDP_FEATURES)}, joint={len(STAGE2_FEATURES)}) "
        f"| train={len(tr)} test={len(te)} "
        f"| train_conns={len(set(groups[tr]))} test_conns={len(set(groups[te]))}"
    )

    # Impute on train only (leakage-safe), then apply to both splits.
    imputer = SimpleImputer(strategy="median").fit(X[tr])
    Xtr, Xte = imputer.transform(X[tr]), imputer.transform(X[te])
    ytr, yte = y[tr], y[te]

    print("[fit ] two-stage reject classifier (Stage 1 = UDP-only, Stage 2 = UDP+QUIC) ...")
    model = TwoStageRejectClassifier(
        stage1_features=STAGE1_INDICES, escalate_threshold=ESCALATE_THRESHOLD
    )
    model.fit(Xtr, ytr)

    labels = model.predict(Xte, reject_threshold=0.0)
    proba = model.predict_proba(Xte)
    classes = list(model.classes_)

    macro_f1 = float(f1_score(yte, labels, average="macro"))
    accuracy = float((labels == yte).mean())
    report = classification_report(yte, labels, output_dict=True, zero_division=0)
    per_class_f1 = {c: round(report[c]["f1-score"], 4) for c in classes if c in report}

    # PR-AUC (average precision).
    Yte_bin = label_binarize(yte, classes=classes)
    per_class_prauc, n_pos = {}, {}
    for j, c in enumerate(classes):
        pos = int(Yte_bin[:, j].sum())
        n_pos[c] = pos
        if pos > 0:
            per_class_prauc[c] = round(_pr_auc(Yte_bin[:, j], proba[:, j]), 4)
    macro_prauc = round(float(np.mean(list(per_class_prauc.values()))), 4)

    # Binary DDoS-detection view derived from the multi-class scores.
    is_attack = (yte != "benign").astype(int)
    p_benign = proba[:, classes.index("benign")] if "benign" in classes else np.zeros(len(yte))
    binary_attack_prauc = round(_pr_auc(is_attack, 1.0 - p_benign), 4)
    benign_prauc = round(_pr_auc((yte == "benign").astype(int), p_benign), 4)

    curve = model.coverage_reliability_curve(Xte, yte, THRESHOLDS)
    escalation_rate = curve[0]["escalation_rate"]

    # Ablation: single-stage RF on the full UDP+QUIC space (no hierarchy, no reject),
    # so the value of the two-stage design is measurable, not asserted.
    single = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced_subsample"
    ).fit(Xtr, ytr)
    single_pred = single.classes_[single.predict_proba(Xte).argmax(axis=1)]
    single_macro_f1 = round(float(f1_score(yte, single_pred, average="macro")), 4)
    single_acc = round(float((single_pred == yte).mean()), 4)

    metrics = {
        "dataset": "BCCC-UDP-QUIC-IDS-2025 (real; public CC BY 4.0)",
        "data_note": (
            "committed stratified sample of the public dataset: all flows of the 7 "
            "rare UDP DDoS families + capped benign/UDP-RAW. Metrics are real, "
            "measured on a connection-grouped leakage-safe held-out test split."
        ),
        "split": (
            "GroupShuffleSplit on the UDP 5-tuple connection; "
            "no connection in both train and test"
        ),
        "n_rows": int(len(df)),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "n_features_stage1_udp": len(UDP_FEATURES),
        "n_features_stage2_joint": len(STAGE2_FEATURES),
        "classes": classes,
        "test_class_support": {c: n_pos[c] for c in classes},
        "escalate_threshold": ESCALATE_THRESHOLD,
        "escalation_rate": escalation_rate,
        "accuracy_full_coverage": round(accuracy, 4),
        "macro_f1_full_coverage": round(macro_f1, 4),
        "binary_attack_detection_pr_auc": binary_attack_prauc,
        "benign_detection_pr_auc": benign_prauc,
        "macro_pr_auc_ovr": macro_prauc,
        "per_class_pr_auc": per_class_prauc,
        "per_class_f1": per_class_f1,
        "coverage_reliability_curve": curve,
        "ablation_single_rf": {
            "note": "single 200-tree RF on the full UDP+QUIC space, no hierarchy/reject",
            "macro_f1_full_coverage": single_macro_f1,
            "accuracy_full_coverage": single_acc,
        },
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "imputer": imputer,
            "model": model,
            "stage1_features": STAGE1_INDICES,
            "udp_features": list(UDP_FEATURES),
            "stage2_features": list(STAGE2_FEATURES),
            "classes": classes,
        },
        ARTIFACT_DIR / "flowsentry.joblib",
    )
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))
    print(f"[save] {ARTIFACT_DIR / 'flowsentry.joblib'}")
    return metrics


if __name__ == "__main__":
    main()

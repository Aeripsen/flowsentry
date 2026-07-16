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

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, classification_report, f1_score
from sklearn.preprocessing import label_binarize

from .config import get_settings
from .data import (
    STAGE1_INDICES,
    STAGE2_FEATURES,
    UDP_FEATURES,
    build_matrices,
    leakage_safe_split,
    load_sample,
)
from .drift import reference_from_matrix
from .model import TwoStageRejectClassifier
from .registry import make_stage_estimator


def _pr_auc(y_true_bin, scores) -> float:
    return float(average_precision_score(y_true_bin, scores))


def main() -> dict:
    # Defaults in config.py are the exact values every reported number was
    # measured with; env vars / flowsentry.yaml override them (see config.py).
    cfg = get_settings()
    train_cfg = cfg.training

    print("[data] loading BCCC-UDP-QUIC-IDS-2025 sample ...")
    df = load_sample()
    X, y, groups = build_matrices(df)
    tr, te = leakage_safe_split(groups, test_size=train_cfg.test_size, seed=train_cfg.seed)
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

    print(
        f"[fit ] two-stage reject classifier "
        f"(Stage 1 = UDP-only, Stage 2 = UDP+QUIC, estimator={train_cfg.stage_estimator}) ..."
    )
    model = TwoStageRejectClassifier(
        stage1_features=STAGE1_INDICES,
        escalate_threshold=train_cfg.escalate_threshold,
        stage1_estimator=make_stage_estimator(
            train_cfg.stage_estimator, **train_cfg.stage1_params
        ),
        stage2_estimator=make_stage_estimator(
            train_cfg.stage_estimator, **train_cfg.stage2_params
        ),
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

    curve = model.coverage_reliability_curve(Xte, yte, train_cfg.reject_thresholds)
    escalation_rate = curve[0]["escalation_rate"]

    # Ablation: single-stage estimator on the full UDP+QUIC space (no hierarchy, no
    # reject), so the value of the two-stage design is measurable, not asserted.
    single = make_stage_estimator(
        train_cfg.stage_estimator, **train_cfg.stage2_params
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
        "escalate_threshold": train_cfg.escalate_threshold,
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

    artifact_dir = cfg.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "imputer": imputer,
            "model": model,
            "stage1_features": STAGE1_INDICES,
            "udp_features": list(UDP_FEATURES),
            "stage2_features": list(STAGE2_FEATURES),
            "classes": classes,
            # Positions (into load_sample() order) of the held-out TEST rows, so the
            # dashboard and any replay can evaluate on the exact same leakage-safe
            # split these metrics are measured on, not the shuffled full sample.
            "test_indices": [int(i) for i in te],
            # Per-feature decile edges + proportions of the imputed TRAIN matrix,
            # so scored windows can be PSI-checked against what the model actually
            # learned from (see drift.py).
            "drift_reference": reference_from_matrix(Xtr, list(STAGE2_FEATURES)),
        },
        artifact_dir / "flowsentry.joblib",
    )
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))
    print(f"[save] {artifact_dir / 'flowsentry.joblib'}")
    return metrics


if __name__ == "__main__":
    main()

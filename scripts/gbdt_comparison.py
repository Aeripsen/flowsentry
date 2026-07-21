"""Boosted trees against the shipped forest, head to head on the same split.
Run:  python scripts/gbdt_comparison.py   (needs pip install -e ".[gbdt]")

Writes artifacts/gbdt_comparison.json and prints everything it does on the way,
including every grid entry with its validation score, so nothing is searched
silently.

Design, leakage-safe end to end and deliberately parallel to how the repo's
other experiments are cut:

  * the held-out TEST split is carved exactly as train.py does (grouped, seed
    42) and is never touched by tuning or calibration;
  * a VALIDATION split is carved out of the remaining training connections
    (grouped, seed 43, same carve calibration_experiment.py uses); the grids
    fit on what is left and are scored on it by binary attack PR-AUC, the
    repo's headline metric;
  * each family's winner is then refit on the FULL training split, so its test
    numbers are comparable to the committed baseline, which also trained on the
    full training split;
  * the calibration question is answered on the validation-fit model (the
    refit model has no unseen data left to fit a calibrator on): the validation
    split is halved by connection (seed 44), isotonic and Platt fit on one half,
    the pick happens by ECE on the other, and the chosen map is scored on the
    untouched test split.

The GBDTs could take the NaNs natively, but they get the same median-imputed
matrices the forest gets: changing the model family and the missing-value
policy at once would leave a difference unattributable.

The baseline block is read from the committed artifacts (metrics.json,
calibration.json) at run time rather than copied by hand, so this file cannot
drift from them.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flowsentry.calibration import (  # noqa: E402
    brier_multiclass,
    brier_top_label,
    ece,
    mce,
)
from flowsentry.config import get_settings  # noqa: E402
from flowsentry.data import build_matrices, leakage_safe_split, load_sample  # noqa: E402
from flowsentry.evaluation import per_family  # noqa: E402
from flowsentry.gbdt import (  # noqa: E402
    LIGHTGBM_GRID,
    XGBOOST_GRID,
    binary_attack_pr_auc,
    fit_confidence_calibrators,
    make_lightgbm,
    make_xgboost,
    pick_calibrator,
    reject_curve,
    top_label,
)

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
OUT = ARTIFACTS / "gbdt_comparison.json"

FAMILIES = {
    "lightgbm": (make_lightgbm, LIGHTGBM_GRID),
    "xgboost": (make_xgboost, XGBOOST_GRID),
}


def evaluate(model: Any, Xte: np.ndarray, yte: np.ndarray, thresholds: list[float]) -> dict:
    """The same battery train.py and the report scripts run on the shipped
    model, so every row of the comparison table has a committed counterpart."""
    classes = [str(c) for c in model.classes_]
    proba = model.predict_proba(Xte)
    labels, conf = top_label(proba, classes)
    correct = (labels == yte).astype(float)

    is_benign = (yte == "benign").astype(int)
    p_benign = proba[:, classes.index("benign")]
    return {
        "binary_attack_pr_auc": round(binary_attack_pr_auc(yte, proba, classes), 4),
        "benign_pr_auc": round(float(average_precision_score(is_benign, p_benign)), 4),
        "accuracy_full_coverage": round(float(correct.mean()), 4),
        "macro_f1_full_coverage": round(float(f1_score(yte, labels, average="macro")), 4),
        "calibration_raw_confidence": {
            "mean_confidence": round(float(conf.mean()), 4),
            "ece": round(ece(conf, correct), 4),
            "mce": round(mce(conf, correct), 4),
            "brier_top_label": round(brier_top_label(conf, correct), 4),
            "brier_multiclass": round(brier_multiclass(proba, yte, classes), 4),
        },
        "per_family_full_coverage": per_family(yte, labels, classes),
        "reject_curve": reject_curve(conf, correct, thresholds),
    }


def main() -> dict:
    cfg = get_settings().training
    df = load_sample()
    X, y, groups = build_matrices(df)
    tr, te = leakage_safe_split(groups, test_size=cfg.test_size, seed=cfg.seed)

    # validation carve out of the training connections, grouped like everything else
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=43)
    sub_i, val_i = next(gss.split(np.zeros(len(tr)), groups=groups[tr]))
    sub, val = tr[sub_i], tr[val_i]
    assert not (set(groups[sub]) & set(groups[val]))
    assert not (set(groups[val]) & set(groups[te]))

    imputer = SimpleImputer(strategy="median").fit(X[tr])
    imputer_sub = SimpleImputer(strategy="median").fit(X[sub])
    Xte = imputer.transform(X[te])
    yte = y[te]
    print(f"[data] train={len(tr)} (fit={len(sub)} val={len(val)}) test={len(te)}")

    arms: dict[str, Any] = {}
    winners: dict[str, Any] = {}
    for name, (factory, grid) in FAMILIES.items():
        print(f"[grid] {name}: {len(grid)} configs, scored on validation by binary PR-AUC")
        rows = []
        best_score, best_params = -1.0, None
        for params in grid:
            t0 = time.perf_counter()
            model = factory(seed=cfg.seed, **params).fit(imputer_sub.transform(X[sub]), y[sub])
            proba_val = model.predict_proba(imputer_sub.transform(X[val]))
            score = binary_attack_pr_auc(y[val], proba_val, [str(c) for c in model.classes_])
            rows.append({**params, "val_binary_pr_auc": round(score, 4)})
            print(f"       {params} -> {score:.4f}  ({time.perf_counter() - t0:.1f}s)")
            if score > best_score:
                best_score, best_params = score, params
        print(f"[grid] {name} winner: {best_params} (val {best_score:.4f})")

        # refit the winner on the full training split so the test numbers are
        # comparable to the committed baseline, then run the full battery
        final = factory(seed=cfg.seed, **best_params).fit(imputer.transform(X[tr]), y[tr])
        test_eval = evaluate(final, Xte, yte, cfg.reject_thresholds)
        print(
            f"[test] {name}: binary PR-AUC {test_eval['binary_attack_pr_auc']} "
            f"accuracy {test_eval['accuracy_full_coverage']} "
            f"macro-F1 {test_eval['macro_f1_full_coverage']} "
            f"ece {test_eval['calibration_raw_confidence']['ece']}"
        )

        # calibration on the validation-fit model: fit on one half of the
        # validation connections, pick on the other, score on the test split
        val_model = factory(seed=cfg.seed, **best_params).fit(
            imputer_sub.transform(X[sub]), y[sub]
        )
        half = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=44)
        fit_i, pick_i = next(half.split(np.zeros(len(val)), groups=groups[val]))
        cal_fit, cal_pick = val[fit_i], val[pick_i]

        def conf_correct(model: Any, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            proba = model.predict_proba(imputer_sub.transform(X[idx]))
            labels, conf = top_label(proba, [str(c) for c in model.classes_])
            return conf, (labels == y[idx]).astype(float)

        calibrators = fit_confidence_calibrators(*conf_correct(val_model, cal_fit))
        chosen, pick_scores = pick_calibrator(calibrators, *conf_correct(val_model, cal_pick))
        conf_te, correct_te = conf_correct(val_model, te)
        conf_te_cal = calibrators[chosen](conf_te)
        calibration = {
            "note": (
                "measured on the validation-fit model, not the refit one; the refit "
                "model trained on every non-test connection, so no honest data was "
                "left to fit its calibrator on"
            ),
            "chosen": chosen,
            "pick_ece_on_held_out_half": pick_scores,
            "test_ece_raw": round(ece(conf_te, correct_te), 4),
            "test_ece_calibrated": round(ece(conf_te_cal, correct_te), 4),
            "test_brier_top_label_raw": round(brier_top_label(conf_te, correct_te), 4),
            "test_brier_top_label_calibrated": round(
                brier_top_label(conf_te_cal, correct_te), 4
            ),
        }
        print(
            f"[cal ] {name}: {chosen} picked "
            f"(test ece {calibration['test_ece_raw']} -> {calibration['test_ece_calibrated']})"
        )

        arms[name] = {
            "grid": rows,
            "winner_params": best_params,
            "winner_val_binary_pr_auc": round(best_score, 4),
            "test": test_eval,
            "confidence_calibration": calibration,
        }
        winners[name] = final

    baseline_metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    baseline_cal = json.loads((ARTIFACTS / "calibration.json").read_text())
    baseline = {
        "source": "artifacts/metrics.json + artifacts/calibration.json (committed, unchanged)",
        "model": "two-stage random forest with reject (the shipped model)",
        "binary_attack_pr_auc": baseline_metrics["binary_attack_detection_pr_auc"],
        "benign_pr_auc": baseline_metrics["benign_detection_pr_auc"],
        "accuracy_full_coverage": baseline_metrics["accuracy_full_coverage"],
        "macro_f1_full_coverage": baseline_metrics["macro_f1_full_coverage"],
        "ece_raw_confidence": baseline_cal["overall"]["ece"],
        "brier_top_label": baseline_cal["overall"]["brier_top_label"],
    }

    best_arm = max(arms, key=lambda a: arms[a]["test"]["binary_attack_pr_auc"])
    import lightgbm
    import xgboost

    report = {
        "what": (
            "LightGBM and XGBoost single-stage joint models against the shipped "
            "two-stage forest: same features, same grouped leakage-safe split, same "
            "metrics battery; grids printed, winners refit on the full training split"
        ),
        "config": {
            "test_size": cfg.test_size,
            "seed": cfg.seed,
            "validation_carve": "grouped, 25% of training connections, seed 43",
            "calibration_halves": "grouped halves of the validation carve, seed 44",
            "reject_thresholds": cfg.reject_thresholds,
        },
        "n_train": int(len(tr)),
        "n_fit": int(len(sub)),
        "n_validation": int(len(val)),
        "n_test": int(len(te)),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "xgboost": xgboost.__version__,
        },
        "baseline": baseline,
        "arms": arms,
        "best_arm_by_test_binary_pr_auc": best_arm,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[best] {best_arm}: {arms[best_arm]['test']['binary_attack_pr_auc']} "
          f"vs baseline {baseline['binary_attack_pr_auc']}")
    print(f"[save] {OUT}")
    return report


if __name__ == "__main__":
    main()

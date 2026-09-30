"""Log the model comparison to MLflow, one run per arm.
Run:  python scripts/track_arms.py   (needs pip install -e ".[gbdt,mlops]")

The comparison already exists as three committed JSON files written by three
scripts (metrics.json, hierarchy_benchmark.json, gbdt_comparison.json). This puts
the same arms side by side in a tracking store, each as its own run with its
params, its metrics, its reject curve and its fitted model, so the comparison can
be sorted and plotted in the MLflow UI instead of read across files.

The arms, all fit on the same grouped leakage-safe training split (seed 42) and
scored on the same held-out connections:

  random_forest_joint  the 200-tree forest on all 132 UDP+QUIC features answering
                       every flow. This is model.stage2_ of the hierarchy fit, which
                       is the same fitted forest hierarchy_benchmark.json calls
                       single_joint and metrics.json calls ablation_single_rf.
  two_stage_hierarchy  the shipped model: the 60-tree UDP-only forest, escalating
                       its low-confidence tail to the joint forest.
  xgboost_tuned        XGBoost with the grid winner recorded in
  lightgbm_tuned       gbdt_comparison.json (picked there on a grouped validation
                       carve by binary PR-AUC), refit on the full training split
                       exactly as that script does. The grid is not re-searched
                       here; the committed artifact is the record of the search.

Every run is cross-checked against the committed artifact that already reports
the arm. A tracking store that disagreed with the committed numbers would be worse
than no tracking store, so any mismatch fails the script (exit 1). Under a library
version other than the locked one a mismatch in the 4th decimal can be real
version drift, not a bug, and --allow-mismatch lets the runs log anyway, tagged.

After logging, the runs are read BACK from the store with mlflow.search_runs and
written to artifacts/mlflow_arms.json. Run ids are random and live only in the
store; that file holds what is deterministic, so CI can diff it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

from flowsentry import tracking  # noqa: E402
from flowsentry.calibration import ece  # noqa: E402
from flowsentry.config import get_settings  # noqa: E402
from flowsentry.data import (  # noqa: E402
    STAGE1_INDICES,
    build_matrices,
    leakage_safe_split,
    load_sample,
)
from flowsentry.gbdt import make_lightgbm, make_xgboost, reject_curve, top_label  # noqa: E402
from flowsentry.model import TwoStageRejectClassifier  # noqa: E402
from flowsentry.registry import make_stage_estimator  # noqa: E402

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
OUT = ARTIFACTS / "mlflow_arms.json"
EXPERIMENT = "flowsentry-arms"
# the metrics every arm reports, and the ones compared to the committed files
HEADLINE = ["binary_attack_pr_auc", "benign_pr_auc", "accuracy_full_coverage",
            "macro_f1_full_coverage", "ece_top_label"]


def score(proba: np.ndarray, classes: list[str], yte: np.ndarray, thresholds: list[float],
          labels: np.ndarray | None = None, conf: np.ndarray | None = None) -> dict[str, Any]:
    """The battery train.py and gbdt_comparison.py run, computed one way for all arms.
    The hierarchy passes its own labels and confidence (Stage 1's where Stage 1
    answered, Stage 2's where it escalated), which is what it serves."""
    if labels is None or conf is None:
        labels, conf = top_label(proba, classes)
    correct = (labels == yte).astype(float)
    is_attack = (yte != "benign").astype(int)
    p_benign = proba[:, classes.index("benign")]
    return {
        "binary_attack_pr_auc": round(float(average_precision_score(is_attack, 1.0 - p_benign)), 4),
        "benign_pr_auc": round(float(average_precision_score(1 - is_attack, p_benign)), 4),
        "accuracy_full_coverage": round(float(correct.mean()), 4),
        "macro_f1_full_coverage": round(float(f1_score(yte, labels, average="macro")), 4),
        "ece_top_label": round(ece(conf, correct), 4),
        "reject_curve": reject_curve(conf, correct, thresholds),
    }


def committed_reference() -> dict[str, dict[str, float]]:
    """What the committed artifacts already say about each arm, field by field.
    Only fields a committed file actually reports are checked; nothing is filled in."""
    m = json.loads((ARTIFACTS / "metrics.json").read_text())
    cal = json.loads((ARTIFACTS / "calibration.json").read_text())
    hb = json.loads((ARTIFACTS / "hierarchy_benchmark.json").read_text())["quality"]
    gb = json.loads((ARTIFACTS / "gbdt_comparison.json").read_text())["arms"]
    ref = {
        "two_stage_hierarchy": {
            "binary_attack_pr_auc": m["binary_attack_detection_pr_auc"],
            "benign_pr_auc": m["benign_detection_pr_auc"],
            "accuracy_full_coverage": m["accuracy_full_coverage"],
            "macro_f1_full_coverage": m["macro_f1_full_coverage"],
            "ece_top_label": cal["overall"]["ece"],
        },
        "random_forest_joint": {
            "binary_attack_pr_auc": hb["single_joint"]["binary_attack_detection_pr_auc"],
            "accuracy_full_coverage": m["ablation_single_rf"]["accuracy_full_coverage"],
            "macro_f1_full_coverage": m["ablation_single_rf"]["macro_f1_full_coverage"],
        },
    }
    for fam in ("xgboost", "lightgbm"):
        t = gb[fam]["test"]
        ref[f"{fam}_tuned"] = {
            "binary_attack_pr_auc": t["binary_attack_pr_auc"],
            "benign_pr_auc": t["benign_pr_auc"],
            "accuracy_full_coverage": t["accuracy_full_coverage"],
            "macro_f1_full_coverage": t["macro_f1_full_coverage"],
            "ece_top_label": t["calibration_raw_confidence"]["ece"],
        }
    return ref


def check(arm: str, live: dict[str, Any], ref: dict[str, dict[str, float]]) -> dict[str, Any]:
    rows = {}
    for k, want in ref.get(arm, {}).items():
        rows[k] = {"committed": want, "tracked": live[k], "match": live[k] == want,
                   "abs_diff": round(abs(live[k] - want), 4)}
    return {"fields": rows, "all_match": all(r["match"] for r in rows.values()),
            "max_abs_diff": max((r["abs_diff"] for r in rows.values()), default=0.0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="log even if a run disagrees with the committed artifacts")
    ap.add_argument("--out", type=Path, default=OUT,
                    help="where to write the read-back summary (default: the committed file)")
    args = ap.parse_args()

    import mlflow
    import mlflow.pyfunc
    import mlflow.sklearn

    settings = get_settings()
    cfg = settings.training
    gb = json.loads((ARTIFACTS / "gbdt_comparison.json").read_text())

    df = load_sample()
    X, y, groups = build_matrices(df)
    tr, te = leakage_safe_split(groups, test_size=cfg.test_size, seed=cfg.seed)
    imputer = SimpleImputer(strategy="median").fit(X[tr])
    Xtr, Xte = imputer.transform(X[tr]), imputer.transform(X[te])
    ytr, yte = y[tr], y[te]
    thresholds = cfg.reject_thresholds
    print(f"[data] train={len(tr)} test={len(te)} (grouped split, seed {cfg.seed})")

    arms: dict[str, dict[str, Any]] = {}

    # --- two-stage hierarchy (and its joint forest, the random_forest_joint arm)
    t0 = time.perf_counter()
    hier = TwoStageRejectClassifier(
        stage1_features=STAGE1_INDICES,
        escalate_threshold=cfg.escalate_threshold,
        stage1_estimator=make_stage_estimator(cfg.stage_estimator, **cfg.stage1_params),
        stage2_estimator=make_stage_estimator(cfg.stage_estimator, **cfg.stage2_params),
    ).fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0
    classes = [str(c) for c in hier.classes_]
    lab_h, conf_h, escalated, proba_h = hier._stage_predict(Xte)
    arms["two_stage_hierarchy"] = {
        "family": "random_forest x2 (stage 1 UDP-only, stage 2 UDP+QUIC)",
        "params": {"stage_estimator": cfg.stage_estimator,
                   "escalate_threshold": cfg.escalate_threshold,
                   **tracking.flatten(cfg.stage1_params, "stage1."),
                   **tracking.flatten(cfg.stage2_params, "stage2.")},
        "metrics": {**score(proba_h, classes, yte, thresholds, np.asarray(lab_h), conf_h),
                    "escalation_rate": round(float(np.mean(escalated)), 4)},
        "fit_seconds": fit_s,
        "model": Pipeline([("impute", imputer), ("model", hier)]),
        "trusted": tracking.SKOPS_TRUSTED_TYPES,
    }
    joint = hier.stage2_
    arms["random_forest_joint"] = {
        "family": "random_forest (single joint model, no hierarchy, no escalation)",
        "params": {"stage_estimator": cfg.stage_estimator,
                   **tracking.flatten(cfg.stage2_params)},
        "metrics": score(joint.predict_proba(Xte), [str(c) for c in joint.classes_],
                         yte, thresholds),
        "fit_seconds": None,  # fit inside the hierarchy above; not timed separately
        "model": Pipeline([("impute", imputer), ("model", joint)]),
        "trusted": ["numpy.dtype", "sklearn.tree._tree.Tree"],
    }

    # --- tuned boosters, winner params from the committed grid search
    for fam, factory in (("xgboost", make_xgboost), ("lightgbm", make_lightgbm)):
        params = gb["arms"][fam]["winner_params"]
        t0 = time.perf_counter()
        model = factory(seed=cfg.seed, **params).fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0
        arms[f"{fam}_tuned"] = {
            "family": fam,
            "params": {**params, "seed": cfg.seed,
                       "selected_by": "grouped validation carve (seed 43), binary PR-AUC",
                       "grid_size": len(gb["arms"][fam]["grid"])},
            "metrics": score(model.predict_proba(Xte), [str(c) for c in model.classes_],
                             yte, thresholds),
            "fit_seconds": fit_s,
            "model": Pipeline([("impute", imputer), ("model", model)]),
            "trusted": None,  # compiled booster state; see the log_model call below
        }

    ref = committed_reference()
    checks = {name: check(name, a["metrics"], ref) for name, a in arms.items()}
    for name, c in checks.items():
        bad = {k: f"{r['committed']} -> {r['tracked']}" for k, r in c["fields"].items()
               if not r["match"]}
        status = "OK" if not bad else f"MISMATCH {bad}"
        print(f"[check] {name}: {status}")
    all_match = all(c["all_match"] for c in checks.values())
    if not all_match and not args.allow_mismatch:
        print("[check] FAIL: a tracked arm disagrees with the committed artifacts. "
              "Under the locked environment this is a bug; rerun with --allow-mismatch "
              "only if you know it is library-version drift.")
        return 1

    # --- log, one run per arm
    tracking.use_experiment(EXPERIMENT)
    batch = uuid.uuid4().hex[:12]
    tags = tracking.standard_tags(settings.sample_path)
    for name, a in arms.items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags({**tags, "comparison_batch": batch, "arm": name,
                             "family": a["family"],
                             "matches_committed": str(checks[name]["all_match"]),
                             "shipped": str(name == "two_stage_hierarchy")})
            mlflow.log_params({"split.seed": cfg.seed, "split.test_size": cfg.test_size,
                               "split.kind": "GroupShuffleSplit on the UDP 4-tuple",
                               "imputer": "median, fit on train",
                               **{f"model.{k}": v for k, v in a["params"].items()}})
            mlflow.log_metrics({k: float(v) for k, v in a["metrics"].items()
                                if isinstance(v, int | float)})
            for row in a["metrics"]["reject_curve"]:
                step = int(round(row["threshold"] * 100))
                mlflow.log_metric("curve.coverage", row["coverage"], step=step)
                if row["reliability"] is not None:
                    mlflow.log_metric("curve.reliability", row["reliability"], step=step)
            if a["fit_seconds"] is not None:
                mlflow.log_metric("fit_seconds", round(a["fit_seconds"], 2))
            mlflow.log_dict(checks[name], "committed_crosscheck.json")
            if a["trusted"] is not None:
                info = mlflow.sklearn.log_model(
                    a["model"], name="model", input_example=Xte[:5],
                    pip_requirements=tracking.pip_requirements(),
                    code_paths=[str(tracking.REPO_ROOT / "src" / "flowsentry")],
                    skops_trusted_types=a["trusted"])
                expected = a["model"].predict(Xte)
            else:
                # skops cannot walk a compiled xgboost/lightgbm booster and the repo's
                # xgboost wrapper is not a full sklearn estimator, so these log as a
                # pyfunc (cloudpickle) returning the probability matrix. Load them only
                # from a store you wrote yourself.
                from flowsentry.mlflow_models import ImputedProbaModel

                imp, est = a["model"].steps[0][1], a["model"].steps[1][1]
                info = mlflow.pyfunc.log_model(
                    name="model", python_model=ImputedProbaModel(imp, est),
                    input_example=Xte[:5],
                    pip_requirements=tracking.pip_requirements([a["family"]]),
                    code_paths=[str(tracking.REPO_ROOT / "src" / "flowsentry")])
                expected = est.predict_proba(Xte)
            # round trip: load the logged model back from the store and require it to
            # reproduce the in-memory predictions on every held-out row
            loaded = mlflow.pyfunc.load_model(info.model_uri)
            got = np.asarray(loaded.predict(Xte))
            same = (bool(np.array_equal(got, expected)) if got.dtype == object
                    else bool(np.allclose(got, expected, rtol=0, atol=1e-9)))
            mlflow.set_tag("roundtrip_predictions_identical", str(same))
            if not same:
                print(f"[check] FAIL: {name} logged model does not reproduce its predictions")
                return 1
            print(f"[mlflow] {name}: run {run.info.run_id}")

    # --- read back from the store, not from the dicts above
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT],
                              filter_string=f"tags.comparison_batch = '{batch}'")
    readback: dict[str, Any] = {}
    for _, r in runs.sort_values("tags.arm").iterrows():
        readback[r["tags.arm"]] = {
            "family": r["tags.family"],
            "shipped": r["tags.shipped"] == "True",
            # sorted: search_runs returns param columns in no fixed order
            "params": {c.removeprefix("params.model."): r[c] for c in sorted(runs.columns)
                       if c.startswith("params.model.") and isinstance(r[c], str)},
            "metrics": {k: round(float(r[f"metrics.{k}"]), 4) for k in HEADLINE
                        + ["escalation_rate"] if f"metrics.{k}" in runs.columns
                        and r[f"metrics.{k}"] == r[f"metrics.{k}"]},
            "matches_committed": r["tags.matches_committed"] == "True",
            "max_abs_diff_vs_committed": checks[r["tags.arm"]]["max_abs_diff"],
            "logged_model_reproduces_predictions":
                r["tags.roundtrip_predictions_identical"] == "True",
        }
    best = max(readback, key=lambda k: readback[k]["metrics"]["binary_attack_pr_auc"])
    report = {
        "what": ("the flowsentry model comparison as MLflow runs, read back from the "
                 "tracking store; one run per arm in experiment " + EXPERIMENT),
        "how_to_view": "make track && mlflow ui --backend-store-uri sqlite:///mlflow.db",
        "split": {"seed": cfg.seed, "test_size": cfg.test_size,
                  "n_train": int(len(tr)), "n_test": int(len(te))},
        "sample_sha256": tags["data.sample_sha256"],
        "arms": readback,
        "best_by_binary_attack_pr_auc": best,
        "shipped": "two_stage_hierarchy",
        "why_the_best_is_not_shipped": (
            "see README and gbdt_comparison.json: the boosters rank flows better "
            "(PR-AUC) but lose on accuracy and macro-F1 at full coverage and on raw "
            "calibration of the confidence the reject knob thresholds"),
        "all_arms_match_committed_artifacts": all_match,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[save] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

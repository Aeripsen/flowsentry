"""Evidently data-drift + model-performance report on the repo's own sample.
Run:  python scripts/evidently_report.py   (needs pip install -e ".[mlops]")

Writes:
  reports/evidently_drift_performance.html   the Evidently report (open in a browser)
  artifacts/evidently_summary.json            the numbers it contains, plus the repo's
                                              own PSI on the same two windows

## What the two windows are, and why not a time split

A drift report compares a reference window with a current window. The honest
first choice is time (last week vs this week), and this sample cannot do it:
scripts/build_sample.py drops `timestamp` with the other identifier columns and
shuffles the rows, so the committed 25,615 flows carry no time field (the README
roadmap says the same about cross-day evaluation). Inventing an order would be
fabricating a time axis.

So the windows are the split the repo already trusts, cut exactly as train.py
cuts it:
  reference  the TRAINING connections (19,045 flows): what the model learned from
  current    the HELD-OUT connections (6,570 flows): 5-tuple connections the
             model never saw, which is the one axis of "new traffic" this sample
             really has

The question the report answers is therefore: do unseen connections look like
the training traffic (covariate shift across connections), and does the model
perform on them the way cross-validation on the training connections said it
would? It does NOT answer whether next month's traffic drifts. That needs the
sample rebuilt with timestamps kept.

## Where the predictions come from

  current    the shipped two-stage model (fit on all training connections, same
             config and seed as train.py) scoring the held-out connections. Its
             accuracy and macro-F1 are asserted equal to artifacts/metrics.json.
  reference  OUT-OF-FOLD predictions: 3-fold GroupKFold over the training
             connections, a fresh two-stage model per fold (imputer refit per
             fold), each fold scored by the model that did not see it. Scoring the
             training rows with the final model would compare a memorised window
             against a held-out one and every "performance drop" would be fake.

## Stattests (what Evidently chose, and how it differs from drift.py)

Evidently's defaults at these sizes (more than 1,000 rows per window): numeric
columns get the normed Wasserstein distance, drift if > 0.1; columns with few
distinct values are treated as categorical and get Jensen-Shannon distance, drift
if > 0.1. The dataset is flagged drifted if at least half the columns drift.
drift.py uses PSI on the training deciles with the 0.10 / 0.25 bands. They measure
different things (distance between distributions in units of the reference's
spread vs. log-ratio of bin proportions), so both are reported on the same windows
and their agreement is counted rather than assumed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flowsentry.config import get_settings  # noqa: E402
from flowsentry.data import (  # noqa: E402
    STAGE1_INDICES,
    STAGE2_FEATURES,
    build_matrices,
    leakage_safe_split,
    load_sample,
)
from flowsentry.drift import MAJOR, MODERATE, band, psi, reference_from_matrix  # noqa: E402
from flowsentry.model import TwoStageRejectClassifier  # noqa: E402
from flowsentry.registry import make_stage_estimator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "reports" / "evidently_drift_performance.html"
OUT = ROOT / "artifacts" / "evidently_summary.json"
N_FOLDS = 3
FEATURES = list(STAGE2_FEATURES)
# data.py: the sample keeps every flow of the rare attack families and caps the two
# dominant classes, so "rare" is every class except these two
DOMINANT = ("benign", "UDP-RAW")


def two_stage(cfg: Any) -> TwoStageRejectClassifier:
    return TwoStageRejectClassifier(
        stage1_features=STAGE1_INDICES,
        escalate_threshold=cfg.escalate_threshold,
        stage1_estimator=make_stage_estimator(cfg.stage_estimator, **cfg.stage1_params),
        stage2_estimator=make_stage_estimator(cfg.stage_estimator, **cfg.stage2_params),
    )


def frame(X: np.ndarray, y: np.ndarray, labels: np.ndarray, proba: np.ndarray,
          classes: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(X, columns=FEATURES)
    df["target"] = y.astype(str)
    df["prediction"] = np.asarray(labels).astype(str)
    # Evidently maps probability columns to classes by name, so each column is
    # named exactly as its class (no feature is named like a class)
    for j, c in enumerate(classes):
        df[c] = proba[:, j]
    return df


def metric_values(snapshot: Any) -> list[dict[str, Any]]:
    return list(snapshot.dict()["metrics"])


def classification_numbers(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    wanted = {"Accuracy": "accuracy", "Precision": "precision_macro",
              "Recall": "recall_macro", "F1Score": "f1_macro", "RocAuc": "roc_auc_ovr",
              "LogLoss": "log_loss", "RecallByLabel": "recall_by_label",
              "F1ByLabel": "f1_by_label"}
    out: dict[str, Any] = {}
    for m in metrics:
        kind = m["config"]["type"].rsplit(":", 1)[-1]
        if kind in wanted:
            v = m["value"]
            out[wanted[kind]] = (
                {k: round(float(x), 4) for k, x in sorted(v.items())}
                if isinstance(v, dict) else round(float(v), 4)
            )
    return out


def evidently_dataset_drift(cur_ds: Any, ref_ds: Any, columns: list[str]) -> dict[str, Any]:
    """Evidently's own dataset-level verdict: the default test DriftedColumnsCount
    attaches (share of drifted columns must stay below drift_share). Read from the
    test result, not recomputed here."""
    from evidently import Report
    from evidently.metrics import DriftedColumnsCount

    snap = Report([DriftedColumnsCount(columns=columns)], include_tests=True).run(
        current_data=cur_ds, reference_data=ref_ds)
    d = snap.dict()
    (test,) = [t for t in d["tests"]
               if t["metric_config"]["params"]["type"].endswith(":DriftedColumnsCount")]
    status = str(getattr(test["status"], "value", test["status"]))
    return {"dataset_drift": status == "FAIL", "evidently_test": test["name"],
            "evidently_test_status": status,
            "drift_share": float(test["metric_config"]["params"]["drift_share"]),
            "share": float(d["metrics"][0]["value"]["share"])}


def main() -> int:
    from evidently import DataDefinition, Dataset, MulticlassClassification, Report
    from evidently.presets import ClassificationPreset, DataDriftPreset

    cfg = get_settings().training
    df = load_sample()
    X, y, groups = build_matrices(df)
    tr, te = leakage_safe_split(groups, test_size=cfg.test_size, seed=cfg.seed)

    # current window: the shipped model on the held-out connections
    imputer = SimpleImputer(strategy="median").fit(X[tr])
    Xtr, Xte = imputer.transform(X[tr]), imputer.transform(X[te])
    print(f"[fit ] shipped two-stage model on {len(tr)} training flows ...")
    model = two_stage(cfg).fit(Xtr, y[tr])
    classes = [str(c) for c in model.classes_]
    lab_cur, _, _, proba_cur = model._stage_predict(Xte)

    committed = json.loads((ROOT / "artifacts" / "metrics.json").read_text())
    acc = round(float((np.asarray(lab_cur) == y[te]).mean()), 4)
    mf1 = round(float(f1_score(y[te], lab_cur, average="macro")), 4)
    if (acc, mf1) != (committed["accuracy_full_coverage"], committed["macro_f1_full_coverage"]):
        print(f"[check] FAIL: current-window model scores {acc}/{mf1}, metrics.json says "
              f"{committed['accuracy_full_coverage']}/{committed['macro_f1_full_coverage']}")
        return 1
    print(f"[check] current-window model matches metrics.json (accuracy {acc}, macro-F1 {mf1})")

    # reference window: out-of-fold predictions over the training connections
    lab_ref = np.empty(len(tr), dtype=object)
    proba_ref = np.zeros((len(tr), len(classes)))
    rare = [c for c in classes if c not in DOMINANT]
    fold_support: dict[str, list[int]] = {c: [] for c in rare}
    for k, (fit_i, val_i) in enumerate(
        GroupKFold(n_splits=N_FOLDS).split(np.zeros(len(tr)), groups=groups[tr])
    ):
        imp_k = SimpleImputer(strategy="median").fit(X[tr][fit_i])
        m_k = two_stage(cfg).fit(imp_k.transform(X[tr][fit_i]), y[tr][fit_i])
        for c in rare:
            fold_support[c].append(int(np.sum(y[tr][fit_i] == c)))
        lab_k, _, _, proba_k = m_k._stage_predict(imp_k.transform(X[tr][val_i]))
        lab_ref[val_i] = lab_k
        # a fold can in principle miss a rare class; map its columns by name
        for j, c in enumerate(m_k.classes_):
            proba_ref[val_i, classes.index(str(c))] = proba_k[:, j]
        print(f"[fold] {k + 1}/{N_FOLDS}: {len(val_i)} out-of-fold flows scored")

    ref_df = frame(Xtr, y[tr], lab_ref, proba_ref, classes)
    cur_df = frame(Xte, y[te], np.asarray(lab_cur), proba_cur, classes)

    definition = DataDefinition(
        numerical_columns=FEATURES,
        classification=[MulticlassClassification(
            target="target", prediction_labels="prediction",
            prediction_probas=list(classes),
        )],
    )
    ref_ds = Dataset.from_pandas(ref_df, data_definition=definition)
    cur_ds = Dataset.from_pandas(cur_df, data_definition=definition)

    print("[evidently] data drift over the 132 model features + classification quality ...")
    snapshot = Report(
        [DataDriftPreset(columns=FEATURES), ClassificationPreset()],
        metadata={"reference": "training connections, out-of-fold predictions",
                  "current": "held-out connections, shipped model"},
    ).run(current_data=cur_ds, reference_data=ref_ds)
    HTML.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(HTML))
    verdict = evidently_dataset_drift(cur_ds, ref_ds, FEATURES)

    # the same classification battery on the reference window alone, for the numbers
    ref_only = Report([ClassificationPreset()]).run(current_data=ref_ds)

    metrics = metric_values(snapshot)
    per_column: dict[str, dict[str, Any]] = {}
    drifted_count = drift_share = None
    for m in metrics:
        kind = m["config"]["type"].rsplit(":", 1)[-1]
        if kind == "DriftedColumnsCount":
            drifted_count, drift_share = m["value"]["count"], m["value"]["share"]
        elif kind == "ValueDrift":
            c = m["config"]
            per_column[c["column"]] = {
                "method": c["method"], "threshold": c["threshold"],
                "score": round(float(m["value"]), 4),
                "drifted": float(m["value"]) > float(c["threshold"]),
            }

    # the repo's own drift check on the same two windows
    psi_vals = psi(reference_from_matrix(Xtr, FEATURES), Xte, FEATURES)
    bands = {f: band(v) for f, v in psi_vals.items()}
    ev_drift = {f for f, r in per_column.items() if r["drifted"]}
    psi_flag = {f for f, b in bands.items() if b in (MODERATE, MAJOR)}
    methods: dict[str, int] = {}
    for r in per_column.values():
        methods[r["method"]] = methods.get(r["method"], 0) + 1
    top = sorted(per_column.items(), key=lambda kv: -kv[1]["score"])[:10]
    if drift_share is None or round(float(drift_share), 6) != round(verdict["share"], 6):
        print("[check] FAIL: the dataset-drift test saw a different share than the report")
        return 1
    fold_all = [n for v in fold_support.values() for n in v]
    full_train = {c: int(np.sum(y[tr] == c)) for c in rare}

    summary = {
        "what": ("Evidently data drift + classification quality, reference = training "
                 "connections (out-of-fold predictions), current = held-out connections "
                 "(shipped model). No time axis: the sample has no timestamp column."),
        "html_report": HTML.relative_to(ROOT).as_posix(),
        "evidently_version": __import__("evidently").__version__,
        "n_reference": int(len(tr)),
        "n_current": int(len(te)),
        "n_features_checked": len(FEATURES),
        "reference_predictions": f"{N_FOLDS}-fold GroupKFold out-of-fold over training connections",
        "data_drift": {
            "drifted_columns": int(drifted_count) if drifted_count is not None else None,
            "drifted_share": round(float(drift_share), 4) if drift_share is not None else None,
            "dataset_drift": verdict["dataset_drift"],
            "dataset_drift_source": (
                f"Evidently's own test on DriftedColumnsCount: \"{verdict['evidently_test']}\" "
                f"-> {verdict['evidently_test_status']} (FAIL means dataset drift)"),
            "drift_share_threshold": verdict["drift_share"],
            "methods_used": dict(sorted(methods.items())),
            "top_10_by_score": {f: r for f, r in top},
        },
        "repo_psi_same_windows": {
            "moderate_or_major": len(psi_flag),
            "major": sum(1 for b in bands.values() if b == MAJOR),
            "max_psi": round(max(psi_vals.values()), 4),
            "agreement": {
                "both_flag": len(ev_drift & psi_flag),
                "evidently_only": sorted(ev_drift - psi_flag),
                "psi_only": sorted(psi_flag - ev_drift),
                "neither": len(set(FEATURES) - ev_drift - psi_flag),
            },
        },
        "rare_family_training_support": {
            "why": ("each out-of-fold model trains on the other two folds of the training "
                    "connections; these are the rare-family flow counts it sees, next to "
                    "what the shipped model trains on"),
            "families": rare,
            "per_fold_model": fold_support,
            "per_fold_model_range": [min(fold_all), max(fold_all)],
            "shipped_model": full_train,
            "shipped_model_range": [min(full_train.values()), max(full_train.values())],
            "held_out_test_support": {c: int(np.sum(y[te] == c)) for c in rare},
        },
        "performance": {
            "reference_out_of_fold": classification_numbers(metric_values(ref_only)),
            "current_held_out": classification_numbers(metrics),
            "current_matches_metrics_json": True,
        },
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n")
    d = summary["data_drift"]
    ref_p = summary["performance"]["reference_out_of_fold"]
    cur_p = summary["performance"]["current_held_out"]
    print(f"[drift] Evidently: {d['drifted_columns']}/{len(FEATURES)} columns drifted "
          f"(dataset drift: {d['dataset_drift']}); PSI moderate+: "
          f"{summary['repo_psi_same_windows']['moderate_or_major']}")
    print(f"[perf ] accuracy ref(OOF) {ref_p.get('accuracy')} -> current "
          f"{cur_p.get('accuracy')}; macro-F1 {ref_p.get('f1_macro')} -> {cur_p.get('f1_macro')}")
    print(f"[save] {HTML}\n[save] {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

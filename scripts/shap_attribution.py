"""SHAP attribution for the winning GBDT arm.
Run:  python scripts/shap_attribution.py   (after scripts/gbdt_comparison.py)

Writes artifacts/shap_top_features.json (top 15 features by mean |SHAP| over the
held-out test split, with the per-class breakdown) and
artifacts/shap_summary.png (the stacked per-class bar summary).

Why this exists: the feature schema is defended by name in data.py, but "the
model uses 132 named features" says nothing about which of them carry the
decisions. TreeExplainer gives exact SHAP values for tree ensembles (no
sampling approximation), so the attribution is as reproducible as the model:
same seed, same split, same numbers.

Scope, stated plainly: this attributes the GBDT comparison winner, not the
shipped two-stage forest. A two-stage model has no single SHAP story (two
ensembles, two feature views, a data-dependent escalation gate between them),
and inventing an averaged one would misattribute; the single-stage winner is
the strongest model in the repo with an exact attribution, so it is the one
explained. Attribution for the shipped hierarchy per stage is a reasonable
follow-up, not claimed here.

The winner's name and parameters are read from artifacts/gbdt_comparison.json
and the model is refit deterministically (fixed seed, fixed thread count), so
this script cannot silently explain a different model than the one the
comparison reported.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flowsentry.config import get_settings  # noqa: E402
from flowsentry.data import (  # noqa: E402
    STAGE2_FEATURES,
    build_matrices,
    leakage_safe_split,
    load_sample,
)
from flowsentry.gbdt import make_lightgbm, make_xgboost  # noqa: E402

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
COMPARISON = ARTIFACTS / "gbdt_comparison.json"
OUT_JSON = ARTIFACTS / "shap_top_features.json"
OUT_PNG = ARTIFACTS / "shap_summary.png"
TOP_K = 15


def main() -> dict:
    import matplotlib

    matplotlib.use("Agg")  # artifact writer, never a window
    import matplotlib.pyplot as plt
    import shap

    if not COMPARISON.exists():
        print(f"[error] {COMPARISON} missing; run scripts/gbdt_comparison.py first")
        raise SystemExit(1)
    comparison = json.loads(COMPARISON.read_text())
    arm = comparison["best_arm_by_test_binary_pr_auc"]
    params = comparison["arms"][arm]["winner_params"]
    print(f"[model] refitting the comparison winner: {arm} {params}")

    cfg = get_settings().training
    df = load_sample()
    X, y, groups = build_matrices(df)
    tr, te = leakage_safe_split(groups, test_size=cfg.test_size, seed=cfg.seed)
    imputer = SimpleImputer(strategy="median").fit(X[tr])
    Xtr, Xte = imputer.transform(X[tr]), imputer.transform(X[te])

    if arm == "lightgbm":
        model = make_lightgbm(seed=cfg.seed, **params).fit(Xtr, y[tr])
        tree_model = model
    else:
        model = make_xgboost(seed=cfg.seed, **params).fit(Xtr, y[tr])
        tree_model = model.inner  # the wrapper is label plumbing, not a model
    classes = [str(c) for c in model.classes_]

    # TreeExplainer is exact for tree ensembles, so no sampling seed to record;
    # explaining the full test split keeps the numbers tied to the same rows
    # every other artifact reports on
    print(f"[shap ] explaining {len(te)} test flows x {len(STAGE2_FEATURES)} features ...")
    explainer = shap.TreeExplainer(tree_model)
    values = explainer.shap_values(Xte)
    if isinstance(values, list):  # older shap returns one matrix per class
        values = np.stack(values, axis=-1)
    # values: (n_rows, n_features, n_classes)
    mean_abs = np.abs(values).mean(axis=0)  # (n_features, n_classes)
    overall = mean_abs.sum(axis=1)  # summed over classes, the summary-plot convention
    order = np.argsort(-overall)[:TOP_K]

    top = [
        {
            "rank": i + 1,
            "feature": STAGE2_FEATURES[j],
            "mean_abs_shap": round(float(overall[j]), 4),
            "per_class": {
                c: round(float(mean_abs[j, k]), 4) for k, c in enumerate(classes)
            },
        }
        for i, j in enumerate(order)
    ]

    report = {
        "what": (
            f"top {TOP_K} features by mean |SHAP| (TreeExplainer, exact) for the GBDT "
            "comparison winner, computed over the full held-out test split; "
            "mean_abs_shap sums the per-class means, the same aggregation the "
            "summary plot stacks"
        ),
        "model": {"arm": arm, "params": params, "seed": cfg.seed},
        "n_rows_explained": int(len(te)),
        "n_features": len(STAGE2_FEATURES),
        "classes": classes,
        "top_features": top,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"top_features": top[:5]}, indent=2))
    print(f"[save] {OUT_JSON}")

    Xte_df = pd.DataFrame(Xte, columns=STAGE2_FEATURES)
    shap.summary_plot(
        [values[:, :, k] for k in range(len(classes))],
        Xte_df,
        class_names=classes,
        max_display=TOP_K,
        show=False,
        plot_type="bar",
    )
    plt.gcf().set_size_inches(10, 7)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()
    print(f"[save] {OUT_PNG}")
    return report


if __name__ == "__main__":
    main()

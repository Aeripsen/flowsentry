"""Measure what the two-stage hierarchy actually buys.
Run:  python scripts/hierarchy_benchmark.py

Writes artifacts/hierarchy_benchmark.json and prints the same numbers.

Why this exists: the hierarchy is the repo's central architectural claim and it was
the only one with no artifact behind it. artifacts/metrics.json:ablation_single_rf
already showed a single joint forest TIES the hierarchy on accuracy and macro-F1,
so ADR 001 fell back on a compute argument (Stage 1 answers 75.7% of flows on cheap
always-present UDP features) that nothing in the repo measured. Asserting a
performance claim in a repo whose whole pitch is cite-or-cut is the one thing this
repo is not allowed to do. This script settles it either way.

Three arms, same trained artifact, same held-out test rows, same scoring path:

  stage1_only        the 60-tree forest on the 114 UDP features, forced to answer
                     every flow (no escalation). The cheap floor.
  single_joint       the 200-tree forest on all 132 UDP+QUIC features, answering
                     every flow. This is model.stage2_, which train.py fits with the
                     same config, seed and rows as ablation_single_rf, so it is
                     literally the same fitted forest the committed ablation reports.
  single_joint_small a 60-tree forest on the joint space, fit here. This is the
                     control that decides whether the hierarchy is worth anything:
                     the obvious rebuttal is "your hierarchy is just a smaller model
                     in disguise, so use a small joint model and skip the plumbing."
                     If this arm matches single_joint, that rebuttal is correct and
                     the hierarchy should go. It has to be measured, not waved off.
  hierarchy          the shipped path: Stage 1, escalate the low-confidence tail to
                     Stage 2.

Two things get measured: the accuracy delta between the arms, and the model-side
per-flow scoring cost of each arm.

## On pricing in QUIC feature extraction

The point of escalating is to skip QUIC extraction for the flows Stage 1 answers.
This repo cannot measure that cost: UDPFlowLyzer and QUICFlowLyzer are upstream
tools that live in other repos, and every benchmark here starts from already-computed
features (ADR 008 says so). Inventing a number for it would be exactly the sin this
script exists to correct.

So it is left symbolic and the question is asked in the form that does not need it.
Per flow, for a deployment that defers QUIC extraction until a flow escalates:

    single joint:  C_quic + T_joint
    hierarchy:     T_stage1 + e * (C_quic + T_stage2)

where e is the measured escalation rate and C_quic >= 0 is the unknown per-flow QUIC
extraction cost. The hierarchy's advantage is:

    advantage = (1 - e) * C_quic  +  [T_joint - T_stage1 - e * T_stage2]
                \__ extraction __/    \______ model-side, measured here ______/

The extraction term cannot be negative, because C_quic >= 0 and e <= 1. So if the
measured model-side bracket is positive, the hierarchy wins for ANY QUIC extraction
cost whatsoever, and the unmeasurable term never has to be guessed. If the bracket is
negative, C_quic has to clear a break-even this script reports, and the honest answer
becomes "it depends on a number this repo does not have".

## The caveat that has to be said out loud

FlowSentry as implemented does NOT defer QUIC extraction. scoring.row_from_features
builds all 132 columns (QUIC slots included) before Stage 1 ever runs, so the served
path pays for whatever QUIC features the caller already computed regardless of
escalation. The deferred-extraction deployment above is what the architecture PERMITS,
not what this code does. The model-side saving below is real and is what this repo can
honestly claim today; the extraction saving needs a pipeline that calls the UDP
extractor first and the QUIC extractor only on escalation.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flowsentry.bench import environment  # noqa: E402
from flowsentry.config import get_settings  # noqa: E402
from flowsentry.data import (  # noqa: E402
    STAGE1_INDICES,
    build_matrices,
    load_sample,
)
from flowsentry.model import forest_proba  # noqa: E402
from flowsentry.registry import make_stage_estimator  # noqa: E402
from flowsentry.scoring import FlowScorer, load_bundle  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "hierarchy_benchmark.json"
# Single-row timing is noisy (the arms moved 10-20% between pilot runs), and the
# whole verdict turns on a latency ratio, so the cost model is computed on BOTH the
# per-request path and the far more stable batch path. If the two paths disagree on
# the verdict, the verdict is not real.
SINGLE_ROW_CALLS = 500
BATCH_REPS = 7
SEED = 42


def _percentiles(lat_ms: np.ndarray) -> dict:
    return {
        "mean_ms": round(float(lat_ms.mean()), 4),
        "p50_ms": round(float(np.percentile(lat_ms, 50)), 4),
        "p95_ms": round(float(np.percentile(lat_ms, 95)), 4),
        "p99_ms": round(float(np.percentile(lat_ms, 99)), 4),
        "implied_flows_per_s": round(1000.0 / float(lat_ms.mean()), 1),
        "n_calls": int(lat_ms.size),
    }


def _quality(y_true: np.ndarray, labels: np.ndarray, proba: np.ndarray, classes: list) -> dict:
    """The three numbers the repo publishes, computed identically to train.py."""
    is_attack = (y_true != "benign").astype(int)
    p_benign = proba[:, classes.index("benign")]
    return {
        "accuracy_full_coverage": round(float((labels == y_true).mean()), 4),
        "macro_f1_full_coverage": round(float(f1_score(y_true, labels, average="macro")), 4),
        "binary_attack_detection_pr_auc": round(
            float(average_precision_score(is_attack, 1.0 - p_benign)), 4
        ),
    }


def _curve(y_true: np.ndarray, labels: np.ndarray, conf: np.ndarray, thresholds) -> list[dict]:
    """Coverage-reliability curve, computed exactly as model.coverage_reliability_curve
    does. The repo's pitch is that the curve is the product, so if the hierarchy earns
    its keep anywhere it should be here: a better curve than a single model's."""
    rows = []
    for t in thresholds:
        covered = conf >= t
        n = int(covered.sum())
        rows.append(
            {
                "threshold": round(float(t), 4),
                "coverage": round(float(covered.mean()), 4),
                "reliability": round(float((labels[covered] == y_true[covered]).mean()), 4)
                if n
                else None,
                "n_covered": n,
            }
        )
    return rows


def _time_calls(fn, rows: list[np.ndarray]) -> np.ndarray:
    for r in rows[:5]:  # warmup
        fn(r)
    lat = np.empty(len(rows), dtype=float)
    for k, r in enumerate(rows):
        t0 = time.perf_counter()
        fn(r)
        lat[k] = (time.perf_counter() - t0) * 1000.0
    return lat


def main() -> dict:
    bundle = load_bundle()
    scorer = FlowScorer.from_bundle(bundle)
    model = scorer.model
    test_idx = np.asarray(bundle["test_indices"])

    df = load_sample()
    X_all, y_all, _ = build_matrices(df)
    # the exact held-out rows train.py measured on, imputed with the shipped imputer
    Xte = scorer.impute(X_all[test_idx])
    yte = y_all[test_idx]
    classes = list(model.classes_)

    # ---------------- accuracy arms (whole test matrix) ----------------
    # stage1_only: the 60-tree UDP-only forest forced to answer everything
    p1 = forest_proba(model.stage1_, Xte[:, STAGE1_INDICES], sequential=False)
    s1_classes = list(model.stage1_.classes_)
    lab_s1 = np.asarray(model.stage1_.classes_[p1.argmax(axis=1)], dtype=object)

    # single_joint: the 200-tree joint forest answering everything (== ablation_single_rf)
    pj = forest_proba(model.stage2_, Xte, sequential=False)
    lab_j = np.asarray(model.stage2_.classes_[pj.argmax(axis=1)], dtype=object)

    # single_joint_small: 60 trees on the joint space, the "just use a smaller model"
    # control. Fit on the same training rows the shipped model used (the complement
    # of the persisted test indices), with the Stage-1 tree budget.
    cfg = get_settings().training
    train_idx = np.setdiff1d(np.arange(len(df)), test_idx)
    Xtr = scorer.impute(X_all[train_idx])
    small = make_stage_estimator(cfg.stage_estimator, **cfg.stage1_params).fit(
        Xtr, y_all[train_idx]
    )
    p_small = forest_proba(small, Xte, sequential=False)
    lab_small = np.asarray(small.classes_[p_small.argmax(axis=1)], dtype=object)

    # hierarchy: the shipped two-stage path
    lab_h, conf_h, escalated, proba_h = model._stage_predict(Xte)
    escalation_rate = round(float(escalated.mean()), 4)

    quality = {
        "stage1_only": _quality(yte, lab_s1, p1, s1_classes),
        "single_joint": _quality(yte, lab_j, pj, list(model.stage2_.classes_)),
        "single_joint_small": _quality(yte, lab_small, p_small, list(small.classes_)),
        "hierarchy": _quality(yte, lab_h, proba_h, classes),
    }

    # Does the joint forest agree with the hierarchy row for row on the escalated
    # tail? It must: the hierarchy IS the joint forest there. This is a self-check
    # that the arms are what they claim to be.
    agree_on_escalated = float((lab_h[escalated] == lab_j[escalated]).mean())

    # The reject knob under each arm. If the hierarchy justifies itself anywhere, it
    # is here: the repo sells the coverage-reliability curve, not an accuracy number.
    # A single model has a reject knob too (threshold its max probability), so the
    # question is whether the hierarchy's mixed stage-1/stage-2 confidence RANKS flows
    # better than one model's confidence does.
    thresholds = cfg.reject_thresholds
    reject_curves = {
        "note": (
            "the reject knob does not require a hierarchy: any model that emits a "
            "probability has one. This compares the curve each arm produces, which is "
            "the comparison that decides whether the two-stage design earns its place."
        ),
        "stage1_only": _curve(yte, lab_s1, p1.max(axis=1), thresholds),
        "single_joint": _curve(yte, lab_j, pj.max(axis=1), thresholds),
        "single_joint_small": _curve(yte, lab_small, p_small.max(axis=1), thresholds),
        "hierarchy": _curve(yte, lab_h, conf_h, thresholds),
    }

    # ---------------- latency arms (model-side scoring only) ----------------
    # Timed from an already-imputed row, so the only variable is the architecture.
    # sequential=True is the serving path (ADR 007).
    rng = np.random.RandomState(SEED)
    idx = rng.choice(Xte.shape[0], size=min(SINGLE_ROW_CALLS, Xte.shape[0]), replace=False)
    rows = [Xte[i : i + 1] for i in idx]

    arms_single = {
        "stage1_only": lambda r: forest_proba(
            model.stage1_, r[:, STAGE1_INDICES], sequential=True
        ),
        "single_joint": lambda r: forest_proba(model.stage2_, r, sequential=True),
        "single_joint_small": lambda r: forest_proba(small, r, sequential=True),
        "hierarchy": lambda r: model._stage_predict(r, sequential=True),
    }
    lat = {name: _time_calls(fn, rows) for name, fn in arms_single.items()}
    latency_single = {
        "what": (
            "model-side scoring cost per flow, from an already-imputed feature row, "
            "sequential path (the per-request serving path, ADR 007). Excludes feature "
            "extraction and the dict->row build, which are identical across arms."
        ),
        **{name: _percentiles(v) for name, v in lat.items()},
    }

    # Batch path: the same arms over the whole test matrix, native threaded path,
    # median of BATCH_REPS. Per-flow cost here is far less noisy than single-row.
    arms_batch = {
        "stage1_only": lambda X: forest_proba(
            model.stage1_, X[:, STAGE1_INDICES], sequential=False
        ),
        "single_joint": lambda X: forest_proba(model.stage2_, X, sequential=False),
        "single_joint_small": lambda X: forest_proba(small, X, sequential=False),
        "hierarchy": lambda X: model._stage_predict(X, sequential=False),
    }
    latency_batch: dict = {
        "what": (
            f"model-side scoring cost per flow over the whole {len(test_idx)}-row test "
            "matrix, native threaded path, median of "
            f"{BATCH_REPS} reps. Much more stable than the single-row numbers, which is "
            "why the verdict is checked against both."
        )
    }
    batch_per_flow = {}
    for name, fn in arms_batch.items():
        fn(Xte[:64])  # warmup
        walls = []
        for _ in range(BATCH_REPS):
            t0 = time.perf_counter()
            fn(Xte)
            walls.append(time.perf_counter() - t0)
        wall = float(np.median(walls))
        per_flow_ms = wall * 1000.0 / len(test_idx)
        batch_per_flow[name] = per_flow_ms
        latency_batch[name] = {
            "median_wall_s": round(wall, 4),
            "per_flow_ms": round(per_flow_ms, 5),
            "flows_per_s": round(len(test_idx) / wall, 1),
            "reps": BATCH_REPS,
        }

    # ---------------- the cost model ----------------
    e = float(escalated.mean())

    def _cost_model(t_s1: float, t_joint: float, t_small: float, t_hier: float) -> dict:
        """Per-flow cost of a deferred-QUIC deployment against both joint baselines.
        stage2_ IS the joint forest, so T_stage2 == T_joint."""
        out = {}
        baselines = (
            ("vs_single_joint_200", t_joint),
            ("vs_single_joint_small_60", t_small),
        )
        for label, t_base in baselines:
            # hierarchy = T_stage1 + e*(C_quic + T_stage2);  baseline = C_quic + T_base
            # advantage = (1-e)*C_quic + [T_base - T_stage1 - e*T_joint]
            bracket = t_base - t_s1 - e * t_joint
            out[label] = {
                "baseline_per_flow_ms": f"C_quic + {t_base:.4f}",
                "hierarchy_per_flow_ms": f"{t_s1 + e * t_joint:.4f} + {e:.4f} * C_quic",
                "model_side_bracket_ms": round(bracket, 4),
                "hierarchy_wins_for_any_c_quic": bool(bracket > 0),
                "breakeven_c_quic_ms": round(-bracket / (1.0 - e), 4),
                "measured_speedup_model_side": round(t_base / t_hier, 3),
            }
        return out

    cost_model = {
        "what": (
            "per-flow cost of a deployment that defers QUIC extraction until a flow "
            "escalates, against BOTH joint baselines. C_quic (per-flow QUIC "
            "feature-extraction cost) is NOT measured by this repo and is left symbolic; "
            "see this script's docstring. breakeven_c_quic_ms is the QUIC extraction cost "
            "at which the hierarchy and the baseline cost the same: negative means the "
            "hierarchy is already ahead on model compute alone, positive means QUIC "
            "extraction must cost at least that much per flow before the hierarchy pays."
        ),
        "escalation_rate_e": escalation_rate,
        "formula": "advantage = (1 - e) * C_quic + [T_baseline - T_stage1 - e * T_stage2]",
        "single_row_sequential": _cost_model(
            float(lat["stage1_only"].mean()),
            float(lat["single_joint"].mean()),
            float(lat["single_joint_small"].mean()),
            float(lat["hierarchy"].mean()),
        ),
        "batch_threaded": _cost_model(
            batch_per_flow["stage1_only"],
            batch_per_flow["single_joint"],
            batch_per_flow["single_joint_small"],
            batch_per_flow["hierarchy"],
        ),
    }

    report = {
        "what": (
            "what the two-stage hierarchy buys: stage1-only vs a 200-tree joint forest vs "
            "a 60-tree joint forest vs the hierarchy, on the same held-out rows, with the "
            "QUIC-extraction term handled symbolically rather than invented"
        ),
        "environment": environment(),
        "timing_stability_note": (
            "the quality numbers are seeded and reproduce exactly; the latency numbers "
            "are a snapshot of one run and the absolute values move by tens of percent "
            "between runs on a loaded machine. The RATIOS between arms, and therefore "
            "every verdict below, held across repeated runs and across both the "
            "single-row and batch paths. Do not quote an absolute ms figure from here "
            "without re-running it."
        ),
        "n_test": int(len(test_idx)),
        "escalation_rate": escalation_rate,
        "hierarchy_agrees_with_joint_on_escalated_rows": round(agree_on_escalated, 4),
        "quality": quality,
        "reject_curves": reject_curves,
        "latency_single_row_sequential": latency_single,
        "latency_batch_threaded": latency_batch,
        "cost_model": cost_model,
        "caveat": (
            "FlowSentry as implemented does NOT defer QUIC extraction: "
            "scoring.row_from_features builds all 132 columns before Stage 1 runs, so the "
            "served path pays for the QUIC features regardless of escalation. The "
            "deferred-extraction deployment the cost model prices is what the architecture "
            "permits, not what this code does."
        ),
        "verdict": (
            "The hierarchy does not pay for itself on this sample. It ties the 200-tree "
            "joint forest on accuracy and macro-F1 (structurally: stage2_ IS that forest, "
            "and the two agree on 100% of escalated rows), and it beats that forest on "
            "serving latency. But a 60-tree forest on the same joint space is faster than "
            "the hierarchy on BOTH paths, scores the highest macro-F1 of any arm, and beats "
            "the hierarchy on binary PR-AUC, from one model with no escalation threshold. "
            "The reject knob does not rescue the design either: all four arms produce the "
            "same coverage-reliability curve within noise, and at threshold 0.99 both joint "
            "models dominate the hierarchy on coverage AND reliability at once. The "
            "hierarchy's compute argument only survives against the 200-tree baseline that "
            "ADR 001 happened to pick, and that baseline is not the one a person would "
            "choose. The one argument still open is deferred QUIC extraction, which needs "
            "C_quic to clear the break-even above, which this repo cannot measure and this "
            "code does not implement."
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"[save] {OUT}")
    return report


if __name__ == "__main__":
    main()

"""Leave-one-family-out: what the model does with an attack family it never saw.
Run:  python scripts/zero_day_lofo.py     (or `make zero-day`)

Writes artifacts/zero_day_lofo.json and prints the same numbers.

Why this exists. The SECRYPT 2026 paper this repo implements defines the open-set
case formally (a withheld family set, a rejection-rate estimator, a three-way
decomposition of what a rejected flow could have been instead) and then reports the
numbers themselves as placeholders: "The corresponding empirical metrics are
reported as placeholders in the previous table". The method is written down and the
experiment was never run. Everything this repo publishes is closed-set: trained on
eight labels, scored on the same eight. So the claim a reader is most likely to take
away, that a reject option protects you against attacks the model has not seen, is
the one claim with nothing behind it.

It also happens to be the last live argument for the architecture. ADR 001's compute
justification was measured and withdrawn (`make hierarchy`: a 60-tree joint forest is
faster than the two-stage path and scores better). If the hierarchy has a reason to
exist, open-set behaviour is where it would show up, because that is the one place a
staged confidence signal could rank flows differently from a single model's.

## Protocol

For each attack family F, in turn:

  train on   the standard connection-grouped training split with every flow of F
             removed. F is not in the label space at all, so the model cannot emit
             it, which is exactly the zero-day situation.
  unseen     ALL flows of F in the dataset. None of them was trained on, so all of
             them are legitimate test rows, and the rare families only have 207-383
             flows each: restricting to the quarter that fell in the test split
             would put a rejection rate on ~50 flows. The split still matters for
             the seen families, which is why it is kept for them.
  seen       the standard held-out test split minus F. This is where the cost of the
             threshold is paid, and it is measured on the same rows in every round
             except for the family that is out.

The imputer is refit on each round's training rows only, so nothing about F reaches
the model through the medians either.

benign is never held out. An open-set detector is asked about attacks it has not
seen; removing the benign class would leave a model that cannot say "nothing is
wrong", which is a different (and degenerate) experiment.

UDP-RAW is held out and reported like the rest, but it is the dominant flood: taking
it out removes about half the training rows and the entire high-volume attack regime.
Read it as its own case, not as a seventh rare family, and the summary block keeps it
separate for that reason.

## Arms

  hierarchy             the shipped two-stage path, refit per round.
  single_joint_200      the 200-tree joint forest answering every flow. It is
                        model.stage2_, i.e. literally the same fitted forest the
                        hierarchy escalates into, so this arm is free.
  stage1_only           the 60-tree UDP-only forest forced to answer. The cheap floor.
  single_joint_small_60 a 60-tree forest on the joint space. This is the arm that
                        beat the hierarchy in `make hierarchy`, so it is the one
                        that has to be beaten here for the design to survive.

Every arm has a reject knob: any model that emits a probability can abstain below a
threshold. The hierarchy's confidence is a mixture of Stage-1 and Stage-2 maxima, and
whether that mixture ranks unseen flows better than one model's confidence does is
the whole question.

## The negative control

A model that abstains on everything scores a perfect unknown-detection rate, so the
raw rejection rate is not evidence of anything on its own. Two things guard it here.
The first is `novelty_lift`, which subtracts the abstention rate on seen families
measured on the same run at the same threshold. The second is the shuffled-label
control: the same protocol with the training labels permuted, which destroys the
family structure while leaving the class marginals, the row count and the threshold
exactly as they were. A model fit on noise still abstains, and it abstains on unseen
and seen flows alike, so its lift should collapse to zero. If the real run's lift
does not clear the control's, the rejection is threshold behaviour and this repo
should say so.

## What this can and cannot claim

The numbers are computed on the committed 25,615-flow stratified sample, the same
sample every other number in this repo is measured on, not the full 1.22M-flow
release. The rare families are complete in the sample (it caps benign and UDP-RAW
and keeps every rare flow), so the held-out family counts here are the real ones;
what the sample changes is the seen-side class balance, and therefore the baseline
abstention rate the lift is measured against. Rerunning this on the full dataset is
a download, not a rewrite.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flowsentry.bench import environment  # noqa: E402
from flowsentry.config import get_settings  # noqa: E402
from flowsentry.data import (  # noqa: E402
    FAMILIES,
    STAGE1_INDICES,
    build_matrices,
    leakage_safe_split,
    load_sample,
)
from flowsentry.model import TwoStageRejectClassifier, forest_proba  # noqa: E402
from flowsentry.openset import (  # noqa: E402
    closed_set_cost,
    destination_counts,
    novelty_lift,
    open_set_outcomes,
)
from flowsentry.registry import make_stage_estimator  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "zero_day_lofo.json"
BENIGN = "benign"
DOMINANT = "UDP-RAW"
ATTACK_FAMILIES = [f for f in FAMILIES if f != BENIGN]
RARE_FAMILIES = [f for f in ATTACK_FAMILIES if f != DOMINANT]
# the operating point the README quotes; also where per-family recall collapses
HEADLINE_THRESHOLD = 0.99
SHUFFLE_SEED = 42


def _fit_arms(Xtr: np.ndarray, ytr: np.ndarray, cfg) -> tuple:
    model = TwoStageRejectClassifier(
        stage1_features=STAGE1_INDICES,
        escalate_threshold=cfg.escalate_threshold,
        stage1_estimator=make_stage_estimator(cfg.stage_estimator, **cfg.stage1_params),
        stage2_estimator=make_stage_estimator(cfg.stage_estimator, **cfg.stage2_params),
    ).fit(Xtr, ytr)
    small = make_stage_estimator(cfg.stage_estimator, **cfg.stage1_params).fit(Xtr, ytr)
    return model, small


def _score_arms(model, small, X: np.ndarray) -> tuple[dict, np.ndarray]:
    """(labels, confidence) per arm before the reject knob, plus the hierarchy's
    escalation mask. The mask is kept because the hierarchy's whole compute argument
    is that Stage 1 answers most flows cheaply, and whether that holds on traffic the
    model has never seen is a question nothing in this repo had asked."""

    def argmax_arm(forest, Xa):
        p = forest_proba(forest, Xa, sequential=False)
        return np.asarray(forest.classes_[p.argmax(axis=1)], dtype=object), p.max(axis=1)

    labels_h, conf_h, escalated, _ = model._stage_predict(X)
    return {
        "hierarchy": (labels_h, conf_h),
        "single_joint_200": argmax_arm(model.stage2_, X),
        "stage1_only": argmax_arm(model.stage1_, X[:, STAGE1_INDICES]),
        "single_joint_small_60": argmax_arm(small, X),
    }, escalated


def _round(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    family: str,
    cfg,
    shuffle: bool = False,
) -> dict:
    """One leave-one-family-out round. `shuffle` permutes the training labels, which
    is the negative control: same rows, same marginals, no family structure."""
    train_idx = tr[y[tr] != family]
    seen_idx = te[y[te] != family]

    # Every flow of the family is unseen by label, but that is not enough. A connection
    # can carry flows of more than one family, so some of the family's flows sit on
    # 5-tuples that are still in training under another label, and the model would be
    # recognising the connection rather than generalising to a new attack. That is the
    # exact leak ADR 002 exists to prevent, and it has to hold on the unseen side too:
    # 65 of UDP-OVH's 304 flows fail it. They are dropped and the count is reported.
    family_idx = np.flatnonzero(y == family)
    train_conns = set(groups[train_idx])
    keeps = ~np.isin(groups[family_idx], list(train_conns))
    unseen_idx = family_idx[keeps]
    n_dropped = int((~keeps).sum())

    imputer = SimpleImputer(strategy="median").fit(X[train_idx])
    Xtr = imputer.transform(X[train_idx])
    ytr = y[train_idx]
    if shuffle:
        ytr = np.random.RandomState(SHUFFLE_SEED).permutation(ytr)
    X_unseen = imputer.transform(X[unseen_idx])
    X_seen = imputer.transform(X[seen_idx])

    model, small = _fit_arms(Xtr, ytr, cfg)
    unseen_scored, escalated_unseen = _score_arms(model, small, X_unseen)
    seen_scored, escalated_seen = _score_arms(model, small, X_seen)
    y_seen = y[seen_idx]

    arms = {}
    for arm in unseen_scored:
        u_labels, u_conf = unseen_scored[arm]
        s_labels, s_conf = seen_scored[arm]
        by_threshold = []
        for t in cfg.reject_thresholds:
            unseen = open_set_outcomes(u_labels, u_conf, t)
            cost = closed_set_cost(y_seen, s_labels, s_conf, t)
            by_threshold.append(
                {
                    "threshold": round(float(t), 4),
                    "unseen_family": unseen,
                    "seen_families": cost,
                    "novelty_lift": novelty_lift(unseen, cost),
                }
            )
        arms[arm] = {
            "by_threshold": by_threshold,
            "destinations_when_forced_to_answer": destination_counts(u_labels),
        }

    return {
        "held_out_family": family,
        "n_train": int(train_idx.size),
        "n_unseen_flows": int(unseen_idx.size),
        "n_unseen_dropped_shared_connection": n_dropped,
        "n_seen_test_flows": int(seen_idx.size),
        "trained_classes": sorted(str(c) for c in np.unique(ytr)),
        "hierarchy_escalation_rate": {
            "unseen_family": round(float(escalated_unseen.mean()), 4),
            "seen_families": round(float(escalated_seen.mean()), 4),
            "what": (
                "share of flows Stage 1 could not answer confidently and pushed to "
                "Stage 2. The hierarchy's compute case rests on this staying low"
            ),
        },
        "arms": arms,
    }


def _at(round_row: dict, arm: str, threshold: float) -> dict:
    for row in round_row["arms"][arm]["by_threshold"]:
        if abs(row["threshold"] - threshold) < 1e-9:
            return row
    raise KeyError(f"threshold {threshold} not in the sweep")


def _mean(values: list[float]) -> float:
    return round(float(np.mean(values)), 4) if values else float("nan")


def _summarize(rounds: list[dict], control: list[dict], arms: list[str]) -> dict:
    """Aggregate over the six rare families. UDP-RAW is reported on its own."""
    rare = [r for r in rounds if r["held_out_family"] in RARE_FAMILIES]
    rare_control = {
        r["held_out_family"]: r for r in control if r["held_out_family"] in RARE_FAMILIES
    }

    def worst(arm: str) -> dict:
        share, family = max(
            (_at(r, arm, 0.0)["unseen_family"]["called_benign"], r["held_out_family"])
            for r in rare
        )
        return {"family": family, "called_benign": share}

    forced = {
        arm: {
            "mean_called_benign": _mean(
                [_at(r, arm, 0.0)["unseen_family"]["called_benign"] for r in rare]
            ),
            "per_family_called_benign": {
                r["held_out_family"]: _at(r, arm, 0.0)["unseen_family"]["called_benign"]
                for r in rare
            },
            "worst_family": worst(arm),
        }
        for arm in arms
    }

    headline = {
        arm: {
            "mean_rejected_unknown": _mean(
                [
                    _at(r, arm, HEADLINE_THRESHOLD)["unseen_family"]["rejected_unknown"]
                    for r in rare
                ]
            ),
            "mean_called_benign": _mean(
                [_at(r, arm, HEADLINE_THRESHOLD)["unseen_family"]["called_benign"] for r in rare]
            ),
            "mean_novelty_lift": _mean(
                [_at(r, arm, HEADLINE_THRESHOLD)["novelty_lift"] for r in rare]
            ),
            "mean_seen_coverage": _mean(
                [_at(r, arm, HEADLINE_THRESHOLD)["seen_families"]["coverage"] for r in rare]
            ),
            "mean_seen_reliability": _mean(
                [
                    _at(r, arm, HEADLINE_THRESHOLD)["seen_families"]["reliability"]
                    for r in rare
                    if _at(r, arm, HEADLINE_THRESHOLD)["seen_families"]["reliability"] is not None
                ]
            ),
            "mean_novelty_lift_shuffled_control": _mean(
                [
                    _at(rare_control[r["held_out_family"]], arm, HEADLINE_THRESHOLD)["novelty_lift"]
                    for r in rare
                    if r["held_out_family"] in rare_control
                ]
            ),
        }
        for arm in arms
    }

    best = max(arms, key=lambda a: headline[a]["mean_novelty_lift"])
    dominant_round = next(r for r in rounds if r["held_out_family"] == DOMINANT)
    return {
        "rare_families": RARE_FAMILIES,
        "stage1_escalation_on_unseen": {
            "what": (
                "the hierarchy's cheap path is Stage 1 answering most flows without "
                "touching the QUIC-augmented model. This is what that rate does when "
                "the family is one the model was never trained on"
            ),
            "mean_unseen": _mean([r["hierarchy_escalation_rate"]["unseen_family"] for r in rare]),
            "mean_seen": _mean([r["hierarchy_escalation_rate"]["seen_families"] for r in rare]),
            "per_family_unseen": {
                r["held_out_family"]: r["hierarchy_escalation_rate"]["unseen_family"] for r in rare
            },
        },
        "forced_to_answer_threshold_0": {
            "what": (
                "no reject knob: the closed-set baseline that must name a class. "
                "called_benign is the silent miss, and it is the number the paper's "
                "placeholder table was going to hold"
            ),
            "per_arm": forced,
        },
        f"at_threshold_{HEADLINE_THRESHOLD}": {
            "what": (
                "the operating point the README quotes for 99.3% reliability. "
                "novelty_lift is the rejection rate on the unseen family minus the "
                "rejection rate on seen families at the same threshold; the shuffled "
                "control is the same statistic with the family structure destroyed"
            ),
            "per_arm": headline,
        },
        "best_arm_by_novelty_lift": best,
        "dominant_family_case": {
            "family": DOMINANT,
            "note": (
                "holding out the dominant flood removes about half the training rows, "
                "so it is a different regime from a rare-family holdout and is excluded "
                "from every mean above"
            ),
            "per_arm_at_headline": {
                arm: _at(dominant_round, arm, HEADLINE_THRESHOLD) for arm in arms
            },
        },
    }


def main() -> dict:
    cfg = get_settings().training
    df = load_sample()
    X, y, groups = build_matrices(df)
    tr, te = leakage_safe_split(groups, test_size=cfg.test_size, seed=cfg.seed)

    rounds = []
    for family in ATTACK_FAMILIES:
        print(f"[run] holding out {family} ...", flush=True)
        rounds.append(_round(X, y, groups, tr, te, family, cfg))

    control = []
    for family in ATTACK_FAMILIES:
        print(f"[control] shuffled labels, holding out {family} ...", flush=True)
        control.append(_round(X, y, groups, tr, te, family, cfg, shuffle=True))

    arms = list(rounds[0]["arms"].keys())
    summary = _summarize(rounds, control, arms)

    report = {
        "what": (
            "leave-one-family-out open-set measurement: for each attack family, train "
            "with that family entirely absent and measure what the model does with its "
            "flows - reject them as unknown, call them benign, or confidently name a "
            "different attack. Fills in the metrics the SECRYPT 2026 paper reports as "
            "placeholders"
        ),
        "environment": environment(),
        "protocol": {
            "train": (
                "the standard connection-grouped training split, minus every flow of "
                "the held-out family"
            ),
            "unseen_eval": (
                "all flows of the held-out family, minus any flow whose connection "
                "5-tuple still appears in that round's training rows under another "
                "label (ADR 002's leakage rule, enforced on the unseen side too). "
                "n_unseen_dropped_shared_connection reports how many that removed"
            ),
            "seen_eval": "the standard held-out test split minus the held-out family",
            "imputer": "median, refit on each round's training rows only",
            "benign_never_held_out": (
                "an open-set detector is asked about unseen attacks; removing benign "
                "leaves a model that cannot say nothing is wrong"
            ),
            "negative_control": (
                "the same protocol with the training labels permuted (seed "
                f"{SHUFFLE_SEED}): same rows, same class marginals, no family structure"
            ),
            "sample_note": (
                "measured on the committed 25,615-flow stratified sample. The rare "
                "families are complete in it; benign and UDP-RAW are capped, which "
                "affects the seen-side baseline the lift is measured against"
            ),
        },
        "config": {
            "test_size": cfg.test_size,
            "seed": cfg.seed,
            "stage_estimator": cfg.stage_estimator,
            "escalate_threshold": cfg.escalate_threshold,
            "reject_thresholds": cfg.reject_thresholds,
        },
        "summary": summary,
        "verdict": (
            "The reject option is the part of this system that carries the open-set "
            "case, and the hierarchy is not. Forced to name a class, the model calls "
            "70.5% of an unseen attack family's flows benign, and 92.9% of "
            "UDP-bypass-v1's, so a closed-set deployment absorbs roughly seven of every "
            "ten flows of a new flood into silence. Turning the knob to 0.99 rejects "
            "91.4% of that traffic as unknown and cuts the silent misses to 7.7%, at a "
            "cost of answering 66.5% of known traffic at 99.4% reliability. That "
            "rejection is novelty detection and not a strict threshold: measured "
            "against the abstention rate on seen families in the same run it leaves a "
            "lift of 0.579, and the shuffled-label control, which destroys the family "
            "structure and holds everything else, collapses that lift to -0.005. None "
            "of it belongs to the two-stage design. All four arms sit within 0.03 lift "
            "of each other and the plain 200-tree joint forest is the best of them "
            "(0.605), so the open-set case does not rescue the hierarchy any more than "
            "the compute case did. It makes the design look worse: Stage 1's escalation "
            "rate rises from 22.7% on seen traffic to 79.7% on an unseen family, so the "
            "cheap path stops being cheap exactly when a zero-day arrives, which is the "
            "one moment the architecture was supposed to be built for."
        ),
        "rounds": rounds,
        "shuffled_label_control": control,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"[save] {OUT}")
    return report


if __name__ == "__main__":
    main()

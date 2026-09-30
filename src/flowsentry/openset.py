"""
Open-set measurement: what the classifier does with an attack family it was never
trained on.

Every number this repo publishes is closed-set. The model is trained on eight
labels and scored on the same eight, so "accuracy" only ever answers "given that
this flow belongs to a family you have seen, do you name it correctly". The
operator question is the other one: a new UDP flood appears next month, the model
has never seen it, what comes out of the box. There are exactly three answers and
they are not equally bad.

  rejected as unknown   the reject knob fires and the flow is escalated to a human.
                        This is the only correct behaviour available to a
                        closed-set model, because the true label is not in its
                        output space at all.
  called benign         the worst outcome. An attack is absorbed into the class
                        that generates no alert, so the miss is silent.
  called a wrong attack the middle outcome. The family name is wrong, which
                        corrupts triage and any downstream family-specific
                        playbook, but an alert still fires and a human still looks.

Those three shares are exhaustive and sum to 1 by construction, which is the point:
on a family the model has never seen, there is no such thing as a correct answer,
so the usual precision/recall vocabulary does not apply and `evaluation.per_family`
is the wrong tool. Accuracy on these flows is 0 whatever happens. What the model
is graded on here is the SHAPE of its failure.

`novelty_lift` is the statistic that decides whether any of this is novelty
detection at all. A model that abstains on 90% of unseen flows looks impressive
until you notice it also abstains on 90% of the flows it was trained on: then the
abstention is the threshold being strict, not the model noticing anything. The
lift subtracts the baseline abstention rate measured on seen families at the same
threshold, on the same run. Only the difference is evidence of novelty detection.

The seen baseline in `novelty_lift` is ALL seen traffic, which on this sample is
mostly UDP-RAW and benign, the two classes the model finds easiest. That is not
the control a novelty claim needs. A rare flood can be rejected because it is
novel or simply because rare floods are hard, and those two readings predict
different things for a rare family that WAS in training: the first says it is
answered, the second says it is rejected too. `seen_family_outcomes` measures
exactly that group (the other rare families, trained on, in the same round) and
`novelty_lift_vs` subtracts it. Only a lift that survives that subtraction is
evidence the knob notices novelty rather than rarity.
"""
from __future__ import annotations

import numpy as np

BENIGN = "benign"


def open_set_outcomes(
    labels: np.ndarray,
    conf: np.ndarray,
    threshold: float,
    benign_label: str = BENIGN,
) -> dict:
    """The three-way outcome split for flows whose true family was never trained on.

    `labels` / `conf` are the arm's predictions and confidences for those flows,
    BEFORE the reject knob is applied; the knob is applied here at `threshold` so
    one scoring pass serves the whole sweep. Shares sum to 1.
    """
    labels = np.asarray(labels)
    conf = np.asarray(conf, dtype=float)
    n = int(labels.size)
    if n == 0:
        raise ValueError("open_set_outcomes needs at least one held-out flow")
    rejected = conf < threshold
    answered = ~rejected
    called_benign = answered & (labels == benign_label)
    called_attack = answered & (labels != benign_label)
    return {
        "n": n,
        "rejected_unknown": round(float(rejected.mean()), 4),
        "called_benign": round(float(called_benign.mean()), 4),
        "called_wrong_attack": round(float(called_attack.mean()), 4),
        "n_rejected": int(rejected.sum()),
        "n_called_benign": int(called_benign.sum()),
        "n_called_wrong_attack": int(called_attack.sum()),
    }


def closed_set_cost(
    y_true: np.ndarray,
    labels: np.ndarray,
    conf: np.ndarray,
    threshold: float,
) -> dict:
    """What the same threshold costs on the families the model DID train on.

    Rejecting everything scores a perfect unknown-detection rate and is useless, so
    the unseen-family number is only readable next to this one: coverage is the
    share of seen-family flows still answered, reliability the accuracy among them.
    """
    y_true = np.asarray(y_true)
    labels = np.asarray(labels)
    conf = np.asarray(conf, dtype=float)
    covered = conf >= threshold
    n_cov = int(covered.sum())
    return {
        "n": int(y_true.size),
        "coverage": round(float(covered.mean()), 4),
        "reliability": round(float((labels[covered] == y_true[covered]).mean()), 4)
        if n_cov
        else None,
        "n_covered": n_cov,
    }


def seen_family_outcomes(
    y_true: np.ndarray,
    labels: np.ndarray,
    conf: np.ndarray,
    threshold: float,
    benign_label: str = BENIGN,
) -> dict:
    """The same three-way split as open_set_outcomes, for attack flows whose family
    WAS trained on, with the answered attack calls split into right and wrong.
    Shares sum to 1: rejected + called_benign + called_right + called_wrong_attack."""
    y_true = np.asarray(y_true)
    labels = np.asarray(labels)
    conf = np.asarray(conf, dtype=float)
    n = int(labels.size)
    if n == 0:
        raise ValueError("seen_family_outcomes needs at least one flow")
    rejected = conf < threshold
    answered = ~rejected
    called_benign = answered & (labels == benign_label)
    called_right = answered & (labels == y_true)
    called_wrong = answered & ~called_benign & ~called_right
    return {
        "n": n,
        "rejected_unknown": round(float(rejected.mean()), 4),
        "called_benign": round(float(called_benign.mean()), 4),
        "called_right": round(float(called_right.mean()), 4),
        "called_wrong_attack": round(float(called_wrong.mean()), 4),
        "n_rejected": int(rejected.sum()),
        "n_called_benign": int(called_benign.sum()),
        "n_called_right": int(called_right.sum()),
        "n_called_wrong_attack": int(called_wrong.sum()),
    }


def novelty_lift_vs(unseen: dict, seen_outcomes: dict) -> float:
    """Unseen rejection rate minus the rejection rate on a named seen group
    (seen_family_outcomes output). With the other rare families as that group,
    this is the rarity-controlled lift."""
    return round(float(unseen["rejected_unknown"]) - float(seen_outcomes["rejected_unknown"]), 4)


def novelty_lift(unseen: dict, seen_cost: dict) -> float:
    """Unseen rejection rate minus the baseline rejection rate on seen families.

    Zero means the model abstains on a never-seen family exactly as often as on
    traffic it knows, i.e. the abstention carries no information about novelty.
    """
    seen_rejected = 1.0 - float(seen_cost["coverage"])
    return round(float(unseen["rejected_unknown"]) - seen_rejected, 4)


def destination_counts(labels: np.ndarray) -> dict:
    """Which known classes an unseen family's flows were called, most frequent first.

    A low benign-absorption share is only reassuring if the flows went somewhere
    sensible; this is what says whether an unseen flood is being read as a
    neighbouring flood or scattered across every label the model has.
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        return {}
    values, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    return {str(values[i]): int(counts[i]) for i in order}

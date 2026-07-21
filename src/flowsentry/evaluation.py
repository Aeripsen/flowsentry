"""
Evaluation helpers that answer per-family questions, kept out of train.py so the
reproducibility contract on artifacts/metrics.json is untouched.

train.py reports per-class PR-AUC and F1. Both are the wrong shape for the
operator question: PR-AUC scores a ranking, and F1 collapses misses and false
alarms into one number, so two families with opposite failure modes can share it.
Precision and recall keep them apart, and the confusion row says where a missed
family's flows actually went, which is the part that tells you whether a family
is being lost to benign or traded against a neighbouring flood.

Abstentions (a prediction of model.UNKNOWN) are handled the honest way here: a
rejected flow is not a hit and not a false positive, so it costs recall and
leaves precision describing only the answers the model chose to give. That is
what makes the reject knob measurable per family instead of only in aggregate.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def per_family(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    return {
        c: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i, c in enumerate(classes)
    }


def benign_absorption(rows: dict, benign_label: str = "benign") -> dict:
    """How much attack traffic the confusion rows lose to the benign class.

    A missed family shows up as a low recall either way, but "called a different
    flood" and "called benign" are different incidents: the first still raises an
    alert, the second is silence. Reported separately for the dominant flood and
    for the rest, because the dominant flood's near-perfect recall otherwise
    averages the rare families out of view.
    """
    attack = [c for c in rows if c != benign_label]
    if not attack:
        return {}
    dominant = max(attack, key=lambda c: rows[c]["n_flows"])
    rare = [c for c in attack if c != dominant]

    def block(names: list[str]) -> dict:
        flows = sum(rows[c]["n_flows"] for c in names)
        called_benign = sum(rows[c]["called"].get(benign_label, 0) for c in names)
        return {
            "flows": flows,
            "called_benign": called_benign,
            "share": round(called_benign / flows, 4) if flows else None,
        }

    return {
        "dominant_family": dominant,
        "all_attack": block(attack),
        "rare_families": block(rare),
    }


def predicted_label_counts(rows: dict) -> dict:
    counts: dict[str, int] = {}
    for row in rows.values():
        for label, n in row["called"].items():
            counts[label] = counts.get(label, 0) + n
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def confusion_rows(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    """For each true class, what it was called, most frequent first."""
    rows = {}
    for c in classes:
        m = y_true == c
        called, counts = np.unique(y_pred[m], return_counts=True)
        order = np.argsort(-counts)
        rows[c] = {
            "n_flows": int(m.sum()),
            "called": {str(called[i]): int(counts[i]) for i in order},
        }
    return rows

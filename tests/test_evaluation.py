"""Tests for the per-family evaluation helpers.

The load-bearing behaviour is how abstentions are counted: a rejected flow must
cost recall without flattering precision. Get that wrong and the per-family
reject numbers become an advertisement instead of a measurement.
"""
import numpy as np

from flowsentry.evaluation import (
    benign_absorption,
    confusion_rows,
    per_family,
    predicted_label_counts,
)
from flowsentry.model import UNKNOWN

CLASSES = ["a", "b"]


def test_precision_and_recall_split_the_two_failure_modes():
    y_true = np.array(["a", "a", "a", "b", "b"])
    y_pred = np.array(["a", "b", "b", "b", "b"])
    rows = per_family(y_true, y_pred, CLASSES)
    assert rows["a"] == {"precision": 1.0, "recall": round(1 / 3, 4), "f1": 0.5, "support": 3}
    # b answers 4 flows and only 2 are really b: the false-alarm side
    assert rows["b"]["precision"] == 0.5
    assert rows["b"]["recall"] == 1.0


def test_abstention_costs_recall_but_not_precision():
    y_true = np.array(["a", "a", "a", "a"])
    answered = np.array(["a", "a", UNKNOWN, UNKNOWN])
    rows = per_family(y_true, answered, CLASSES)
    assert rows["a"]["recall"] == 0.5
    assert rows["a"]["precision"] == 1.0


def test_confusion_row_names_where_the_misses_went():
    y_true = np.array(["a", "a", "a"])
    y_pred = np.array(["b", "b", "a"])
    rows = confusion_rows(y_true, y_pred, ["a"])
    assert rows["a"]["n_flows"] == 3
    assert list(rows["a"]["called"]) == ["b", "a"]  # most frequent first
    assert rows["a"]["called"]["b"] == 2


def test_benign_absorption_separates_the_dominant_flood_from_the_rare_ones():
    rows = {
        "big": {"n_flows": 100, "called": {"big": 99, "benign": 1}},
        "rare": {"n_flows": 10, "called": {"benign": 6, "rare": 4}},
        "benign": {"n_flows": 50, "called": {"benign": 50}},
    }
    out = benign_absorption(rows)
    assert out["dominant_family"] == "big"
    assert out["all_attack"] == {"flows": 110, "called_benign": 7, "share": 0.0636}
    # the number that matters: the rare family is silent 60% of the time, and
    # averaging it with the dominant flood hides that
    assert out["rare_families"] == {"flows": 10, "called_benign": 6, "share": 0.6}


def test_predicted_label_counts_totals_a_column_of_the_confusion():
    rows = {
        "a": {"n_flows": 3, "called": {"a": 2, "b": 1}},
        "b": {"n_flows": 4, "called": {"b": 4}},
    }
    assert predicted_label_counts(rows) == {"b": 5, "a": 2}

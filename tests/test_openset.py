"""Tests for the open-set helpers.

The load-bearing behaviour is that the three outcomes for an unseen family are
exhaustive and are not scored as a normal classification, and that the novelty
lift subtracts the baseline abstention rate. Get the second one wrong and a model
that abstains on everything looks like a zero-day detector.
"""
import numpy as np
import pytest

from flowsentry.openset import (
    closed_set_cost,
    destination_counts,
    novelty_lift,
    open_set_outcomes,
)


def test_the_three_outcomes_are_exhaustive():
    labels = np.array(["benign", "UDP-RAW", "benign", "UDP-VSE"])
    conf = np.array([0.99, 0.95, 0.40, 0.30])
    out = open_set_outcomes(labels, conf, threshold=0.9)
    assert out["rejected_unknown"] == 0.5  # the two below 0.9
    assert out["called_benign"] == 0.25  # only the 0.99 benign survives the knob
    assert out["called_wrong_attack"] == 0.25
    total = out["rejected_unknown"] + out["called_benign"] + out["called_wrong_attack"]
    assert total == pytest.approx(1.0)


def test_a_rejected_flow_is_not_counted_as_a_benign_miss():
    # the low-confidence benign call is an abstention, not a silent miss; conflating
    # them would understate the danger at threshold 0 and overstate it at 0.99
    labels = np.array(["benign", "benign"])
    conf = np.array([0.99, 0.10])
    out = open_set_outcomes(labels, conf, threshold=0.9)
    assert out["n_called_benign"] == 1
    assert out["n_rejected"] == 1


def test_threshold_zero_forces_an_answer_on_every_flow():
    labels = np.array(["benign", "UDP-RAW"])
    conf = np.array([0.0, 0.0])
    out = open_set_outcomes(labels, conf, threshold=0.0)
    assert out["rejected_unknown"] == 0.0


def test_closed_set_cost_reports_accuracy_only_on_answered_flows():
    y_true = np.array(["a", "a", "b", "b"])
    labels = np.array(["a", "b", "b", "a"])
    conf = np.array([0.99, 0.10, 0.95, 0.20])
    cost = closed_set_cost(y_true, labels, conf, threshold=0.9)
    assert cost["coverage"] == 0.5
    assert cost["reliability"] == 1.0  # both wrong answers were abstained on
    assert cost["n_covered"] == 2


def test_novelty_lift_is_zero_when_the_model_abstains_on_everything_equally():
    unseen = {"rejected_unknown": 0.9}
    seen = {"coverage": 0.1}  # so seen rejection is also 0.9
    assert novelty_lift(unseen, seen) == 0.0


def test_novelty_lift_is_positive_only_for_the_excess_over_the_baseline():
    unseen = {"rejected_unknown": 0.95}
    seen = {"coverage": 0.75}  # baseline rejection 0.25
    assert novelty_lift(unseen, seen) == 0.7


def test_destinations_are_ordered_most_frequent_first():
    labels = np.array(["benign", "UDP-RAW", "benign", "benign", "UDP-RAW"])
    assert list(destination_counts(labels)) == ["benign", "UDP-RAW"]
    assert destination_counts(labels)["benign"] == 3


def test_empty_holdout_is_an_error_not_a_silent_zero():
    with pytest.raises(ValueError):
        open_set_outcomes(np.array([]), np.array([]), threshold=0.9)


def test_seen_family_outcomes_split_answered_attacks_into_right_and_wrong():
    import numpy as np

    from flowsentry.openset import novelty_lift_vs, open_set_outcomes, seen_family_outcomes

    y = np.array(["A", "A", "A", "B", "B"])
    labels = np.array(["A", "benign", "B", "B", "A"])
    conf = np.array([0.99, 0.99, 0.99, 0.5, 0.99])
    o = seen_family_outcomes(y, labels, conf, 0.9)
    assert (o["n_rejected"], o["n_called_benign"], o["n_called_right"],
            o["n_called_wrong_attack"]) == (1, 1, 1, 2)
    assert o["rejected_unknown"] + o["called_benign"] + o["called_right"] + o[
        "called_wrong_attack"
    ] == 1.0
    unseen = open_set_outcomes(np.array(["A", "A"]), np.array([0.1, 0.1]), 0.9)
    assert novelty_lift_vs(unseen, o) == 0.8

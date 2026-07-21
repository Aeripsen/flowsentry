"""Tests for the calibration measures.

Checked against cases whose answer can be worked out by hand, because a
calibration bug is silent: it produces a plausible small number either way.
"""
import numpy as np

from flowsentry.calibration import (
    brier_multiclass,
    brier_top_label,
    ece,
    mce,
    reliability_bins,
)


def test_perfectly_calibrated_confidence_scores_zero_ece():
    # 100 flows at confidence 0.7, exactly 70 of them correct
    conf = np.full(100, 0.7)
    correct = np.array([1.0] * 70 + [0.0] * 30)
    assert ece(conf, correct) == 0.0
    assert mce(conf, correct) == 0.0
    # and Brier still charges for the uncertainty, which is the point of using it
    assert round(brier_top_label(conf, correct), 4) == 0.21


def test_ece_weights_bins_by_how_many_flows_they_hold():
    conf = np.array([1.0] * 99 + [0.55])
    correct = np.array([1.0] * 99 + [0.0])
    # the bad bin is one flow in a hundred: ECE barely moves, MCE reports it
    assert round(ece(conf, correct), 4) == 0.0055
    assert round(mce(conf, correct), 2) == 0.55


def test_bins_carry_their_counts_so_a_thin_bin_is_visible():
    conf = np.array([0.05, 0.95, 0.95])
    correct = np.array([0.0, 1.0, 1.0])
    rows = reliability_bins(conf, correct)
    assert [r["n"] for r in rows] == [1, 0, 0, 0, 0, 0, 0, 0, 0, 2]
    assert rows[1]["mean_confidence"] is None  # empty bin, not a fabricated 0
    assert rows[-1]["accuracy"] == 1.0


def test_multiclass_brier_charges_both_classes_when_confidently_wrong():
    proba = np.array([[1.0, 0.0], [0.5, 0.5]])
    y_true = np.array(["b", "b"])
    # first flow: 1 on the wrong class + 1 on the true one; second: 0.25 + 0.25
    assert brier_multiclass(proba, y_true, ["a", "b"]) == 1.25

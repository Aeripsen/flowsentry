"""Tests for the gradient-boosted comparison arms.

The helper functions are checked against cases whose answer can be worked out by
hand, same policy as the calibration tests: a curve or calibrator bug produces a
plausible number either way, so plausibility is not a test. The estimator tests
skip when lightgbm/xgboost are absent, because they are an optional extra and CI
installs only [dev]; the helpers have no such dependency and always run.
"""
import numpy as np
import pytest

from flowsentry.gbdt import (
    LIGHTGBM_GRID,
    XGBOOST_GRID,
    binary_attack_pr_auc,
    fit_confidence_calibrators,
    pick_calibrator,
    reject_curve,
    top_label,
)


def test_grids_are_small_enough_to_print():
    # the whole point of the printed grid is that a reader can audit every row;
    # if either grid grows past a screenful, that promise silently dies
    assert len(LIGHTGBM_GRID) <= 12
    assert len(XGBOOST_GRID) <= 12
    # and every entry names its values explicitly, no callables or ranges
    for entry in LIGHTGBM_GRID + XGBOOST_GRID:
        assert all(isinstance(v, int | float) for v in entry.values())


def test_top_label_matches_argmax_by_hand():
    proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    labels, conf = top_label(proba, ["a", "b", "c"])
    assert list(labels) == ["a", "c"]
    assert conf.tolist() == [0.7, 0.8]


def test_binary_attack_pr_auc_perfect_separation_scores_one():
    # benign flows get high P(benign), attacks get low: ranking is perfect
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])
    y = np.array(["benign", "benign", "UDP-RAW", "UDP-RAW"])
    assert binary_attack_pr_auc(y, proba, ["benign", "UDP-RAW"]) == 1.0


def test_reject_curve_counts_by_hand():
    conf = np.array([0.2, 0.6, 0.95, 0.99])
    correct = np.array([0.0, 0.0, 1.0, 1.0])
    rows = reject_curve(conf, correct, [0.0, 0.9])
    assert rows[0] == {"threshold": 0.0, "coverage": 1.0, "reliability": 0.5, "n_covered": 4}
    assert rows[1] == {"threshold": 0.9, "coverage": 0.5, "reliability": 1.0, "n_covered": 2}


def test_reject_curve_empty_coverage_reports_none_not_nan():
    rows = reject_curve(np.array([0.1, 0.2]), np.array([1.0, 0.0]), [0.99])
    assert rows[0]["reliability"] is None
    assert rows[0]["n_covered"] == 0


def test_pick_calibrator_chooses_the_lower_held_out_ece():
    """The pick is checked with hand-built maps whose ECE is exact: everything
    sits at confidence 0.9 with true accuracy 0.5, so the map that says 0.5 has
    ECE 0 and the identity map has ECE 0.4. A fitted-map contest was tried here
    first and dropped: which of isotonic and Platt wins depends on the sample,
    which makes the winner an assumption, not an assertion."""
    conf = np.full(100, 0.9)
    correct = np.array([1.0] * 50 + [0.0] * 50)
    honest = {"says_half": lambda c: np.full_like(c, 0.5), "identity": lambda c: c}
    best, scores = pick_calibrator(honest, conf, correct)
    assert best == "says_half"
    assert scores == {"says_half": 0.0, "identity": 0.4}


def test_fitted_calibrators_output_probabilities():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.2, 1.0, size=500)
    correct = (rng.uniform(size=500) < conf).astype(float)
    calibrators = fit_confidence_calibrators(conf, correct)
    assert set(calibrators) == {"isotonic", "platt"}
    for cal in calibrators.values():
        out = cal(conf)
        assert out.min() >= 0.0 and out.max() <= 1.0


def test_lightgbm_arm_fits_and_keeps_string_classes():
    pytest.importorskip("lightgbm")
    from flowsentry.gbdt import make_lightgbm

    X, y = _tiny_data()
    model = make_lightgbm(n_estimators=10, num_leaves=7).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert set(model.classes_) == set(y)


def test_xgboost_wrapper_round_trips_string_labels():
    pytest.importorskip("xgboost")
    from flowsentry.gbdt import make_xgboost

    X, y = _tiny_data()
    model = make_xgboost(n_estimators=10, max_depth=3).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 3)
    # the contract the repo relies on: classes_ are the original names, sorted
    # the way LabelEncoder sorts them, and proba columns follow that order
    assert list(model.classes_) == sorted(set(y))
    labels, conf = top_label(proba, list(model.classes_))
    assert set(labels).issubset(set(y))
    assert conf.max() <= 1.0


def _tiny_data():
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=300, n_features=12, n_informative=6, n_classes=3, random_state=0
    )
    return X, np.array([f"family_{k}" for k in y])

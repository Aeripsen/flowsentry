"""Registry tests. The estimator seam is proven by actually swapping what runs in
both stages and getting a working two-stage pipeline out the other end, not by an
interface with one implementer."""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from flowsentry.model import TwoStageRejectClassifier
from flowsentry.registry import StageClassifier, available, make_stage_estimator


def _data():
    X, y = make_classification(
        n_samples=400, n_features=24, n_informative=12, n_classes=3, random_state=0
    )
    return X, y.astype(str)


def test_registry_has_two_real_implementations():
    assert set(available()) >= {"random_forest", "hist_gradient_boosting"}


def test_every_registered_estimator_runs_the_full_pipeline():
    """The actual swap: fit the whole two-stage reject pipeline with each
    registered estimator family and exercise predict, the reject knob, the
    curve, and the sequential fast path (which must fall back cleanly for
    non-forest estimators)."""
    X, y = _data()
    params = {
        "random_forest": {"n_estimators": 15, "random_state": 0},
        "hist_gradient_boosting": {"max_iter": 15, "random_state": 0},
    }
    for name in available():
        model = TwoStageRejectClassifier(
            stage1_estimator=make_stage_estimator(name, **params[name]),
            stage2_estimator=make_stage_estimator(name, **params[name]),
        ).fit(X, y)
        assert isinstance(model.stage1_, StageClassifier)
        preds = model.predict(X)
        assert set(preds).issubset(set(y))
        curve = model.coverage_reliability_curve(X, y, [0.0, 0.9])
        assert curve[0]["coverage"] == 1.0
        proba_seq = model.predict_proba(X, sequential=True)
        proba_nat = model.predict_proba(X, sequential=False)
        assert np.allclose(proba_seq, proba_nat, atol=1e-12), name


def test_unknown_estimator_fails_loudly_with_the_menu():
    with pytest.raises(ValueError, match="available: hist_gradient_boosting, random_forest"):
        make_stage_estimator("transformer_xl")


def test_default_path_is_the_measured_random_forest():
    """With no estimators injected, the model must keep building exactly the
    forests every reported number was measured with."""
    X, y = _data()
    model = TwoStageRejectClassifier(
        n_estimators_stage1=10, n_estimators_stage2=12
    ).fit(X, y)
    assert isinstance(model.stage1_, RandomForestClassifier)
    assert model.stage1_.n_estimators == 10
    assert model.stage2_.n_estimators == 12
    assert model.stage1_.class_weight == "balanced_subsample"
    assert model.stage1_.random_state == 42

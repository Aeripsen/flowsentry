"""MLflow for real, not a string test: log a training run to a throwaway SQLite
store, read it back, reload the model, and show that a model which does not
reproduce its predictions stops the run.

Skipped when mlflow is not installed (the CI test matrix runs the locked base
environment on purpose, to prove the seam is optional). The CI mlops job installs
mlflow and runs this file."""
from __future__ import annotations

import json

import numpy as np
import pytest

mlflow = pytest.importorskip("mlflow")

from sklearn.datasets import make_classification  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402

from flowsentry import tracking  # noqa: E402
from flowsentry.config import get_settings  # noqa: E402
from flowsentry.model import TwoStageRejectClassifier  # noqa: E402


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A fresh store per test; artifacts land under tmp_path, never in the repo."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'store.db').as_posix()}")
    monkeypatch.setenv(tracking.SWITCH, "1")
    monkeypatch.setenv("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    return tmp_path


def _fitted():
    X, y = make_classification(n_samples=400, n_features=24, n_informative=12,
                               n_classes=3, random_state=0)
    X[::7, 3] = np.nan  # so the logged imputer has work to do on the raw rows
    y = y.astype(str)
    imp = SimpleImputer(strategy="median").fit(X[:300])
    model = TwoStageRejectClassifier(stage1_features=list(range(8)), n_estimators_stage1=10,
                                     n_estimators_stage2=20).fit(imp.transform(X[:300]), y[:300])
    return X[300:], imp, model


def _log(tmp_path, proba_offset: float = 0.0) -> str:
    X_raw, imp, model = _fitted()
    Xte = imp.transform(X_raw)
    metrics = {"accuracy_full_coverage": 0.81, "per_class_pr_auc": {"0": 0.9, "1": 0.8},
               "coverage_reliability_curve": [
                   {"threshold": 0.5, "coverage": 0.9, "reliability": 0.85},
                   {"threshold": 0.9, "coverage": 0.6, "reliability": 0.97}]}
    mpath = tmp_path / "metrics.json"
    mpath.write_text(json.dumps(metrics))
    return tracking.log_training_run(
        settings=get_settings(), metrics=metrics, imputer=imp, model=model,
        X_example=Xte[:5], metrics_path=mpath, X_test_raw=X_raw,
        expected_proba=model.predict_proba(Xte, sequential=True) + proba_offset,
        expected_labels=model.predict(Xte, reject_threshold=0.0, sequential=True))


def test_training_run_is_stored_and_its_model_reloads_identically(store):
    run_id = _log(store)
    run = mlflow.get_run(run_id)
    assert run.data.params["training.seed"] == str(get_settings().training.seed)
    assert run.data.metrics["accuracy_full_coverage"] == 0.81
    assert run.data.metrics["pr_auc.1"] == 0.8
    assert run.data.tags["roundtrip_predictions_identical"] == "True"
    assert run.data.metrics["roundtrip.max_abs_proba_diff"] == 0.0
    assert run.data.metrics["roundtrip.n_rows"] == 100
    curve = mlflow.MlflowClient().get_metric_history(run_id, "curve.coverage")
    assert sorted((m.step, m.value) for m in curve) == [(50, 0.9), (90, 0.6)]

    # independent reload through the sklearn flavor, from the raw rows
    X_raw, imp, model = _fitted()
    (logged,) = mlflow.search_logged_models(filter_string=f"source_run_id = '{run_id}'",
                                            output_format="list")
    loaded = mlflow.sklearn.load_model(logged.model_uri)
    assert np.array_equal(loaded.predict_proba(X_raw, sequential=True),
                          model.predict_proba(imp.transform(X_raw), sequential=True))


def test_a_model_that_does_not_reproduce_stops_the_run(store):
    # the smallest possible disagreement: the round trip is exact, not approximate
    with pytest.raises(tracking.RoundTripError):
        _log(store, proba_offset=1e-15)
    (run,) = mlflow.search_runs(experiment_names=["flowsentry-train"], output_format="list")
    assert run.data.tags["roundtrip_predictions_identical"] == "False"

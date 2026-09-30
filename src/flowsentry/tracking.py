"""
Experiment tracking seam (MLflow). Optional: the base install, the serving path
and the locked reproduce job never import mlflow.

Why this exists: every number this repo publishes already lands in a committed
JSON file, which is the reproducibility contract. What a JSON file cannot do is
put the runs side by side: which config, which data hash, which commit, what it
scored, and the fitted model that scored it. That is what a tracking store is for,
so train.py logs every training run here and scripts/track_arms.py logs the
model comparison (forest vs two-stage hierarchy vs tuned boosters) as one run per
arm in its own experiment.

Where runs go:
  * default: a local SQLite store at <repo>/mlflow.db with artifacts under
    <repo>/mlruns/ (both gitignored; they hold absolute paths and 60 MB
    model pickles, so they are rebuilt, not committed). View them with
        mlflow ui --backend-store-uri sqlite:///mlflow.db
  * MLFLOW_TRACKING_URI set: that server or store is used instead, untouched.
  * FLOWSENTRY_MLFLOW=0: tracking is off even if mlflow is installed.

Install with pip install -e ".[mlops]".
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = REPO_ROOT / "mlflow.db"
DEFAULT_ARTIFACTS = REPO_ROOT / "mlruns"
SWITCH = "FLOWSENTRY_MLFLOW"

# mlflow 3 saves sklearn models with skops, not pickle, and refuses to load any
# type nobody vouched for. These are the types the shipped pipeline contains,
# each reviewed rather than copied from get_untrusted_types(): our own model
# class, numpy's dtype, and sklearn's tree node storage (skops warns it indexes
# nodes without bounds checks, which is only a risk for a file from someone else).
SKOPS_TRUSTED_TYPES = [
    "flowsentry.model.TwoStageRejectClassifier",
    "numpy.dtype",
    "sklearn.tree._tree.Tree",
]


def enabled() -> bool:
    """True when mlflow is importable and tracking has not been switched off."""
    if os.environ.get(SWITCH, "1").strip() == "0":
        return False
    try:
        import mlflow  # noqa: F401
    except ImportError:
        return False
    return True


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{DEFAULT_DB.as_posix()}"


def use_experiment(name: str) -> str:
    """Point mlflow at the store and select (creating if needed) an experiment.
    Returns the experiment id. With the default local store, artifacts are pinned
    to <repo>/mlruns so they never scatter into whatever the cwd is."""
    import mlflow

    uri = tracking_uri()
    mlflow.set_tracking_uri(uri)
    exp = mlflow.get_experiment_by_name(name)
    if exp is not None:
        mlflow.set_experiment(experiment_id=exp.experiment_id)
        return str(exp.experiment_id)
    location = DEFAULT_ARTIFACTS.as_uri() if "MLFLOW_TRACKING_URI" not in os.environ else None
    exp_id = mlflow.create_experiment(name, artifact_location=location)
    mlflow.set_experiment(experiment_id=exp_id)
    return str(exp_id)


def flatten(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Nested config dict -> flat dotted keys, the shape mlflow params take."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(flatten(v, prefix=f"{key}."))
        elif isinstance(v, list | tuple):
            out[key] = ",".join(str(x) for x in v)
        else:
            out[key] = v
    return out


def metric_key(name: str) -> str:
    """mlflow metric names allow alphanumerics and _ - . / and space only."""
    return "".join(c if c.isalnum() or c in "_-./ " else "_" for c in name)


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pip_requirements(extra: list[str] | None = None) -> list[str]:
    """Pinned requirements for a logged model, read from the running environment.
    Passing them explicitly skips mlflow's inference step, which reloads the model
    in a subprocess and is slow for a 60 MB forest."""
    from importlib.metadata import PackageNotFoundError, version

    names = ["scikit-learn", "numpy", "pandas", "scipy", "joblib", *(extra or [])]
    reqs = []
    for n in names:
        try:
            reqs.append(f"{n}=={version(n)}")
        except PackageNotFoundError:
            continue
    return reqs


def standard_tags(sample_path: Path) -> dict[str, str]:
    return {
        "git_sha": git_sha(),
        "data.sample_sha256": file_sha256(sample_path),
        "data.sample_path": sample_path.relative_to(REPO_ROOT).as_posix()
        if sample_path.is_relative_to(REPO_ROOT)
        else str(sample_path),
    }


def log_training_run(
    *,
    settings: Any,
    metrics: Mapping[str, Any],
    imputer: Any,
    model: Any,
    X_example: Any,
    metrics_path: Path,
    X_test_raw: Any,
    expected_proba: Any,
    expected_labels: Any,
) -> str:
    """Log one train.py run: every training param, every scalar metric, the
    coverage-reliability curve as a stepped metric, metrics.json, and the fitted
    imputer+model as an mlflow sklearn model. Returns the run id.

    The logged model is then loaded back from the store and run on the raw
    (un-imputed) held-out rows. Its class probabilities and its labels must equal
    the in-memory ones exactly, or this raises RoundTripError and train.py exits
    non-zero. The run is tagged with the outcome either way. Both sides score on
    the sequential forest path (model.forest_proba): the default n_jobs=-1 path
    sums trees across threads in whatever order they finish, so two calls on the
    SAME in-memory model differ by up to 2.2e-16, and an exact check on it would
    measure thread scheduling instead of the store."""
    import mlflow
    import mlflow.sklearn
    from sklearn.pipeline import Pipeline

    use_experiment("flowsentry-train")
    with mlflow.start_run(run_name="two_stage_hierarchy") as run:
        mlflow.set_tags(
            {**standard_tags(settings.sample_path), "entrypoint": "flowsentry.train"}
        )
        mlflow.log_params(flatten(settings.training.model_dump(), prefix="training."))
        scalars = {
            metric_key(k): float(v)
            for k, v in metrics.items()
            if isinstance(v, int | float) and not isinstance(v, bool)
        }
        for k, v in metrics.get("per_class_pr_auc", {}).items():
            scalars[metric_key(f"pr_auc.{k}")] = float(v)
        for k, v in metrics.get("per_class_f1", {}).items():
            scalars[metric_key(f"f1.{k}")] = float(v)
        for k, v in metrics.get("ablation_single_rf", {}).items():
            if isinstance(v, int | float):
                scalars[metric_key(f"ablation_single_rf.{k}")] = float(v)
        mlflow.log_metrics(scalars)
        # the reject knob's curve; step = threshold in percent so the UI plots it
        for row in metrics.get("coverage_reliability_curve", []):
            step = int(round(float(row["threshold"]) * 100))
            mlflow.log_metric("curve.coverage", float(row["coverage"]), step=step)
            if row.get("reliability") is not None:
                mlflow.log_metric("curve.reliability", float(row["reliability"]), step=step)
        mlflow.log_artifact(str(metrics_path))
        info = mlflow.sklearn.log_model(
            Pipeline([("impute", imputer), ("model", model)]),
            name="model",
            input_example=X_example,
            pip_requirements=pip_requirements(),
            code_paths=[str(REPO_ROOT / "src" / "flowsentry")],
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
        check = roundtrip_check(info.model_uri, X_test_raw, expected_proba, expected_labels)
        mlflow.set_tag("roundtrip_predictions_identical", str(check["identical"]))
        mlflow.log_metric("roundtrip.max_abs_proba_diff", check["max_abs_proba_diff"])
        mlflow.log_metric("roundtrip.n_rows", check["n_rows"])
        if not check["identical"]:
            raise RoundTripError(
                f"logged model {info.model_uri} does not reproduce the in-memory "
                f"predictions: {check}"
            )
        return str(run.info.run_id)


class RoundTripError(RuntimeError):
    """The model read back from the tracking store scores differently from the
    model that was logged."""


def roundtrip_check(model_uri: str, X_raw: Any, expected_proba: Any,
                    expected_labels: Any) -> dict[str, Any]:
    """Load a logged imputer+model pipeline back (sklearn flavor, skops with the
    reviewed trusted types) and compare it with the in-memory predictions on every
    row: the probability matrix bit for bit and the labels. The labels are the
    full-coverage ones train.py scores (reject_threshold=0). Scored on the
    sequential path, which is deterministic; pass expectations from the same path."""
    import mlflow.sklearn
    import numpy as np

    loaded = mlflow.sklearn.load_model(model_uri)
    Xt = loaded[:-1].transform(X_raw)
    proba = np.asarray(loaded[-1].predict_proba(Xt, sequential=True))
    labels = np.asarray(loaded[-1].predict(Xt, reject_threshold=0.0, sequential=True))
    want_p = np.asarray(expected_proba)
    same_shape = proba.shape == want_p.shape
    diff = float(np.max(np.abs(proba - want_p))) if same_shape and proba.size else float("inf")
    return {
        "n_rows": int(len(proba)),
        "proba_identical": bool(same_shape and np.array_equal(proba, want_p)),
        "labels_identical": bool(np.array_equal(labels, np.asarray(expected_labels))),
        "max_abs_proba_diff": diff,
        "identical": bool(same_shape and np.array_equal(proba, want_p)
                          and np.array_equal(labels, np.asarray(expected_labels))),
    }

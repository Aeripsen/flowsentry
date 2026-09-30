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
) -> str:
    """Log one train.py run: every training param, every scalar metric, the
    coverage-reliability curve as a stepped metric, metrics.json, and the fitted
    imputer+model as an mlflow sklearn model. Returns the run id."""
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
        mlflow.sklearn.log_model(
            Pipeline([("impute", imputer), ("model", model)]),
            name="model",
            input_example=X_example,
            pip_requirements=pip_requirements(),
            code_paths=[str(REPO_ROOT / "src" / "flowsentry")],
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
        return str(run.info.run_id)

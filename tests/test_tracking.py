"""The MLflow seam and the committed MLOps summaries. Runs without mlflow or
evidently installed: the seam must stay optional, and the committed summaries
must agree with metrics.json, the file every published number comes from."""
from __future__ import annotations

import json
from pathlib import Path

from flowsentry import tracking

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def test_flatten_nested_config():
    flat = tracking.flatten({"seed": 42, "stage1": {"n_estimators": 60}, "t": [0.5, 0.9]})
    assert flat == {"seed": 42, "stage1.n_estimators": 60, "t": "0.5,0.9"}


def test_metric_key_keeps_family_names_and_drops_illegal_chars():
    assert tracking.metric_key("pr_auc.UDP-bypass-v1") == "pr_auc.UDP-bypass-v1"
    assert tracking.metric_key("a:b(c)") == "a_b_c_"


def test_switch_turns_tracking_off(monkeypatch):
    monkeypatch.setenv(tracking.SWITCH, "0")
    assert tracking.enabled() is False


def test_tracking_uri_defaults_to_repo_sqlite_and_respects_env(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert tracking.tracking_uri().startswith("sqlite:///")
    assert tracking.tracking_uri().endswith("/mlflow.db")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    assert tracking.tracking_uri() == "http://127.0.0.1:5000"


def test_tracked_arms_agree_with_metrics_json():
    arms = json.loads((ARTIFACTS / "mlflow_arms.json").read_text())
    m = json.loads((ARTIFACTS / "metrics.json").read_text())
    shipped = arms["arms"]["two_stage_hierarchy"]["metrics"]
    assert shipped["binary_attack_pr_auc"] == m["binary_attack_detection_pr_auc"]
    assert shipped["accuracy_full_coverage"] == m["accuracy_full_coverage"]
    assert shipped["macro_f1_full_coverage"] == m["macro_f1_full_coverage"]
    assert arms["all_arms_match_committed_artifacts"] is True
    assert all(a["logged_model_reproduces_predictions"] for a in arms["arms"].values())


def test_evidently_current_window_is_the_published_model():
    ev = json.loads((ARTIFACTS / "evidently_summary.json").read_text())
    m = json.loads((ARTIFACTS / "metrics.json").read_text())
    cur = ev["performance"]["current_held_out"]
    assert cur["accuracy"] == m["accuracy_full_coverage"]
    assert cur["f1_macro"] == m["macro_f1_full_coverage"]
    assert ev["n_current"] == m["n_test"] and ev["n_reference"] == m["n_train"]
    assert (Path(__file__).resolve().parents[1] / ev["html_report"]).exists()

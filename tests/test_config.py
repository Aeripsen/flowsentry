"""Config tests. The one that matters most is the first: the in-code defaults ARE
the measured configuration, so if someone edits them the metrics regression and
this test both fail, loudly, instead of the README quietly going stale."""
import pytest
from pydantic import ValidationError

from flowsentry.config import Settings


def test_defaults_are_the_measured_configuration():
    s = Settings()
    assert s.training.test_size == 0.25
    assert s.training.seed == 42
    assert s.training.escalate_threshold == 0.90
    assert s.training.reject_thresholds == [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    assert s.training.stage_estimator == "random_forest"
    assert s.training.stage1_params == {"n_estimators": 60, "random_state": 42}
    assert s.training.stage2_params == {"n_estimators": 200, "random_state": 42}
    assert s.serving.sequential_cutoff == 2048
    assert s.artifact_dir.name == "artifacts"
    assert s.sample_path.name == "bccc_udp_quic_sample.csv.gz"


def test_env_override(monkeypatch):
    monkeypatch.setenv("FLOWSENTRY_TRAINING__SEED", "7")
    monkeypatch.setenv("FLOWSENTRY_SERVING__SEQUENTIAL_CUTOFF", "512")
    s = Settings()
    assert s.training.seed == 7
    assert s.serving.sequential_cutoff == 512
    # untouched fields keep their defaults
    assert s.training.test_size == 0.25


def test_yaml_override(tmp_path, monkeypatch):
    cfg = tmp_path / "flowsentry.yaml"
    cfg.write_text(
        "training:\n  stage_estimator: hist_gradient_boosting\n  seed: 9\n"
    )
    monkeypatch.setenv("FLOWSENTRY_CONFIG", str(cfg))
    s = Settings()
    assert s.training.stage_estimator == "hist_gradient_boosting"
    assert s.training.seed == 9


def test_env_beats_yaml(tmp_path, monkeypatch):
    cfg = tmp_path / "flowsentry.yaml"
    cfg.write_text("training:\n  seed: 9\n")
    monkeypatch.setenv("FLOWSENTRY_CONFIG", str(cfg))
    monkeypatch.setenv("FLOWSENTRY_TRAINING__SEED", "11")
    assert Settings().training.seed == 11


def test_unknown_keys_are_rejected(tmp_path, monkeypatch):
    """A typo in the config must fail the run, not train a silently different
    model with the default value."""
    cfg = tmp_path / "flowsentry.yaml"
    cfg.write_text("trainnig:\n  seed: 9\n")
    monkeypatch.setenv("FLOWSENTRY_CONFIG", str(cfg))
    with pytest.raises(ValidationError):
        Settings()


def test_invalid_values_are_rejected(monkeypatch):
    monkeypatch.setenv("FLOWSENTRY_TRAINING__TEST_SIZE", "1.5")
    with pytest.raises(ValidationError):
        Settings()

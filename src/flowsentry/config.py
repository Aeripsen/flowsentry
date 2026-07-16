"""
Runtime configuration for training and serving.

Defaults live HERE, in code, and are the exact values every reported number was
measured with, so a missing config file can never silently train a different
model. Overrides, highest precedence first:

  1. environment variables:  FLOWSENTRY_TRAINING__SEED=7  (nested via __)
  2. an optional YAML file:  flowsentry.yaml at the repo root, or wherever
                             FLOWSENTRY_CONFIG points (see flowsentry.example.yaml)
  3. the defaults below

Deliberately NOT configurable: the feature schema (UDP_FEATURES / QUIC_FEATURES in
data.py). Those lists are the dataset contract, guarded by tests; a config knob for
them would turn a typo into a silently different model. See docs/adr/006.
"""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _config_path() -> Path:
    return Path(os.environ.get("FLOWSENTRY_CONFIG", REPO_ROOT / "flowsentry.yaml"))


class TrainingConfig(BaseModel):
    test_size: float = Field(0.25, gt=0.0, lt=1.0)
    seed: int = 42
    escalate_threshold: float = Field(0.90, ge=0.0, le=1.0)
    reject_thresholds: list[float] = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    # any name registered in registry.py
    stage_estimator: str = "random_forest"
    stage1_params: dict = {"n_estimators": 60, "random_state": 42}
    stage2_params: dict = {"n_estimators": 200, "random_state": 42}


class ServingConfig(BaseModel):
    # batch size where score_batch hands off from sequential tree scoring to the
    # threaded path; measured bracket on the dev machine is (1024, 4096)
    sequential_cutoff: int = Field(2048, gt=0)
    # request-size ceiling for POST /predict/batch (memory + oracle-abuse bound)
    max_batch_rows: int = Field(4096, gt=0)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FLOWSENTRY_", env_nested_delimiter="__", extra="forbid"
    )

    artifact_dir: Path = REPO_ROOT / "artifacts"
    sample_path: Path = REPO_ROOT / "data" / "sample" / "bccc_udp_quic_sample.csv.gz"
    training: TrainingConfig = TrainingConfig()
    serving: ServingConfig = ServingConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # precedence: explicit init args, then env vars, then the YAML file
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file=_config_path()),
        )


def get_settings() -> Settings:
    """Fresh settings from env + optional YAML. Cheap enough to not cache; module
    constants that must not change mid-process read it once at import."""
    return Settings()

"""
Stage-estimator registry: swap what each stage runs without touching dispatch.

The two-stage model does not care what its stages are, only that they satisfy the
sklearn classifier contract (fit, predict_proba, classes_) captured by the
StageClassifier protocol below. Two real implementations ship today:

  random_forest           the default and what every reported number uses; the
                          bagged-trees architecture from the SECRYPT paper.
  hist_gradient_boosting  sklearn's histogram gradient booster (its LightGBM
                          equivalent), a genuinely different model family: boosted
                          not bagged, native NaN handling, no estimators_ forest,
                          so it also exercises the scoring fallback path.

A test fits the full two-stage pipeline with each entry and swaps them, which is
the proof the seam is real. Adding a third estimator is one @register function;
model.py and train.py do not change.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier


@runtime_checkable
class StageClassifier(Protocol):
    """What a stage estimator must provide. This is the sklearn classifier
    contract, written down so the boundary is explicit and checkable."""

    classes_: np.ndarray

    def fit(self, X: np.ndarray, y: np.ndarray) -> Any: ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


Factory = Callable[..., StageClassifier]
_FACTORIES: dict[str, Factory] = {}


def register(name: str) -> Callable[[Factory], Factory]:
    def wrap(factory: Factory) -> Factory:
        if name in _FACTORIES:
            raise ValueError(f"stage estimator {name!r} is already registered")
        _FACTORIES[name] = factory
        return factory

    return wrap


def available() -> list[str]:
    return sorted(_FACTORIES)


def make_stage_estimator(name: str, **params: Any) -> StageClassifier:
    """Build a registered stage estimator. Unknown names fail loudly with the
    list of what exists, because a typo in a config must not train silently."""
    try:
        factory = _FACTORIES[name]
    except KeyError:
        raise ValueError(
            f"unknown stage estimator {name!r}; available: {', '.join(available())}"
        ) from None
    return factory(**params)


@register("random_forest")
def _random_forest(**params: Any) -> StageClassifier:
    defaults: dict[str, Any] = {"n_jobs": -1, "class_weight": "balanced_subsample"}
    return RandomForestClassifier(**{**defaults, **params})


@register("hist_gradient_boosting")
def _hist_gradient_boosting(**params: Any) -> StageClassifier:
    return HistGradientBoostingClassifier(**params)

"""
MLflow pyfunc wrapper for the comparison arms that are not sklearn estimators.

The two tuned boosters (the repo's EncodedXGBClassifier and LightGBM) have
predict_proba but no sklearn-complete interface, so they cannot sit at the end of
a sklearn Pipeline for mlflow's sklearn flavor. This wraps imputer + model as a
pyfunc whose predict returns the class-probability matrix, columns in
`classes` order. Imported only by scripts/track_arms.py, never by serving.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from mlflow.pyfunc.model import PythonModel


class ImputedProbaModel(PythonModel):
    def __init__(self, imputer: Any, model: Any) -> None:
        self.imputer = imputer
        self.model = model
        self.classes = [str(c) for c in model.classes_]

    # left unannotated on purpose: mlflow reads type hints on predict as an input
    # schema and warns on anything but list[...]
    def predict(self, context, model_input, params=None):
        X = np.asarray(model_input, dtype=float)
        return np.asarray(self.model.predict_proba(self.imputer.transform(X)))

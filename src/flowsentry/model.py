"""
Two-stage hierarchical classifier with a tunable reject (abstain) option.

This is the deployable form of the architecture in Sepehr Jafari's SECRYPT 2026
paper (hierarchical UDP/QUIC intrusion detection with a reject option). Stage 1 is
a lightweight model on a cheap feature subset; low-confidence flows are escalated
to a stronger Stage 2 model. A separate reject threshold lets the system abstain
("unknown") rather than emit a low-confidence guess. Sweeping that threshold gives
the coverage-vs-reliability curve, which is the point of the reject option: you
trade how many flows you answer for how reliable those answers are.

Note on data: for this public benchmark reproduction the two "stages" run on
feature subsets of the same NSL-KDD vector. flowsentry.train picks stage 1's subset
explicitly by column name (11 cheap numeric flow stats - packet/count/rate fields
that need no payload or session inspection; see docs/MODEL_CARD.md), passed in via
the stage1_features argument below. This class's own stage1_features=None fallback
(first-half positional slice) only fires if a caller does not specify a subset; it
is a generic default for library/test use, not a claim about which columns are cheap.
In the headline system the stages are genuinely different feature sets (UDP-only vs
QUIC-augmented) produced by Sepehr's own UDPFlowLyzer / QUICFlowLyzer extractors.
See docs/MODEL_CARD.md.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

UNKNOWN = "unknown"


class TwoStageRejectClassifier(BaseEstimator):
    def __init__(
        self,
        stage1_features: list[int] | None = None,
        escalate_threshold: float = 0.90,
        n_estimators_stage1: int = 60,
        n_estimators_stage2: int = 200,
        random_state: int = 42,
    ) -> None:
        self.stage1_features = stage1_features
        self.escalate_threshold = escalate_threshold
        self.n_estimators_stage1 = n_estimators_stage1
        self.n_estimators_stage2 = n_estimators_stage2
        self.random_state = random_state

    def fit(self, X, y) -> TwoStageRejectClassifier:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        if self.stage1_features is None:
            # No explicit stage1_features given: fall back to a naive positional slice
            # (first half of the feature vector). This is NOT a claim that those columns
            # are cheap or numeric - it is just a default so the class is usable without
            # a caller who knows the feature layout. Callers who care which columns stage 1
            # sees (e.g. flowsentry.train, which wants real cheap numeric flow stats) must
            # pass stage1_features explicitly.
            self.stage1_features_ = list(range(max(1, n_features // 2)))
        else:
            self.stage1_features_ = list(self.stage1_features)
        self.stage1_ = RandomForestClassifier(
            n_estimators=self.n_estimators_stage1,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        self.stage2_ = RandomForestClassifier(
            n_estimators=self.n_estimators_stage2,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        self.stage1_.fit(X[:, self.stage1_features_], y)
        self.stage2_.fit(X, y)
        return self

    def _stage_predict(self, X):
        """Return (labels, confidence, escalated_mask) with two-stage escalation applied."""
        X = np.asarray(X, dtype=float)
        p1 = self.stage1_.predict_proba(X[:, self.stage1_features_])
        conf1 = p1.max(axis=1)
        pred1 = self.stage1_.classes_[p1.argmax(axis=1)]
        escalate = conf1 < self.escalate_threshold

        labels = np.array(pred1, dtype=object)
        conf = conf1.astype(float).copy()
        if escalate.any():
            p2 = self.stage2_.predict_proba(X[escalate])
            labels[escalate] = self.stage2_.classes_[p2.argmax(axis=1)]
            conf[escalate] = p2.max(axis=1)
        return labels, conf, escalate

    def predict(self, X, reject_threshold: float = 0.0):
        labels, conf, _ = self._stage_predict(X)
        out = labels.copy()
        if reject_threshold > 0:
            out[conf < reject_threshold] = UNKNOWN
        return out

    def predict_detail(self, X, reject_threshold: float = 0.0):
        """Return (labels, confidence, escalated_mask, abstained_mask)."""
        labels, conf, escalate = self._stage_predict(X)
        out = labels.copy()
        abstained = conf < reject_threshold
        out[abstained] = UNKNOWN
        return out, conf, escalate, abstained

    def coverage_reliability_curve(self, X, y, thresholds):
        """For each reject threshold: coverage (fraction answered) and reliability
        (accuracy on the answered subset). Also reports the escalation rate."""
        labels, conf, escalate = self._stage_predict(X)
        y = np.asarray(y)
        rows = []
        for t in thresholds:
            covered = conf >= t
            n_cov = int(covered.sum())
            reliability = float((labels[covered] == y[covered]).mean()) if n_cov else float("nan")
            rows.append(
                {
                    "threshold": round(float(t), 4),
                    "coverage": round(float(covered.mean()), 4),
                    "reliability": round(reliability, 4) if n_cov else None,
                    "n_covered": n_cov,
                    "escalation_rate": round(float(escalate.mean()), 4),
                }
            )
        return rows

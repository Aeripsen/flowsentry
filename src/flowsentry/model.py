"""
Two-stage hierarchical classifier with a tunable reject (abstain) option.

This is the deployable form of the architecture in Sepehr Jafari's SECRYPT 2026
paper (hierarchical UDP/QUIC intrusion detection with a reject option), trained on
the paper's own BCCC-UDP-QUIC-IDS-2025 dataset.

  Stage 1  is a mid-size random forest trained ONLY on the UDP flow statistics
           (data.STAGE1_FEATURES, the 114 UDPFlowLyzer features). It is the cheap,
           always-available layer: every UDP flow has these features.
  Stage 2  is a larger random forest trained on the QUIC-augmented feature space
           (data.STAGE2_FEATURES = the 114 UDP features + 18 QUICFlowLyzer
           features). It is invoked ONLY as a fallback for flows whose Stage-1
           confidence is below `escalate_threshold`.

`stage1_features` is the list of column indices, within the Stage-2 feature matrix,
that make up the UDP-only view. The training code passes data.STAGE1_INDICES
(0..113), i.e. the real UDP columns by position, NOT an arbitrary slice. A separate
`reject_threshold` lets the system abstain ("unknown") rather than emit a
low-confidence guess; sweeping it gives the coverage-vs-reliability curve, which is
the point of the reject option.
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
        # stage1_features: indices (within the full Stage-2 matrix) of the UDP-only
        # columns Stage 1 trains on. None means Stage 1 sees the full feature set
        # (degenerate single-stage); training always passes the real UDP indices.
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
            self.stage1_features_ = list(range(n_features))
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
        """Return (labels, confidence, escalated_mask, proba) with two-stage
        escalation applied. `proba` is the class-probability matrix (columns follow
        self.classes_): Stage-1 rows for confident flows, Stage-2 rows for escalated
        ones, so it is a faithful basis for PR-AUC / ranking metrics."""
        X = np.asarray(X, dtype=float)
        classes = list(self.classes_)
        p1 = self.stage1_.predict_proba(X[:, self.stage1_features_])
        conf1 = p1.max(axis=1)
        pred1 = self.stage1_.classes_[p1.argmax(axis=1)]
        escalate = conf1 < self.escalate_threshold

        proba = np.zeros((X.shape[0], len(classes)), dtype=float)
        for j, c in enumerate(self.stage1_.classes_):
            proba[:, classes.index(c)] = p1[:, j]

        labels = np.array(pred1, dtype=object)
        conf = conf1.astype(float).copy()
        if escalate.any():
            p2 = self.stage2_.predict_proba(X[escalate])
            labels[escalate] = self.stage2_.classes_[p2.argmax(axis=1)]
            conf[escalate] = p2.max(axis=1)
            block = np.zeros((int(escalate.sum()), len(classes)), dtype=float)
            for j, c in enumerate(self.stage2_.classes_):
                block[:, classes.index(c)] = p2[:, j]
            proba[escalate] = block
        return labels, conf, escalate, proba

    def predict(self, X, reject_threshold: float = 0.0):
        labels, conf, _, _ = self._stage_predict(X)
        out = labels.copy()
        if reject_threshold > 0:
            out[conf < reject_threshold] = UNKNOWN
        return out

    def predict_proba(self, X):
        """Two-stage class probabilities (columns follow self.classes_)."""
        _, _, _, proba = self._stage_predict(X)
        return proba

    def predict_detail(self, X, reject_threshold: float = 0.0):
        """Return (labels, confidence, escalated_mask, abstained_mask)."""
        labels, conf, escalate, _ = self._stage_predict(X)
        out = labels.copy()
        abstained = conf < reject_threshold
        out[abstained] = UNKNOWN
        return out, conf, escalate, abstained

    def coverage_reliability_curve(self, X, y, thresholds):
        """For each reject threshold: coverage (fraction answered) and reliability
        (accuracy on the answered subset). Also reports the escalation rate."""
        labels, conf, escalate, _ = self._stage_predict(X)
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

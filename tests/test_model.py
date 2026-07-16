"""Model tests.

Two clearly separated layers:
  * fast SYNTHETIC smoke tests (make_classification) for the reject/escalation logic,
  * a REAL-DATA metrics regression + leakage guard on the committed BCCC sample, which
    is the test that actually protects the headline claims.
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score

from flowsentry.data import (
    STAGE1_INDICES,
    build_matrices,
    leakage_safe_split,
    load_sample,
)
from flowsentry.model import UNKNOWN, TwoStageRejectClassifier

# --------------------------------------------------------------------------- #
# Synthetic smoke tests (no data files, fast)
# --------------------------------------------------------------------------- #


def _synthetic():
    X, y = make_classification(
        n_samples=500, n_features=24, n_informative=12, n_classes=3, random_state=0
    )
    return X, y.astype(str)


def test_fit_predict_shapes():
    X, y = _synthetic()
    model = TwoStageRejectClassifier().fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset(set(y))


def test_reject_emits_unknown_at_high_threshold():
    X, y = _synthetic()
    model = TwoStageRejectClassifier().fit(X, y)
    preds = model.predict(X, reject_threshold=0.999)
    assert (preds == UNKNOWN).any(), "a strict reject threshold should abstain on some flows"


def test_coverage_decreases_with_threshold():
    X, y = _synthetic()
    model = TwoStageRejectClassifier().fit(X, y)
    curve = model.coverage_reliability_curve(X, y, [0.0, 0.5, 0.9, 0.99])
    coverage = [r["coverage"] for r in curve]
    assert coverage == sorted(coverage, reverse=True)


# --------------------------------------------------------------------------- #
# Real-data metrics regression + leakage guard (committed BCCC sample)
# --------------------------------------------------------------------------- #


def test_split_is_connection_leakage_safe():
    df = load_sample()
    _, _, groups = build_matrices(df)
    tr, te = leakage_safe_split(groups, test_size=0.25, seed=42)
    # disjoint rows and, critically, disjoint connections
    assert set(tr).isdisjoint(set(te))
    assert set(groups[tr]).isdisjoint(set(groups[te])), "a connection leaked across the split"


def test_real_binary_detection_pr_auc_regression():
    """Two-stage model on the real sample must separate benign vs attack well.
    Locks the headline binary PR-AUC above a floor so a regression fails CI."""
    df = load_sample()
    X, y, groups = build_matrices(df)
    tr, te = leakage_safe_split(groups, test_size=0.25, seed=42)
    imp = SimpleImputer(strategy="median").fit(X[tr])
    Xtr, Xte = imp.transform(X[tr]), imp.transform(X[te])

    model = TwoStageRejectClassifier(stage1_features=STAGE1_INDICES).fit(Xtr, y[tr])
    proba = model.predict_proba(Xte)
    classes = list(model.classes_)
    p_benign = proba[:, classes.index("benign")]
    is_attack = (y[te] != "benign").astype(int)
    pr_auc = average_precision_score(is_attack, 1.0 - p_benign)
    assert pr_auc > 0.90, f"binary attack PR-AUC regressed to {pr_auc:.4f}"


def test_stage1_uses_real_udp_feature_indices():
    """Guard against a silent regression to a positional slice: Stage 1 must train on
    the named UDP feature indices (0..113), not half the vector."""
    assert STAGE1_INDICES == list(range(114))
    df = load_sample()
    X, y, groups = build_matrices(df)
    tr, _ = leakage_safe_split(groups, seed=42)
    imp = SimpleImputer(strategy="median").fit(X[tr])
    Xtr = imp.transform(X[tr])
    model = TwoStageRejectClassifier(stage1_features=STAGE1_INDICES).fit(Xtr, y[tr])
    assert model.stage1_features_ == list(range(114))
    assert model.stage1_.n_features_in_ == 114  # UDP-only
    assert model.stage2_.n_features_in_ == np.asarray(X).shape[1]  # UDP + QUIC


def test_dashboard_slice_is_the_heldout_test_split():
    """Guard against the dashboard reporting training rows as 'test flows'.
    `load_test_stream` must return exactly the leakage-safe held-out test rows the
    artifact persists (`test_indices`), NOT head(n) of the shuffled full sample."""
    from flowsentry.stream import load_test_stream

    df = load_sample()
    _, y, groups = build_matrices(df)
    tr, te = leakage_safe_split(groups, test_size=0.25, seed=42)
    bundle = {"test_indices": [int(i) for i in te]}
    _, truth = load_test_stream(bundle, 0)
    # exactly the held-out rows, same order
    assert list(truth) == list(y[te])
    # and their connections are disjoint from train -> it really is the test split,
    # so a reliability number computed from this cannot include training rows
    assert set(groups[te]).isdisjoint(set(groups[tr]))

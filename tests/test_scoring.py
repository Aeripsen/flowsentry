"""Scoring-path tests. The fast paths in scoring.py/model.py exist only because
they are provably equivalent to the sklearn-native paths, so most of this file is
exact-equality guards. The tail is a performance guard with generous bounds: it is
not a micro-benchmark, it exists to fail if the per-call thread-pool regression
(the old ~45 ms/flow serving path) ever comes back.
"""
import time

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer

from flowsentry.data import (
    STAGE1_INDICES,
    build_matrices,
    leakage_safe_split,
    load_sample,
)
from flowsentry.model import TwoStageRejectClassifier, forest_proba
from flowsentry.scoring import FlowScorer

# --------------------------------------------------------------------------- #
# Fast synthetic equality guards
# --------------------------------------------------------------------------- #


def _synthetic_fitted():
    X, y = make_classification(
        n_samples=400, n_features=24, n_informative=12, n_classes=3, random_state=0
    )
    # inject missing values so imputation is actually exercised
    rng = np.random.RandomState(1)
    Xm = X.copy()
    Xm[rng.rand(*X.shape) < 0.05] = np.nan
    imputer = SimpleImputer(strategy="median").fit(Xm)
    model = TwoStageRejectClassifier(
        n_estimators_stage1=20, n_estimators_stage2=30
    ).fit(imputer.transform(Xm), y.astype(str))
    return imputer, model, Xm


def test_sequential_forest_proba_bit_identical():
    """The sequential tree walk must return EXACTLY what sklearn returns with
    n_jobs=1 (same accumulation order, same dtype conversion). Not allclose:
    array_equal. If this ever fails the fast path must be deleted, not tuned."""
    imputer, model, Xm = _synthetic_fitted()
    Xi = imputer.transform(Xm)
    for forest, cols in ((model.stage1_, model.stage1_features_), (model.stage2_, None)):
        Xf = Xi[:, cols] if cols is not None else Xi
        forest.n_jobs = 1
        ref = forest.predict_proba(Xf)
        fast = forest_proba(forest, Xf, sequential=True)
        assert np.array_equal(fast, ref)


def test_sequential_predict_detail_matches_native():
    imputer, model, Xm = _synthetic_fitted()
    Xi = imputer.transform(Xm)
    l_seq, c_seq, e_seq, a_seq = model.predict_detail(
        Xi, reject_threshold=0.6, sequential=True
    )
    l_nat, c_nat, e_nat, a_nat = model.predict_detail(
        Xi, reject_threshold=0.6, sequential=False
    )
    assert np.array_equal(l_seq, l_nat)
    assert np.allclose(c_seq, c_nat, atol=1e-12)
    assert np.array_equal(e_seq, e_nat)
    assert np.array_equal(a_seq, a_nat)


def test_fast_impute_exactly_equals_simpleimputer():
    imputer, _, Xm = _synthetic_fitted()
    scorer_medians = np.asarray(imputer.statistics_)
    fast = np.where(np.isnan(Xm), scorer_medians, Xm)
    assert np.array_equal(fast, imputer.transform(Xm))


def test_score_batch_equals_score_one():
    """The two public entry points must agree row for row."""
    imputer, model, Xm = _synthetic_fitted()
    names = [f"f{i}" for i in range(Xm.shape[1])]
    scorer = FlowScorer(imputer, model, feature_names=names)
    rows = Xm[:20]
    labels, conf, esc, abst = scorer.score_batch(rows, reject_threshold=0.5)
    for i in range(rows.shape[0]):
        feats = {
            n: float(v) for n, v in zip(names, rows[i], strict=True) if not np.isnan(v)
        }
        one = scorer.score_one(feats, reject_threshold=0.5)
        assert one["label"] == str(labels[i])
        assert one["confidence"] == round(float(conf[i]), 4)
        assert one["escalated_to_stage2"] == bool(esc[i])
        assert one["abstained"] == bool(abst[i])


def test_scorer_rejects_mismatched_artifact():
    imputer, model, Xm = _synthetic_fitted()
    with pytest.raises(ValueError, match="artifact and schema disagree"):
        FlowScorer(imputer, model, feature_names=["only", "two"])


# --------------------------------------------------------------------------- #
# Real-data equivalence + performance guard (shared fitted model, one fit)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def real_fitted():
    df = load_sample()
    X, y, groups = build_matrices(df)
    tr, te = leakage_safe_split(groups, test_size=0.25, seed=42)
    imputer = SimpleImputer(strategy="median").fit(X[tr])
    model = TwoStageRejectClassifier(stage1_features=STAGE1_INDICES).fit(
        imputer.transform(X[tr]), y[tr]
    )
    return FlowScorer(imputer, model), X[te]


def test_real_sequential_labels_match_native(real_fitted):
    """On real flows, the fast path and the native path must give the same
    verdicts. Confidence may differ in the last float bit (threaded summation
    order), which is why labels are exact and confidences are 1e-12-close."""
    scorer, Xte = real_fitted
    Xi = scorer.impute(Xte[:2000])
    l_seq, c_seq, _, _ = scorer.model.predict_detail(Xi, sequential=True)
    l_nat, c_nat, _, _ = scorer.model.predict_detail(Xi, sequential=False)
    assert np.array_equal(l_seq, l_nat)
    assert np.allclose(c_seq, c_nat, atol=1e-12)


def test_perf_guard_single_row(real_fitted):
    """Fails if the serving path regresses toward the old ~45 ms/flow thread-pool
    behavior. Bound is generous (measured ~2-6 ms on the dev machine) so a slow
    CI runner passes, but a per-call pool spin-up (mean 44.7 ms, p50 29.8 in
    artifacts/benchmark.json:single_row_native_pool) cannot."""
    scorer, Xte = real_fitted
    lat = []
    scorer.score_batch(Xte[:1])  # warmup
    for i in range(50):
        row = Xte[i : i + 1]
        t0 = time.perf_counter()
        scorer.score_batch(row)
        lat.append((time.perf_counter() - t0) * 1000.0)
    p95 = float(np.percentile(lat, 95))
    assert p95 < 30.0, f"single-row p95 regressed to {p95:.1f} ms"


def test_perf_guard_batch_throughput(real_fitted):
    """Batch scoring must stay in bulk territory. The committed benchmark brackets
    this size at 34,398 flows/s (1,024 rows) and 59,895 flows/s (8,192 rows), see
    artifacts/benchmark.json:batch; the 2,000 floor only catches an
    order-of-magnitude regression, not machine variance."""
    scorer, Xte = real_fitted
    n = min(6000, Xte.shape[0])
    scorer.score_batch(Xte[:64])  # warmup
    t0 = time.perf_counter()
    scorer.score_batch(Xte[:n])
    wall = time.perf_counter() - t0
    flows_per_s = n / wall
    assert flows_per_s > 2000, f"batch throughput regressed to {flows_per_s:,.0f} flows/s"

from sklearn.datasets import make_classification

from flowsentry.model import UNKNOWN, TwoStageRejectClassifier


def _data():
    X, y = make_classification(
        n_samples=500, n_features=24, n_informative=12, n_classes=3, random_state=0
    )
    return X, y.astype(str)


def test_fit_predict_shapes():
    X, y = _data()
    model = TwoStageRejectClassifier().fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset(set(y))


def test_reject_emits_unknown_at_high_threshold():
    X, y = _data()
    model = TwoStageRejectClassifier().fit(X, y)
    preds = model.predict(X, reject_threshold=0.999)
    assert (preds == UNKNOWN).any(), "a strict reject threshold should abstain on some flows"


def test_coverage_decreases_with_threshold():
    X, y = _data()
    model = TwoStageRejectClassifier().fit(X, y)
    curve = model.coverage_reliability_curve(X, y, [0.0, 0.5, 0.9, 0.99])
    coverage = [r["coverage"] for r in curve]
    assert coverage == sorted(coverage, reverse=True)


def test_reliability_beats_or_matches_full_coverage():
    # tightening the reject threshold should not lower reliability on average
    X, y = _data()
    model = TwoStageRejectClassifier().fit(X, y)
    curve = model.coverage_reliability_curve(X, y, [0.0, 0.9])
    assert curve[-1]["reliability"] >= curve[0]["reliability"] - 1e-9

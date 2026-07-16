"""Drift tests: the PSI surface must stay quiet on same-distribution data, fire on
a real shift, rank the shifted feature first, and survive degenerate columns."""
import json

import numpy as np
import pytest

from flowsentry.drift import band, drift_report, psi, reference_from_matrix


def _two_halves(seed: int = 0):
    """Two halves of one synthetic distribution: 3 continuous features with
    different scales plus one binary flag and one constant column, the shapes
    real flow features come in."""
    rng = np.random.RandomState(seed)
    n = 4000
    X = np.column_stack(
        [
            rng.lognormal(3.0, 1.0, n),        # bytes-like, heavy tail
            rng.normal(0.0, 1.0, n),           # centered stat
            rng.exponential(0.01, n),          # iat-like
            (rng.rand(n) < 0.2).astype(float), # binary flag
            np.zeros(n),                       # constant
        ]
    )
    names = ["bytes_like", "centered", "iat_like", "flag", "constant"]
    return X[: n // 2], X[n // 2 :], names


def test_same_distribution_is_stable():
    a, b, names = _two_halves()
    ref = reference_from_matrix(a, names)
    scores = psi(ref, b, names)
    assert all(v < 0.10 for v in scores.values()), scores


def test_shift_fires_and_ranks_first():
    a, b, names = _two_halves()
    ref = reference_from_matrix(a, names)
    shifted = b.copy()
    shifted[:, 0] *= 10.0  # a 10x rate shift, the flood-onset shape
    report = drift_report(ref, shifted, names)
    assert report["top"][0]["feature"] == "bytes_like"
    assert report["top"][0]["band"] == "major"
    assert report["bands"]["major"] >= 1
    # the untouched features stay quiet
    scores = psi(ref, shifted, names)
    assert scores["centered"] < 0.10
    assert scores["flag"] < 0.10


def test_constant_feature_is_quiet_not_crashing():
    a, b, names = _two_halves()
    ref = reference_from_matrix(a, names)
    assert psi(ref, b, names)["constant"] == pytest.approx(0.0, abs=1e-6)


def test_reference_is_json_serializable():
    a, _, names = _two_halves()
    ref = reference_from_matrix(a, names)
    roundtrip = json.loads(json.dumps(ref))
    assert roundtrip["features"]["bytes_like"]["edges"] == ref["features"]["bytes_like"]["edges"]


def test_empty_window_raises():
    a, _, names = _two_halves()
    ref = reference_from_matrix(a, names)
    with pytest.raises(ValueError, match="empty window"):
        psi(ref, np.empty((0, len(names))), names)


def test_unknown_feature_raises():
    a, b, names = _two_halves()
    ref = reference_from_matrix(a, names)
    with pytest.raises(KeyError, match="missing from the drift reference"):
        psi(ref, b, ["not_a_feature"] + names[1:])


def test_band_thresholds():
    assert band(0.05) == "stable"
    assert band(0.10) == "moderate"
    assert band(0.25) == "moderate"
    assert band(0.26) == "major"

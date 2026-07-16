"""Service tests. A tiny in-memory model (no data files, no network, no trained
artifact on disk) is injected so the FastAPI wiring is exercised end to end."""
import numpy as np
from fastapi.testclient import TestClient
from sklearn.impute import SimpleImputer

from flowsentry import service
from flowsentry.data import STAGE1_INDICES, STAGE2_FEATURES, UDP_FEATURES
from flowsentry.model import TwoStageRejectClassifier
from flowsentry.scoring import FlowScorer


def _tiny_bundle():
    """Train a small real model on synthetic BCCC-shaped rows (132 features)."""
    rng = np.random.RandomState(0)
    n_feat = len(STAGE2_FEATURES)
    X = rng.rand(200, n_feat)
    # make the label depend on a UDP feature so the model learns a real boundary
    y = np.where(X[:, 0] > 0.5, "UDP-RAW", "benign")
    imputer = SimpleImputer(strategy="median").fit(X)
    model = TwoStageRejectClassifier(
        stage1_features=STAGE1_INDICES, n_estimators_stage1=10, n_estimators_stage2=10
    ).fit(imputer.transform(X), y)
    return {"imputer": imputer, "model": model, "stage2_features": STAGE2_FEATURES}


def _tiny_scorer():
    return FlowScorer.from_bundle(_tiny_bundle())


def test_health_ok(monkeypatch):
    monkeypatch.setattr(service, "_scorer", _tiny_scorer())
    client = TestClient(service.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_503_when_artifact_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "_scorer", None)
    monkeypatch.setattr(service, "ARTIFACT", tmp_path / "no-such-model.joblib")
    client = TestClient(service.app)
    resp = client.get("/health")
    assert resp.status_code == 503  # not a falsely-green 200


def test_predict_returns_label(monkeypatch):
    monkeypatch.setattr(service, "_scorer", _tiny_scorer())
    client = TestClient(service.app)
    resp = client.post(
        "/predict",
        json={"features": {UDP_FEATURES[0]: 0.9, "pkt_count": 1200, "has_quic_subflow": 0}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "label" in body and "confidence" in body
    assert 0.0 <= body["confidence"] <= 1.0
    assert {"escalated_to_stage2", "abstained"} <= set(body)


def test_predict_reject_threshold_abstains(monkeypatch):
    monkeypatch.setattr(service, "_scorer", _tiny_scorer())
    client = TestClient(service.app)
    payload = {"features": {UDP_FEATURES[0]: 0.55, "pkt_count": 500}}

    base = client.post("/predict", json={**payload, "reject_threshold": 0.0}).json()
    assert base["label"] != "unknown"

    hi = client.post("/predict", json={**payload, "reject_threshold": 1.0}).json()
    if base["confidence"] < 1.0:
        assert hi["label"] == "unknown"
    else:
        assert hi["label"] == base["label"]

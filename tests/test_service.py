import pandas as pd
from fastapi.testclient import TestClient

from flowsentry import service
from flowsentry.data import COLUMNS, build_preprocessor
from flowsentry.model import TwoStageRejectClassifier


def _tiny_bundle():
    """Train a small real model on a handful of valid NSL-KDD-shaped rows (no network)."""
    rows = []
    for i in range(60):
        row = {c: 0 for c in COLUMNS}
        row["protocol_type"] = ["tcp", "udp", "icmp"][i % 3]
        row["service"] = ["http", "private", "domain_u"][i % 3]
        row["flag"] = ["SF", "S0", "REJ"][i % 3]
        row["src_bytes"] = i * 11
        row["count"] = i
        row["category"] = ["normal", "dos", "probe"][i % 3]
        rows.append(row)
    df = pd.DataFrame(rows)
    pre = build_preprocessor()
    X = pre.fit_transform(df)
    model = TwoStageRejectClassifier(n_estimators_stage1=10, n_estimators_stage2=10).fit(
        X, df["category"].to_numpy()
    )
    return {"preprocessor": pre, "model": model, "feature_names": list(pre.get_feature_names_out())}


def test_health_ok():
    client = TestClient(service.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] in {"ok", "degraded"}


def test_predict_returns_label(monkeypatch):
    monkeypatch.setattr(service, "_bundle", _tiny_bundle())
    client = TestClient(service.app)
    resp = client.post(
        "/predict",
        json={"features": {"protocol_type": "tcp", "service": "http", "flag": "SF",
                           "src_bytes": 181, "count": 8}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "label" in body and "confidence" in body
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_reject_threshold_plumbing(monkeypatch):
    monkeypatch.setattr(service, "_bundle", _tiny_bundle())
    client = TestClient(service.app)
    payload = {"features": {"protocol_type": "udp", "service": "private", "flag": "S0"}}

    base = client.post("/predict", json={**payload, "reject_threshold": 0.0}).json()
    assert base["label"] != "unknown"

    hi = client.post("/predict", json={**payload, "reject_threshold": 1.0}).json()
    if base["confidence"] < 1.0:
        # confidence below the bar -> abstain
        assert hi["label"] == "unknown"
    else:
        # a unanimous forest is certain; you cannot reject certainty
        assert hi["label"] == base["label"]

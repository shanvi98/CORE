"""Tests for app.py.

Model-dependent tests are skipped with a clear reason if artifacts/ hasn't
been produced yet (run `python train.py` first).
"""
from __future__ import annotations

import importlib
import json

import pytest
from fastapi.testclient import TestClient

from src import config

ARTIFACTS_PRESENT = config.MODEL_PATH.exists() and config.META_PATH.exists()

VALID_ORDER = {
    "order_id": "TEST0001",
    "order_placed_at": "2026-07-01 00:51",
    "customer_id": "KC104433",
    "sku": "KH-AF-01",
    "sales_channel": "app",
    "payment_mode": "cod",
    "discount_pct": 10,
    "qty": 1,
    "order_value_inr": 4679.1,
    "promised_delivery_days": 9,
    "delivery_pincode": "440365",
    "is_gift": "N",
    "customer_prior_orders": 0,
    "customer_prior_returns": 0,
    "delivery_note": "Call before delivery",
}


@pytest.fixture()
def client():
    import app as app_module

    importlib.reload(app_module)
    with TestClient(app_module.app) as c:
        yield c


@pytest.mark.skipif(not ARTIFACTS_PRESENT, reason="artifacts/ missing — run `python train.py` first")
def test_health_ok_when_artifacts_exist(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"]


@pytest.mark.skipif(not ARTIFACTS_PRESENT, reason="artifacts/ missing — run `python train.py` first")
def test_predict_happy_path(client):
    r = client.post("/predict", json=VALID_ORDER)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["score"] <= 1.0
    assert len(body["reasons"]) == 3
    assert body["risk_band"] in {"low", "medium", "high"}
    if body["score"] >= body["threshold"]:
        assert body["recommended_action"] == "CALL_BEFORE_DISPATCH"
    else:
        assert body["recommended_action"] == "DISPATCH"
    assert body["explanation_source"] == "template"  # no API key in test env


@pytest.mark.skipif(not ARTIFACTS_PRESENT, reason="artifacts/ missing — run `python train.py` first")
def test_leaky_field_is_ignored_and_does_not_change_score(client):
    base = client.post("/predict", json=VALID_ORDER).json()
    with_leak = dict(VALID_ORDER, pickup_scheduled_at="2026-07-02 10:00")
    leaked = client.post("/predict", json=with_leak).json()
    assert leaked["score"] == base["score"]
    assert any("pickup_scheduled_at" in w for w in leaked["warnings"])


@pytest.mark.skipif(not ARTIFACTS_PRESENT, reason="artifacts/ missing — run `python train.py` first")
def test_unknown_customer_returns_200_with_warning(client):
    order = dict(VALID_ORDER, customer_id="KC000000_NOPE")
    r = client.post("/predict", json=order)
    assert r.status_code == 200
    assert any("not found" in w for w in r.json()["warnings"])


@pytest.mark.skipif(not ARTIFACTS_PRESENT, reason="artifacts/ missing — run `python train.py` first")
def test_invalid_payment_mode_returns_422_with_field_list(client):
    order = dict(VALID_ORDER, payment_mode="bitcoin")
    r = client.post("/predict", json=order)
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "invalid_input"
    assert any(f["field"] == "payment_mode" for f in body["fields"])


def test_model_missing_starts_app_and_degrades(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "model.txt")
    monkeypatch.setattr(config, "META_PATH", tmp_path / "model_meta.json")
    monkeypatch.setattr(config, "CALIBRATOR_PATH", tmp_path / "calibrator.pkl")

    import app as app_module

    importlib.reload(app_module)
    with TestClient(app_module.app) as c:
        health = c.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "degraded"

        r = c.post("/predict", json=VALID_ORDER)
        assert r.status_code == 503
        assert r.json()["error"] == "model_unavailable"

    # restore for any tests that run after this one in the same session
    importlib.reload(config)
    importlib.reload(app_module)


def test_no_api_key_uses_template_explanations(monkeypatch, client):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    if not ARTIFACTS_PRESENT:
        pytest.skip("artifacts/ missing — run `python train.py` first")
    r = client.post("/predict", json=VALID_ORDER)
    assert r.json()["explanation_source"] == "template"

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

import modelops.serving.api as api

client = TestClient(api.app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    body = response.json()
    assert isinstance(body["prediction"], str)
    assert body["prediction"] in {"setosa", "versicolor", "virginica"}

    probabilities = body["probabilities"]
    assert set(probabilities) == {"setosa", "versicolor", "virginica"}
    assert all(0.0 <= probability <= 1.0 for probability in probabilities.values())
    assert abs(sum(probabilities.values()) - 1.0) < 1e-6


def test_predict_rejects_invalid_measurements():
    payload = {
        "sepal_length": -1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_returns_503_when_model_is_missing(monkeypatch, tmp_path):
    api.load_model.cache_clear()

    monkeypatch.setattr(api, "ROOT", tmp_path)
    monkeypatch.setattr(
        api,
        "load_config",
        lambda: {"model": {"path": "missing_model.joblib"}},
    )

    try:
        response = client.post(
            "/predict",
            json={
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            },
        )

        assert response.status_code == 503
        assert response.json()["detail"] == (
            "Model is not available. Train and register a model first."
        )
    finally:
        api.load_model.cache_clear()

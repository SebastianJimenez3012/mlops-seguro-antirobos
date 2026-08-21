"""Tests del contrato HTTP de la API."""
import numpy as np
from fastapi.testclient import TestClient

import api.app as app_module


class FakeModel:
    def predict_proba(self, frame):
        return np.array([[0.20, 0.80]])


client = TestClient(app_module.app)


def payload():
    return {
        "Mto_TC": 12000,
        "MARCA": "Visa",
        "Nombre_territorio": "T.CENTRO",
        "FLAG_LIMA_PROVINCIA": 1,
        "REGION": "LIMA MODERNA",
        "SUELDO_ESTIMADO": 4000,
        "EDAD": 35,
        "SEXO": "F",
        "ANTIGUEDAD_MES": 80,
        "SEGMENTO": "CLASICO",
        "FLAG_UNICEF": 0,
    }


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_contract(monkeypatch):
    monkeypatch.setattr(app_module.predictor, "_model", FakeModel())
    response = client.post("/predict", json=payload())
    assert response.status_code == 200
    body = response.json()
    assert body["probability"] == 0.8
    assert body["prediction"] == 1
    assert body["priority"] == "alta"


def test_invalid_age_is_rejected():
    bad = payload()
    bad["EDAD"] = 150
    response = client.post("/predict", json=bad)
    assert response.status_code == 422

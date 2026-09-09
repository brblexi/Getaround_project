"""Tests for the prediction API.

Run from `api/`:  pytest -q
Requires: pytest, httpx (add them to a requirements-dev.txt).
"""

import pytest
from fastapi.testclient import TestClient

from app import KNOWN_CATEGORIES, app

client = TestClient(app)

REFERENCE_CAR = {
    "model_key": "Citroën",
    "mileage": 140000,
    "engine_power": 100,
    "fuel": "diesel",
    "paint_color": "black",
    "car_type": "sedan",
    "private_parking_available": True,
    "has_gps": True,
    "has_air_conditioning": False,
    "automatic_car": False,
    "has_getaround_connect": True,
    "has_speed_regulator": True,
    "winter_tires": False,
}


def test_health_reports_a_loaded_model():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_returns_a_plausible_price():
    response = client.post("/predict", json=REFERENCE_CAR)
    assert response.status_code == 200

    price = response.json()["rental_price_per_day"]
    # Loose bounds: this asserts the pipeline is wired correctly, not that the
    # model scores well — that is what the metrics in MLflow are for.
    assert 20 < price < 500


def test_missing_field_is_rejected():
    payload = {k: v for k, v in REFERENCE_CAR.items() if k != "engine_power"}
    assert client.post("/predict", json=payload).status_code == 422


def test_impossible_numeric_value_is_rejected():
    assert client.post("/predict", json={**REFERENCE_CAR, "mileage": -5}).status_code == 422


def test_unknown_category_is_rejected_rather_than_scored():
    """The regression this guards against.

    `handle_unknown="ignore"` encodes an unseen category as all-zeros, so
    "Citroen" without the diaeresis used to return a confident, wrong price
    instead of an error.
    """
    response = client.post("/predict", json={**REFERENCE_CAR, "model_key": "Citroen"})
    assert response.status_code == 422
    assert "Citroën" in response.json()["detail"][0]["allowed_values"]


@pytest.mark.parametrize("column", ["model_key", "fuel", "paint_color", "car_type"])
def test_every_categorical_column_is_guarded(column):
    assert column in KNOWN_CATEGORIES
    assert KNOWN_CATEGORIES[column]

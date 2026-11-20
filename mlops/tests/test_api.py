import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import sys
from pathlib import Path
from fastapi import HTTPException

# Fix path to allow importing serve module directly
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    from serve import check_api_key, create_app, require_api_key
except ImportError:
    from mlops.serve import check_api_key, create_app, require_api_key

class FakeModel:
    """
    Fake model to simulate predictions without needing a live MLflow server.
    """
    def predict(self, features):
        # Feature format: [[claimed, measured, difference]]
        results = []
        for row in features:
            diff = row[2]
            results.append(1 if diff > 2.0 else 0)
        return results

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HIREWISE_ML_API_KEY", "test-api-key")
    # Application factory ke zariye FakeModel inject karke isolated TestClient banate hain
    reference = __import__("mlops.data", fromlist=["generate_sample_data"]).generate_sample_data()[["claimed_skill", "measured_ability", "difference"]]
    test_app = create_app(model_loader=lambda: FakeModel(), reference_loader=lambda: reference,
                          store_path=tmp_path / "test_predictions.sqlite")
    with TestClient(test_app, headers={"X-API-Key": "test-api-key"}) as test_client:
        yield test_client

def test_health_endpoint(client):
    """Health stays public and reveals liveness only."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_key_validator_reports_missing_invalid_and_unconfigured_keys():
    with pytest.raises(HTTPException) as missing:
        check_api_key(None, "configured")
    assert missing.value.status_code == 401
    with pytest.raises(HTTPException) as invalid:
        check_api_key("wrong", "configured")
    assert invalid.value.status_code == 401
    with pytest.raises(HTTPException) as unconfigured:
        check_api_key("provided", None)
    assert unconfigured.value.status_code == 503
    with pytest.raises(HTTPException) as blank:
        check_api_key("provided", "   ")
    assert blank.value.status_code == 503
    check_api_key("correct", "correct")


def test_api_key_dependency_is_attached_to_private_routes_only():
    test_app = create_app(model_loader=lambda: object(), reference_loader=lambda: None)
    routes = {route.path: route for route in test_app.routes if hasattr(route, "dependant")}
    for path in ("/predict", "/predict/batch", "/monitoring"):
        assert any(dependency.call is require_api_key
                   for dependency in routes[path].dependant.dependencies)
    assert not routes["/health"].dependant.dependencies


def test_prediction_and_monitoring_endpoints_require_api_key(client):
    requests = [
        ("/predict", {"claimed_skill": 8.0, "measured_ability": 4.0}),
        ("/predict/batch", {"items": [{"claimed_skill": 8.0, "measured_ability": 4.0}]}),
        ("/monitoring", None),
    ]
    for path, payload in requests:
        response = (client.post(path, json=payload, headers={"X-API-Key": ""})
                    if payload is not None else client.get(path, headers={"X-API-Key": ""}))
        assert response.status_code == 401


def test_prediction_and_monitoring_reject_invalid_api_key(client):
    headers = {"X-API-Key": "wrong-key"}
    assert client.post("/predict", json={"claimed_skill": 8, "measured_ability": 4},
                       headers=headers).status_code == 401
    assert client.post("/predict/batch", json={"items": [{"claimed_skill": 8, "measured_ability": 4}]},
                       headers=headers).status_code == 401
    assert client.get("/monitoring", headers=headers).status_code == 401


def test_protected_endpoints_fail_closed_when_api_key_is_unconfigured(client, monkeypatch):
    monkeypatch.delenv("HIREWISE_ML_API_KEY")
    response = client.post("/predict", json={"claimed_skill": 8, "measured_ability": 4},
                           headers={"X-API-Key": "anything"})
    assert response.status_code == 503


def test_unauthenticated_invalid_request_is_not_logged(client):
    response = client.post("/predict", json={}, headers={"X-API-Key": ""})
    assert response.status_code == 401
    report = client.get("/monitoring").json()
    assert report["prediction_count"] == 0

def test_predict_inflated_resume(client):
    """Verify inflated resume (gap > 2.0) returns inflated=True"""
    payload = {"claimed_skill": 9.0, "measured_ability": 4.5}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["inflated"] is True
    assert data["difference"] == 4.5
    assert data["model_name"] == "hirewise-inflation-detector"
    assert data["request_id"]

def test_predict_honest_resume(client):
    """Verify honest resume (gap <= 2.0) returns inflated=False"""
    payload = {"claimed_skill": 6.5, "measured_ability": 6.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["inflated"] is False
    assert data["difference"] == 0.5

def test_batch_prediction_endpoint(client):
    response = client.post("/predict/batch", json={"items": [
        {"claimed_skill": 9.0, "measured_ability": 4.0},
        {"claimed_skill": 6.0, "measured_ability": 5.5},
    ]})
    assert response.status_code == 200
    assert [item["inflated"] for item in response.json()] == [True, False]

def test_prediction_survives_telemetry_write_failure(client):
    with patch("serve.record_prediction", side_effect=OSError("read-only store")):
        response = client.post("/predict", json={"claimed_skill": 9.0, "measured_ability": 4.0})
    assert response.status_code == 200
    assert response.json()["inflated"] is True

def test_predict_invalid_missing_fields(client):
    """Verify 422 Unprocessable Entity when required field is missing"""
    response = client.post("/predict", json={"claimed_skill": 8.0})
    assert response.status_code == 422

def test_predict_invalid_field_types(client):
    """Verify 422 Unprocessable Entity when fields have invalid types"""
    response = client.post("/predict", json={"claimed_skill": "expert", "measured_ability": 5.0})
    assert response.status_code == 422

def test_predict_logs_candidate_id_free_features_and_monitoring(client):
    response = client.post("/predict", json={"claimed_skill": 8.0, "measured_ability": 4.0})
    assert response.status_code == 200
    report = client.get("/monitoring").json()
    assert report["prediction_count"] == 1
    assert report["class_distribution"] == {"1": 1}
    assert report["concept_drift"]["available"] is False

def test_health_does_not_disclose_model_version(client):
    assert client.get("/health").json() == {"status": "ok"}

def test_monitoring_flags_distribution_shift(tmp_path):
    from mlops.data import generate_sample_data
    from mlops.prediction_store import record_prediction, monitor_summary
    reference = generate_sample_data(random_state=4)[["claimed_skill", "measured_ability", "difference"]]
    store = tmp_path / "predictions.sqlite"
    for _ in range(30):
        record_prediction("demo", "1", 9.5, 2.0, 7.5, 1, 0.99, path=store)
    result = monitor_summary(reference, store)
    assert result["data_drift"]["available"] is True
    assert result["data_drift"]["warning"]

def test_invalid_input_is_logged(client):
    assert client.post("/predict", json={"claimed_skill": 12, "measured_ability": 4}).status_code == 422
    assert client.get("/monitoring").json()["failed_request_count"] == 1

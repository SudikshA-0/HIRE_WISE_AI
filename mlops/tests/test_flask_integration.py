from unittest.mock import Mock, patch

import requests
import pytest
from fastapi.testclient import TestClient
from mlops.data import generate_sample_data
from mlops.serve import create_app
from services.ml_inference_service import MLInferenceService, score_0_100_to_1_10


def test_flask_client_maps_scale_and_sends_only_numeric_payload(monkeypatch):
    monkeypatch.setenv("HIREWISE_ML_API_KEY", "test-api-key")
    assert score_0_100_to_1_10(76) == 7.84
    assert score_0_100_to_1_10(0) == 1.0
    assert score_0_100_to_1_10(100) == 10.0
    with pytest.raises(ValueError):
        score_0_100_to_1_10(101)
    response = Mock()
    response.json.return_value = [{"inflated": True, "difference": 2.16,
                                   "probability": .87, "model_name": "hirewise-inflation-detector",
                                   "model_version": "3"}]
    response.raise_for_status.return_value = None
    service = MLInferenceService(base_url="http://model", timeout=1)
    with patch("services.ml_inference_service.requests.post", return_value=response) as post:
        result = service.predict_skill_mismatches([{
            "skill_name": "Python", "claimed_level": "Expert", "estimated_score": 76.0,
        }])
    assert result["available"] is True
    assert result["predictions"][0]["model_version"] == "3"
    assert post.call_args.args[0] == "http://model/predict/batch"
    assert post.call_args.kwargs["json"] == {"items": [{"claimed_skill": 10.0, "measured_ability": 7.84}]}
    assert post.call_args.kwargs["headers"] == {"X-API-Key": "test-api-key"}


def test_flask_client_degrades_gracefully_when_service_is_down():
    service = MLInferenceService(base_url="http://model", timeout=.01)
    with patch("services.ml_inference_service.requests.post", side_effect=requests.Timeout):
        result = service.predict_skill_mismatches([
            {"skill_name": "Python", "claimed_level": "Expert", "estimated_score": 70},
        ])
    assert result["available"] is False
    assert result["predictions"] == []


def test_flask_to_fastapi_contract_integration(tmp_path, monkeypatch):
    monkeypatch.setenv("HIREWISE_ML_API_KEY", "integration-api-key")
    class FakeModel:
        def predict(self, rows):
            return [int(row[2] > 2.0) for row in rows]

    reference = generate_sample_data()[["claimed_skill", "measured_ability", "difference"]]
    app = create_app(model_loader=lambda: (FakeModel(), "integration-v1"),
                     reference_loader=lambda: reference,
                     store_path=tmp_path / "integration.sqlite")
    with TestClient(app) as fastapi_client:
        response = MockResponseAdapter(fastapi_client)
        service = MLInferenceService(base_url="http://in-process-model")
        with patch("services.ml_inference_service.requests.post", side_effect=response.post) as post:
            result = service.predict_skill_mismatches([{
                "skill_name": "Python", "claimed_level": "Expert", "estimated_score": 50,
            }])
        assert result["available"] is True
        assert result["predictions"][0]["inflated"] is True
        assert result["predictions"][0]["model_version"] == "integration-v1"
        assert post.call_args.args[0] == "http://in-process-model/predict/batch"


class MockResponseAdapter:
    def __init__(self, fastapi_client):
        self.fastapi_client = fastapi_client

    def post(self, url, json, timeout):
        assert url.endswith("/predict/batch")
        response = self.fastapi_client.post("/predict/batch", json=json)
        response.raise_for_status()
        return response

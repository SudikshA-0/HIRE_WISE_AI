"""Optional HTTP client for the separately deployed HireWise ML service."""
from __future__ import annotations

import os
import math

import requests

CLAIMED_LEVEL_SCALE = {"Beginner": 3.0, "Intermediate": 6.0, "Advanced": 8.0, "Expert": 10.0}


def score_0_100_to_1_10(score):
    """Convert existing percentage scores while keeping the model input in range."""
    value = float(score)
    if not 0 <= value <= 100:
        raise ValueError("Interview skill score must be between 0 and 100.")
    return round(1.0 + (value / 100.0) * 9.0, 2)


class MLInferenceService:
    def __init__(self, base_url=None, timeout=None):
        self.base_url = (base_url or os.environ.get("HIREWISE_ML_SERVICE_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.timeout = float(timeout or os.environ.get("HIREWISE_ML_TIMEOUT_SECONDS", "2"))

    def predict_skill_mismatches(self, skills_mismatch):
        """Call FastAPI for eligible numeric pairs; return errors as status, never raise."""
        prepared = []
        for item in skills_mismatch or []:
            level = item.get("claimed_level")
            if level not in CLAIMED_LEVEL_SCALE or item.get("estimated_score") is None:
                continue
            try:
                claimed = CLAIMED_LEVEL_SCALE[level]
                measured = score_0_100_to_1_10(item["estimated_score"])
                prepared.append((item, {"claimed_skill": claimed, "measured_ability": measured}))
            except (ValueError, TypeError, KeyError) as exc:
                return {"available": False, "predictions": [], "message": f"ML input unavailable: {exc.__class__.__name__}"}
        if not prepared:
            return {"available": True, "predictions": [],
                    "message": "No matched resume skill ratings were available for ML inference."}
        try:
            api_key = os.environ.get("HIREWISE_ML_API_KEY", "")
            response = requests.post(
                f"{self.base_url}/predict/batch",
                json={"items": [payload for _, payload in prepared]},
                headers={"X-API-Key": api_key} if api_key else {},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payloads = response.json()
            if not isinstance(payloads, list) or len(payloads) != len(prepared):
                raise ValueError("ML batch response does not match the submitted items.")
            predictions = []
            for (item, input_values), payload in zip(prepared, payloads):
                if not isinstance(payload.get("inflated"), bool):
                    raise ValueError("ML response is missing a boolean 'inflated' result.")
                difference = float(payload["difference"])
                if not -9 <= difference <= 9:
                    raise ValueError("ML response difference is outside the supported range.")
                expected_difference = round(input_values["claimed_skill"] - input_values["measured_ability"], 2)
                if not math.isclose(difference, expected_difference, abs_tol=0.01):
                    raise ValueError("ML response difference does not match the submitted scores.")
                probability = payload.get("probability")
                if probability is not None and not 0 <= float(probability) <= 1:
                    raise ValueError("ML response probability is outside 0..1.")
                if not isinstance(payload.get("model_name"), str) or not payload["model_name"]:
                    raise ValueError("ML response is missing a model name.")
                predictions.append({
                    "skill_name": str(item.get("skill_name", "Skill"))[:80],
                    "inflated": payload["inflated"],
                    "difference": difference,
                    "probability": float(probability) if probability is not None else None,
                    "model_name": payload.get("model_name"),
                    "model_version": payload.get("model_version"),
                })
            return {"available": True, "predictions": predictions, "message": None}
        except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
            return {"available": False, "predictions": [], "message": f"ML service unavailable: {exc.__class__.__name__}"}

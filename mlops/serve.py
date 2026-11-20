"""FastAPI inference service with candidate-ID-free telemetry and monitoring."""
from __future__ import annotations

import asyncio
import os
import secrets
import uuid
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

try:
    from .data import FEATURES
    from .prediction_store import monitor_summary, record_prediction
except ImportError:  # uvicorn serve:app from mlops directory
    from data import FEATURES
    from prediction_store import monitor_summary, record_prediction

MODEL_NAME = "hirewise-inflation-detector"
API_KEY_ENV = "HIREWISE_ML_API_KEY"


def check_api_key(presented_key: str | None, configured_key: str | None) -> None:
    """Fail closed when auth is unconfigured; never reveal the configured key."""
    if not configured_key or not configured_key.strip():
        raise HTTPException(status_code=503, detail="API authentication is not configured.")
    if not presented_key or not secrets.compare_digest(
            presented_key.encode("utf-8"), configured_key.encode("utf-8")):
        raise HTTPException(status_code=401, detail="A valid API key is required.",
                            headers={"WWW-Authenticate": "ApiKey"})


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    check_api_key(x_api_key, os.environ.get(API_KEY_ENV))


class PredictRequest(BaseModel):
    claimed_skill: float = Field(..., ge=1.0, le=10.0)
    measured_ability: float = Field(..., ge=1.0, le=10.0)


class PredictResponse(BaseModel):
    inflated: bool
    difference: float
    probability: float | None = None
    model_name: str
    model_version: str | None = None
    request_id: str


class PredictBatchRequest(BaseModel):
    items: list[PredictRequest] = Field(..., min_length=1, max_length=50)


def load_champion_model():
    import mlflow
    import mlflow.sklearn
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:6001")
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{MODEL_NAME}@champion"
    model = mlflow.sklearn.load_model(model_uri)
    try:
        from mlflow.tracking import MlflowClient
        version = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion").version
    except Exception:
        version = None
    return model, str(version) if version is not None else None


def load_reference_frame():
    """Load the feature reference logged with the champion run when available."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        champion = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
        local = mlflow.artifacts.download_artifacts(run_id=champion.run_id,
                                                   artifact_path="reference/reference_data.csv")
        return pd.read_csv(local)
    except Exception as exc:
        print(f"[Monitoring warning] Reference dataset unavailable: {exc}")
        return None


def create_app(model_loader=load_champion_model, reference_loader=load_reference_frame,
               store_path=None):
    @asynccontextmanager
    async def lifespan(app_instance: FastAPI):
        try:
            loaded = model_loader()
            if isinstance(loaded, tuple):
                app_instance.state.model, app_instance.state.model_version = loaded
            else:
                app_instance.state.model, app_instance.state.model_version = loaded, None
        except Exception as exc:
            app_instance.state.model = None
            app_instance.state.model_version = None
            print(f"[Model warning] Champion model unavailable: {exc}")
        try:
            app_instance.state.reference_frame = reference_loader()
        except Exception as exc:
            app_instance.state.reference_frame = None
            print(f"[Monitoring warning] Reference load failed: {exc}")
        yield

    app = FastAPI(title="HireWise Resume Inflation Detection API", version="2.0.0", lifespan=lifespan)
    app.state.store_path = store_path

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        if request.url.path in {"/predict", "/predict/batch"}:
            try:
                check_api_key(request.headers.get("X-API-Key"), os.environ.get(API_KEY_ENV))
            except HTTPException as auth_error:
                return JSONResponse(status_code=auth_error.status_code,
                                    content={"detail": auth_error.detail},
                                    headers=auth_error.headers)
        try:
            record_prediction(MODEL_NAME, getattr(request.app.state, "model_version", None),
                              None, None, None, None, None, "invalid_request",
                              path=request.app.state.store_path)
        except Exception as log_exc:
            print(f"[Telemetry warning] Could not log invalid request: {log_exc}")
        return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})

    @app.get("/health")
    async def health():
        # Public liveness probe only; do not disclose model or monitoring details.
        return {"status": "ok"}

    def run_prediction(req: PredictRequest):
        model = app.state.model
        if model is None:
            try:
                record_prediction(MODEL_NAME, app.state.model_version, req.claimed_skill,
                                  req.measured_ability, None, None, None, "model_unavailable",
                                  path=app.state.store_path)
            except Exception as exc:
                print(f"[Telemetry warning] Could not log unavailable model: {exc}")
            raise HTTPException(status_code=503, detail="Model is currently unavailable or not loaded.")
        difference = round(req.claimed_skill - req.measured_ability, 2)
        features = [[req.claimed_skill, req.measured_ability, difference]]
        try:
            prediction = int(model.predict(features)[0])
            probability = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)[0]
                classes = list(getattr(model, "classes_", range(len(probs))))
                if prediction in classes:
                    probability = float(probs[classes.index(prediction)])
        except Exception as exc:
            try:
                record_prediction(MODEL_NAME, app.state.model_version, req.claimed_skill,
                                  req.measured_ability, difference, None, None,
                                  "prediction_error", path=app.state.store_path)
            except Exception as log_exc:
                print(f"[Telemetry warning] Could not log prediction failure: {log_exc}")
            raise HTTPException(status_code=503, detail="Prediction failed.") from exc
        request_id = str(uuid.uuid4())
        try:
            request_id = record_prediction(MODEL_NAME, app.state.model_version,
                                           req.claimed_skill, req.measured_ability,
                                           difference, prediction, probability,
                                           path=app.state.store_path)
        except Exception as log_exc:
            print(f"[Telemetry warning] Prediction succeeded but logging failed: {log_exc}")
        return PredictResponse(inflated=bool(prediction == 1), difference=difference,
                               probability=probability, model_name=MODEL_NAME,
                               model_version=app.state.model_version, request_id=request_id)

    @app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_api_key)])
    async def predict(req: PredictRequest):
        return await asyncio.to_thread(run_prediction, req)

    @app.post("/predict/batch", response_model=list[PredictResponse], dependencies=[Depends(require_api_key)])
    async def predict_batch(req: PredictBatchRequest):
        return await asyncio.to_thread(lambda: [run_prediction(item) for item in req.items])

    @app.get("/monitoring", dependencies=[Depends(require_api_key)])
    async def monitoring():
        try:
            raw_window = os.environ.get("MLOPS_MONITORING_WINDOW_DAYS", "30")
            window_days = int(raw_window)
            if window_days < 0:
                raise ValueError("MLOPS_MONITORING_WINDOW_DAYS must be zero or greater.")
            report = await asyncio.to_thread(monitor_summary, app.state.reference_frame,
                                             app.state.store_path, window_days)
            return report
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Monitoring report unavailable.") from exc

    return app


app = create_app()

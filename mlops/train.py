"""Train, evaluate, and conditionally promote the resume-skill demo model."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split

try:
    from .data import FEATURES, TARGET, load_training_data
except ImportError:  # python mlops/train.py
    from data import FEATURES, TARGET, load_training_data

MODEL_NAME = "hirewise-inflation-detector"


def is_promotion_eligible(dataset_type: str) -> bool:
    """Only explicitly attested anonymized real CSV data may promote a model."""
    return dataset_type == "approved_anonymized_real"


def passes_quality_gate(metrics: dict, min_accuracy: float = 0.80, min_f1: float = 0.80) -> bool:
    return metrics.get("validation_accuracy", -1) >= min_accuracy and metrics.get("validation_f1", -1) >= min_f1


def should_promote(candidate_f1: float, champion_f1: float | None, min_gain: float = 0.0,
                   *, dataset_type: str) -> bool:
    """Return whether an eligible, non-synthetic candidate beats the champion."""
    if not is_promotion_eligible(dataset_type):
        return False
    return champion_f1 is None or candidate_f1 >= champion_f1 + min_gain


def train_and_evaluate(data: pd.DataFrame, n_estimators: int = 50, max_depth: int = 5,
                       random_state: int = 42) -> dict:
    """Create deterministic train/validation/test partitions and return metrics/model."""
    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y)
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_hold, y_hold, test_size=0.5, random_state=random_state, stratify=y_hold)
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                   random_state=random_state, n_jobs=1)
    model.fit(X_train, y_train)
    validation_pred = model.predict(X_validation)
    test_pred = model.predict(X_test)
    metrics = {}
    for split, truth, predicted in (("validation", y_validation, validation_pred),
                                    ("test", y_test, test_pred)):
        metrics[f"{split}_accuracy"] = float(accuracy_score(truth, predicted))
        metrics[f"{split}_precision"] = float(precision_score(truth, predicted, zero_division=0))
        metrics[f"{split}_recall"] = float(recall_score(truth, predicted, zero_division=0))
        metrics[f"{split}_f1"] = float(f1_score(truth, predicted, zero_division=0))
    metrics["test_confusion_matrix"] = confusion_matrix(y_test, test_pred, labels=[0, 1]).tolist()
    metrics["test_classification_report"] = classification_report(
        y_test, test_pred, labels=[0, 1], target_names=["not_inflated", "inflated"],
        output_dict=True, zero_division=0)
    return {"model": model, "metrics": metrics, "validation_y": y_validation,
            "validation_pred": validation_pred}


def main(argv=None):
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    parser = argparse.ArgumentParser(description="Train with validated CSV or synthetic demo data")
    parser.add_argument("--dataset", default=None, help="Approved, anonymized CSV; omit to use synthetic demo data")
    parser.add_argument("--dataset-label", choices=["provided_csv_unverified_provenance", "approved_anonymized_real", "synthetic_demo"], default=None,
                        help="Explicit provenance label for a supplied CSV; real-data label is an operator attestation")
    parser.add_argument("--n_estimators", type=int, default=50)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--tracking_uri", default=None)
    parser.add_argument("--min_accuracy", type=float, default=0.80)
    parser.add_argument("--min_f1", type=float, default=0.80)
    parser.add_argument("--min_champion_f1_gain", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.n_estimators < 1 or args.max_depth < 1:
        raise ValueError("n_estimators and max_depth must be positive integers.")
    if not (0 <= args.min_accuracy <= 1 and 0 <= args.min_f1 <= 1):
        raise ValueError("Quality gate thresholds must be between 0 and 1.")

    training_data = load_training_data(args.dataset, args.random_state, args.dataset_label)
    mlflow.set_tracking_uri(args.tracking_uri or __import__("os").environ.get(
        "MLFLOW_TRACKING_URI", "http://127.0.0.1:6001"))
    mlflow.set_experiment("HireWise-AI")
    client = MlflowClient()

    with mlflow.start_run() as run:
        mlflow.log_params({"n_estimators": args.n_estimators, "max_depth": args.max_depth,
                           "random_state": args.random_state, "dataset_type": training_data.dataset_type,
                           "train_fraction": 0.6, "validation_fraction": 0.2, "test_fraction": 0.2,
                           "features": json.dumps(FEATURES)})
        mlflow.set_tags({"dataset_source": training_data.source,
                         "dataset_type": training_data.dataset_type,
                         "synthetic_metrics_are_not_real_world_validation": "true"})
        result = train_and_evaluate(training_data.frame, args.n_estimators,
                                    args.max_depth, args.random_state)
        model, metrics = result["model"], result["metrics"]
        mlflow.log_dict({"dataset_type": training_data.dataset_type,
                         "source": training_data.source,
                         "rows": len(training_data.frame),
                         "class_distribution": {str(k): int(v) for k, v in training_data.frame[TARGET].value_counts().sort_index().items()},
                         "feature_columns": FEATURES}, "dataset/metadata.json")
        numeric_metrics = {key: value for key, value in metrics.items()
                           if isinstance(value, (float, int))}
        mlflow.log_metrics(numeric_metrics)
        mlflow.log_dict(metrics["test_confusion_matrix"], "evaluation/confusion_matrix.json")
        mlflow.log_dict(metrics["test_classification_report"], "evaluation/classification_report.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = Path(temp_dir) / "reference_data.csv"
            training_data.frame[FEATURES].to_csv(reference_path, index=False)
            mlflow.log_artifact(str(reference_path), artifact_path="reference")
        promotion_eligible = is_promotion_eligible(training_data.dataset_type)
        print(json.dumps({"run_id": run.info.run_id,
                          "dataset_type": training_data.dataset_type,
                          "dataset_source": training_data.source,
                          "promotion_eligible": promotion_eligible,
                          "rows": len(training_data.frame), "metrics": metrics}, indent=2))

        validation_pass = passes_quality_gate(metrics, args.min_accuracy, args.min_f1)
        if not validation_pass:
            print(f"Candidate from {training_data.dataset_type} failed validation; champion unchanged.")
            raise ValueError("Quality gate failed on validation data; champion was not changed.")

        if not promotion_eligible:
            print(f"Evaluation-only run ({training_data.dataset_type}); champion was not changed.")
            return metrics

        champion_f1 = None
        champion_version = None
        try:
            champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
            champion_version = champion.version
            champion_run = client.get_run(champion.run_id)
            champion_f1 = champion_run.data.metrics.get(
                "validation_f1", champion_run.data.metrics.get("test_f1"))
        except Exception as exc:
            try:
                existing_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            except Exception as registry_exc:
                raise RuntimeError("Could not read the model registry; refusing to promote candidate.") from registry_exc
            if existing_versions:
                raise RuntimeError("Registered versions exist but champion could not be read; refusing to promote candidate.") from exc
        promotion_pass = should_promote(metrics["validation_f1"],
                                        float(champion_f1) if champion_f1 is not None else None,
                                        args.min_champion_f1_gain,
                                        dataset_type=training_data.dataset_type)
        if not promotion_pass:
            print(f"Candidate passed the gate but did not beat champion v{champion_version}; not promoted.")
            return metrics

        try:
            model_info = mlflow.sklearn.log_model(
                sk_model=model, name="model", registered_model_name=MODEL_NAME,
                input_example=training_data.frame[FEATURES].head(3))
        except TypeError:  # Compatibility with older MLflow client signatures.
            model_info = mlflow.sklearn.log_model(
                sk_model=model, artifact_path="model", registered_model_name=MODEL_NAME,
                input_example=training_data.frame[FEATURES].head(3))
        version = getattr(model_info, "registered_model_version", None)
        if not version:
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            version = max((int(v.version) for v in versions if v.run_id == run.info.run_id), default=None)
        if version is None:
            raise RuntimeError("Model logged but no registered version could be identified; champion unchanged.")
        client.set_registered_model_alias(MODEL_NAME, "champion", str(version))
        print(f"Promoted {MODEL_NAME} version {version} using validation F1={metrics['validation_f1']:.4f}.")
        return metrics


if __name__ == "__main__":
    main()

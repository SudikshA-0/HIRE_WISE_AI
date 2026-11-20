import sys
import types

import pandas as pd
import pytest

from mlops.data import (COLUMNS, TrainingData, generate_sample_data,
                        load_training_data, validate_frame)
from mlops.train import (is_promotion_eligible, passes_quality_gate,
                         should_promote, train_and_evaluate)


def test_synthetic_data_is_reproducible_and_valid():
    left = validate_frame(generate_sample_data(random_state=7))
    right = validate_frame(generate_sample_data(random_state=7))
    pd.testing.assert_frame_equal(left, right)


def test_invalid_schema_and_missing_values_are_rejected():
    data = generate_sample_data()
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_frame(data.drop(columns=["difference"]))
    data.loc[0, "claimed_skill"] = None
    with pytest.raises(ValueError, match="missing or non-numeric"):
        validate_frame(data)


def test_ranges_consistency_labels_and_duplicates_are_checked():
    original = generate_sample_data()
    bad = original.copy()
    bad.loc[0, "measured_ability"] = 11
    with pytest.raises(ValueError, match="between 1 and 10"):
        validate_frame(bad)
    bad = original.copy()
    bad.loc[0, "difference"] = 0
    with pytest.raises(ValueError, match="difference must equal"):
        validate_frame(bad)
    bad = original.copy()
    bad.loc[0, "inflated"] = 2
    with pytest.raises(ValueError, match="labels must be 0 or 1"):
        validate_frame(bad)
    with pytest.raises(ValueError, match="duplicate"):
        validate_frame(pd.concat([original, original.iloc[[0]]], ignore_index=True))


def test_csv_ingestion_and_training_metrics(tmp_path):
    data = generate_sample_data()
    path = tmp_path / "approved.csv"
    data.to_csv(path, index=False)
    loaded = load_training_data(path, random_state=42)
    assert loaded.dataset_type == "provided_csv_unverified_provenance"
    result = train_and_evaluate(loaded.frame, n_estimators=10, max_depth=3)
    assert {"validation_accuracy", "validation_precision", "validation_recall", "validation_f1",
            "test_accuracy", "test_precision", "test_recall", "test_f1",
            "test_confusion_matrix", "test_classification_report"} <= set(result["metrics"])
    assert result["model"].n_features_in_ == len(COLUMNS) - 1


def test_quality_gate_and_champion_comparison():
    assert passes_quality_gate({"validation_accuracy": .9, "validation_f1": .85})
    assert not passes_quality_gate({"validation_accuracy": .99, "validation_f1": .79})
    assert should_promote(.82, None, dataset_type="approved_anonymized_real")
    assert should_promote(.82, .82, dataset_type="approved_anonymized_real")
    assert not should_promote(.81, .82, dataset_type="approved_anonymized_real")
    assert not should_promote(1.0, 0.0, dataset_type="synthetic_demo")
    assert not should_promote(1.0, 0.0, dataset_type="provided_csv_unverified_provenance")
    assert should_promote(.82, .82, dataset_type="approved_anonymized_real")
    assert not is_promotion_eligible("synthetic_demo")
    assert not is_promotion_eligible("provided_csv_unverified_provenance")
    assert is_promotion_eligible("approved_anonymized_real")


@pytest.mark.parametrize("dataset_type,threshold,should_fail", [
    ("synthetic_demo", "0", False),
    ("approved_anonymized_real", "0.9", True),
])
def test_ineligible_or_failed_main_run_never_touches_champion_registry(
        monkeypatch, capsys, dataset_type, threshold, should_fail):
    import mlops.train as training

    class FakeRun:
        info = types.SimpleNamespace(run_id="synthetic-run")

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeClient:
        def __init__(self):
            self.alias_updates = []

        def get_model_version_by_alias(self, *_args):
            raise AssertionError("Synthetic training must not inspect or update champion state.")

        def set_registered_model_alias(self, *args):
            self.alias_updates.append(args)

    fake_client = FakeClient()
    frame = pd.DataFrame({
        "claimed_skill": [8.0] * 10,
        "measured_ability": [5.0] * 10,
        "difference": [3.0] * 10,
        "inflated": [0, 1] * 5,
    })
    monkeypatch.setattr(training, "load_training_data", lambda *_args: TrainingData(
        frame,
        "generated in-memory demo data" if dataset_type == "synthetic_demo" else "provided CSV",
        dataset_type))
    monkeypatch.setattr(training, "train_and_evaluate", lambda *_args: {
        "model": object(), "metrics": {
            "validation_accuracy": 0.8, "validation_precision": 0.8,
            "validation_recall": 0.8, "validation_f1": 0.8,
            "test_accuracy": 1.0, "test_precision": 1.0,
            "test_recall": 1.0, "test_f1": 1.0,
            "test_confusion_matrix": [[1, 0], [0, 1]],
            "test_classification_report": {"accuracy": 1.0},
        },
    })
    mlflow = types.ModuleType("mlflow")
    mlflow.__path__ = []
    mlflow.set_tracking_uri = lambda *_args: None
    mlflow.set_experiment = lambda *_args: None
    mlflow.log_params = lambda *_args: None
    mlflow.set_tags = lambda *_args: None
    mlflow.log_dict = lambda *_args: None
    mlflow.log_metrics = lambda *_args: None
    mlflow.log_artifact = lambda *_args, **_kwargs: None
    mlflow.start_run = lambda: FakeRun()
    mlflow.sklearn = types.ModuleType("mlflow.sklearn")
    mlflow.sklearn.log_model = lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError("Synthetic model must not be registered."))
    mlflow.tracking = types.ModuleType("mlflow.tracking")
    mlflow.tracking.MlflowClient = lambda: fake_client
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.sklearn", mlflow.sklearn)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", mlflow.tracking)

    arguments = ["--min_accuracy", threshold, "--min_f1", threshold]
    if should_fail:
        with pytest.raises(ValueError, match="Quality gate failed"):
            training.main(arguments)
    else:
        training.main(arguments)

    output = capsys.readouterr().out
    assert f'"dataset_type": "{dataset_type}"' in output
    assert f'"promotion_eligible": {str(dataset_type == "approved_anonymized_real").lower()}' in output
    if should_fail:
        assert f"Candidate from {dataset_type} failed validation; champion unchanged." in output
    else:
        assert f"Evaluation-only run ({dataset_type}); champion was not changed." in output
    assert fake_client.alias_updates == []

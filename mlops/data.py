"""Validated, privacy-conscious data loading for the resume inflation demo."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = ["claimed_skill", "measured_ability", "difference"]
TARGET = "inflated"
COLUMNS = FEATURES + [TARGET]
SCHEMA_DESCRIPTION = (
    "CSV columns: claimed_skill (numeric 1..10), measured_ability (numeric 1..10), "
    "difference (claimed_skill - measured_ability, tolerance 0.01), inflated (integer 0 or 1). "
    "One anonymous skill assessment per row; never include names, resume text, contact details, "
    "or other personal data. Synthetic and consented/anonymized real rows must be kept distinct."
)


@dataclass
class TrainingData:
    frame: pd.DataFrame
    source: str
    dataset_type: str


def generate_sample_data(num_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Generate demo-only labels from a transparent rule plus reproducible noise."""
    rng = np.random.default_rng(random_state)
    claimed = np.round(rng.uniform(2.0, 10.0, num_samples), 2)
    measured = np.round(rng.uniform(1.0, 9.5, num_samples), 2)
    difference = np.round(claimed - measured, 2)
    labels = (difference > 2.0).astype(int)
    flip_count = int(0.06 * num_samples)
    if flip_count:
        flip_indices = rng.choice(num_samples, size=flip_count, replace=False)
        labels[flip_indices] = 1 - labels[flip_indices]
    return pd.DataFrame({
        "claimed_skill": claimed,
        "measured_ability": measured,
        "difference": difference,
        "inflated": labels,
    })


def validate_frame(frame: pd.DataFrame, min_class_count: int = 5) -> pd.DataFrame:
    """Validate required columns, missing/duplicate rows, ranges, and binary labels."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Training dataset is empty or is not a tabular dataset.")
    missing_columns = [column for column in COLUMNS if column not in frame.columns]
    extra_columns = [column for column in frame.columns if column not in COLUMNS]
    if missing_columns or extra_columns:
        raise ValueError(f"Dataset schema mismatch; missing={missing_columns}, unexpected={extra_columns}. {SCHEMA_DESCRIPTION}")
    if frame.columns.duplicated().any():
        raise ValueError("Dataset contains duplicate column names.")
    data = frame[COLUMNS].copy()
    for column in COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    missing = data.isna().sum()
    missing = {key: int(value) for key, value in missing.items() if value}
    if missing:
        raise ValueError(f"Dataset contains missing or non-numeric values: {missing}")
    if data.duplicated().any():
        raise ValueError(f"Dataset contains {int(data.duplicated().sum())} duplicate rows.")
    for column in ("claimed_skill", "measured_ability"):
        invalid = ~data[column].between(1.0, 10.0)
        if invalid.any():
            raise ValueError(f"{column} must be between 1 and 10; found {int(invalid.sum())} invalid rows.")
    expected_difference = data.claimed_skill - data.measured_ability
    invalid_difference = ~np.isclose(data.difference, expected_difference, atol=0.01)
    if invalid_difference.any():
        raise ValueError(f"difference must equal claimed_skill - measured_ability; found {int(invalid_difference.sum())} invalid rows.")
    if not data[TARGET].isin([0, 1]).all():
        raise ValueError("inflated target labels must be 0 or 1.")
    if not np.all(np.equal(data[TARGET], data[TARGET].astype(int))):
        raise ValueError("inflated target labels must be integer values 0 or 1.")
    counts = data[TARGET].astype(int).value_counts().to_dict()
    if len(counts) != 2 or min(counts.values()) < min_class_count:
        raise ValueError(f"Both target classes need at least {min_class_count} rows; class counts={counts}.")
    data[TARGET] = data[TARGET].astype(int)
    return data


def load_training_data(dataset_path: str | Path | None, random_state: int = 42,
                       dataset_label: str | None = None) -> TrainingData:
    """Load and validate an explicitly supplied CSV, or generate labeled demo data."""
    if dataset_path:
        path = Path(dataset_path)
        if not path.is_file():
            raise ValueError(f"Training dataset file does not exist: {path}")
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            raise ValueError(f"Could not read training CSV {path}: {exc}") from exc
        allowed = {"provided_csv_unverified_provenance", "approved_anonymized_real", "synthetic_demo"}
        label = dataset_label or "provided_csv_unverified_provenance"
        if label not in allowed:
            raise ValueError(f"Unsupported dataset label {label!r}; choose one of {sorted(allowed)}.")
        return TrainingData(validate_frame(frame), "provided CSV (local path withheld)", label)
    if dataset_label not in (None, "synthetic_demo"):
        raise ValueError("Generated in-memory data must be labeled synthetic_demo.")
    frame = validate_frame(generate_sample_data(random_state=random_state))
    return TrainingData(frame, "generated in-memory demo data", "synthetic_demo")

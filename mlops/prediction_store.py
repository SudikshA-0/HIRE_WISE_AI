"""Small local SQLite store for candidate-ID-free model request telemetry."""
from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path


def database_path() -> Path:
    return Path(os.environ.get("MLOPS_PREDICTION_DB", "mlops/data/predictions.db"))


def connect(path: str | Path | None = None):
    target = Path(path) if path else database_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(target, timeout=5)
    connection.row_factory = sqlite3.Row
    connection.execute("""CREATE TABLE IF NOT EXISTS predictions (
        request_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, model_name TEXT NOT NULL,
        model_version TEXT, claimed_skill REAL, measured_ability REAL, difference REAL,
        predicted_class INTEGER, probability REAL, request_status TEXT NOT NULL)""")
    connection.commit()
    return connection


def record_prediction(model_name: str, model_version: str | None, claimed_skill: float | None,
                      measured_ability: float | None, difference: float | None,
                      predicted_class: int | None, probability: float | None,
                      request_status: str = "success", path: str | Path | None = None) -> str:
    request_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    connection = connect(path)
    try:
        with connection:
            connection.execute("""INSERT INTO predictions
                (request_id,timestamp,model_name,model_version,claimed_skill,measured_ability,
                 difference,predicted_class,probability,request_status)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (request_id,timestamp,model_name,model_version,claimed_skill,measured_ability,
                 difference,predicted_class,probability,request_status))
    finally:
        connection.close()
    return request_id


def monitor_summary(reference_frame=None, path: str | Path | None = None,
                    window_days: int = 30) -> dict:
    """Return all-time counts and PSI for recent rows (0 uses all retained rows)."""
    import pandas as pd

    if window_days < 0:
        raise ValueError("window_days must be zero or greater")

    connection = connect(path)
    try:
        rows = connection.execute("SELECT * FROM predictions ORDER BY timestamp").fetchall()
        failed = connection.execute("SELECT COUNT(*) FROM predictions WHERE request_status != 'success'").fetchone()[0]
    finally:
        connection.close()
    data = pd.DataFrame([dict(row) for row in rows])
    classes = data.predicted_class.dropna().astype(int).value_counts().to_dict() if not data.empty else {}
    summary = {"prediction_count": int(len(data)), "failed_request_count": int(failed),
               "class_distribution": {str(k): int(v) for k, v in classes.items()},
               "model_versions": sorted(data.model_version.dropna().astype(str).unique().tolist()) if not data.empty else [],
               "feature_means": {}, "data_drift": {"available": False, "features": {},
                                                   "warning": "Reference data is unavailable; drift was not evaluated."},
               "concept_drift": {"available": False, "reason": "Ground-truth labels are not collected or compared."}}
    features = ["claimed_skill", "measured_ability", "difference"]
    for feature in features:
        values = pd.to_numeric(data.get(feature, pd.Series(dtype=float)), errors="coerce").dropna()
        if not values.empty:
            summary["feature_means"][feature] = round(float(values.mean()), 4)
    drift_data = data
    if window_days and not data.empty:
        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
        timestamps = pd.to_datetime(data.timestamp, utc=True, errors="coerce")
        drift_data = data.loc[timestamps >= cutoff]
    current = drift_data[features].apply(pd.to_numeric, errors="coerce").dropna()
    summary["data_drift"].update({"window_days": window_days,
                                  "rows_in_window": int(len(drift_data)),
                                  "valid_rows_in_window": int(len(current))})
    if reference_frame is not None and len(drift_data):
        reference = reference_frame[features]
        details = {}
        drifted = []
        evaluated_count = 0
        for feature in features:
            ref = pd.to_numeric(reference[feature], errors="coerce").dropna().to_numpy()
            cur = current[feature].to_numpy()
            if len(ref) < 2 or len(cur) < 2:
                details[feature] = {"psi": None, "drift": False, "reason": "Insufficient rows"}
                continue
            edges = __import__("numpy").unique(__import__("numpy").quantile(ref, [0, .2, .4, .6, .8, 1]))
            if len(edges) < 3:
                details[feature] = {"psi": None, "drift": False, "reason": "Reference has too few distinct values"}
                continue
            edges[0], edges[-1] = -float("inf"), float("inf")
            ref_counts = __import__("numpy").histogram(ref, bins=edges)[0]
            cur_counts = __import__("numpy").histogram(cur, bins=edges)[0]
            ref_pct = (ref_counts + 1e-6) / (ref_counts.sum() + 1e-6 * len(ref_counts))
            cur_pct = (cur_counts + 1e-6) / (cur_counts.sum() + 1e-6 * len(cur_counts))
            psi = float(((cur_pct - ref_pct) * __import__("numpy").log(cur_pct / ref_pct)).sum())
            evaluated_count += 1
            details[feature] = {"psi": round(psi, 4), "drift": psi >= 0.2}
            if psi >= 0.2:
                drifted.append(feature)
        warning = (f"Potential input data drift detected: {', '.join(drifted)}" if drifted else
                   None if evaluated_count else "Not enough valid predictions to evaluate feature drift.")
        summary["data_drift"].update({"available": evaluated_count > 0,
                                      "features": details, "warning": warning})
    return summary


def cleanup_telemetry(older_than_days: int, path: str | Path | None = None,
                      apply: bool = False, now: datetime | None = None) -> dict:
    """Preview or explicitly delete rows older than a cutoff; preview is default."""
    if older_than_days < 1:
        raise ValueError("older_than_days must be a positive integer")
    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    cutoff = (current_time.astimezone(timezone.utc) - timedelta(days=older_than_days)).isoformat()
    connection = connect(path)
    try:
        matched = int(connection.execute(
            "SELECT COUNT(*) FROM predictions WHERE timestamp < ?", (cutoff,)
        ).fetchone()[0])
        deleted = 0
        if apply:
            with connection:
                cursor = connection.execute("DELETE FROM predictions WHERE timestamp < ?", (cutoff,))
                deleted = int(cursor.rowcount)
    finally:
        connection.close()
    return {"cutoff": cutoff, "matched": matched, "deleted": deleted,
            "dry_run": not apply}

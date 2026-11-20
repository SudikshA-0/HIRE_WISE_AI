from mlops.data import generate_sample_data
from mlops.prediction_store import monitor_summary, record_prediction


def test_prediction_store_summary_and_invalid_status(tmp_path):
    store = tmp_path / "monitor.sqlite"
    record_prediction("demo-model", "7", 8.0, 7.0, 1.0, 0, .8, path=store)
    record_prediction("demo-model", "7", 9.0, 6.0, 3.0, 1, .9, path=store)
    record_prediction("demo-model", "7", None, None, None, None, None,
                      request_status="invalid_request", path=store)
    reference = generate_sample_data(random_state=11)[["claimed_skill", "measured_ability", "difference"]]
    result = monitor_summary(reference, store)
    assert result["prediction_count"] == 3
    assert result["failed_request_count"] == 1
    assert result["class_distribution"] == {"0": 1, "1": 1}
    assert result["model_versions"] == ["7"]
    assert result["data_drift"]["available"] is True
    assert result["concept_drift"]["available"] is False


def test_insufficient_valid_telemetry_does_not_claim_drift(tmp_path):
    store = tmp_path / "insufficient.sqlite"
    record_prediction("demo-model", "1", 8.0, 7.0, 1.0, 0, .8, path=store)
    record_prediction("demo-model", "1", None, None, None, None, None,
                      request_status="invalid_request", path=store)
    reference = generate_sample_data()[["claimed_skill", "measured_ability", "difference"]]
    result = monitor_summary(reference, store)
    assert result["data_drift"]["available"] is False
    assert result["data_drift"]["rows_in_window"] == 2
    assert result["data_drift"]["valid_rows_in_window"] == 1
    assert all(detail.get("reason") == "Insufficient rows"
               for detail in result["data_drift"]["features"].values())


def test_distribution_shift_emits_warning(tmp_path):
    store = tmp_path / "drift.sqlite"
    for _ in range(30):
        record_prediction("demo-model", "1", 9.5, 2.0, 7.5, 1, .99, path=store)
    reference = generate_sample_data(random_state=42)[["claimed_skill", "measured_ability", "difference"]]
    result = monitor_summary(reference, store)
    assert result["data_drift"]["warning"]
    assert any(info.get("drift") for info in result["data_drift"]["features"].values())


def test_telemetry_cleanup_is_preview_only_until_explicit_apply(tmp_path):
    from datetime import datetime, timedelta, timezone
    import sqlite3
    from mlops.prediction_store import cleanup_telemetry

    store = tmp_path / "retention.sqlite"
    old_time = datetime.now(timezone.utc) - timedelta(days=120)
    record_prediction("demo-model", "1", 8.0, 7.0, 1.0, 0, .8, path=store)
    with sqlite3.connect(store) as connection:
        connection.execute("UPDATE predictions SET timestamp = ?", (old_time.isoformat(),))
    preview = cleanup_telemetry(90, store)
    assert preview["matched"] == 1
    assert preview["deleted"] == 0
    assert preview["dry_run"] is True
    applied = cleanup_telemetry(90, store, apply=True)
    assert applied["matched"] == 1
    assert applied["deleted"] == 1
    assert applied["dry_run"] is False
    assert cleanup_telemetry(90, store)["matched"] == 0


def test_monitoring_window_excludes_old_rows_from_psi(tmp_path):
    from datetime import datetime, timedelta, timezone
    import sqlite3

    store = tmp_path / "window.sqlite"
    for _ in range(3):
        record_prediction("demo-model", "1", 9.5, 2.0, 7.5, 1, .99, path=store)
    old_time = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
    with sqlite3.connect(store) as connection:
        connection.execute("UPDATE predictions SET timestamp = ?", (old_time,))
    reference = generate_sample_data()[["claimed_skill", "measured_ability", "difference"]]
    result = monitor_summary(reference, store, window_days=30)
    assert result["prediction_count"] == 3
    assert result["data_drift"]["rows_in_window"] == 0
    assert result["data_drift"]["available"] is False

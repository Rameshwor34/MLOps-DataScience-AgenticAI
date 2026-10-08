from pathlib import Path

import pandas as pd

from src.monitoring.drift import (
    calculate_target_drift,
    load_monitoring_data,
)
from src.monitoring.report import (
    extract_drift_metrics,
)


MONITORING_DIR = Path("data/monitoring")


def test_monitoring_datasets_exist():
    """Verify all monitoring datasets were generated."""

    for filename in [
        "reference.csv",
        "current.csv",
        "drifted.csv",
    ]:
        path = MONITORING_DIR / filename

        assert path.exists()

        df = pd.read_csv(path)

        assert not df.empty
        assert "Churn" in df.columns


def test_target_drift_not_detected_for_current_data():
    """Normal current data should not trigger target drift."""

    current, reference = load_monitoring_data()

    result = calculate_target_drift(
        current=current,
        reference=reference,
    )

    assert result["target_drift_detected"] is False
    assert (
        result["absolute_churn_rate_delta"]
        < result["target_drift_threshold"]
    )


def test_target_drift_detected_for_drifted_data():
    """Synthetic drifted data should trigger target drift."""

    current, reference = load_monitoring_data(
        current_path=MONITORING_DIR / "drifted.csv",
        reference_path=MONITORING_DIR / "reference.csv",
    )

    result = calculate_target_drift(
        current=current,
        reference=reference,
    )

    assert result["target_drift_detected"] is True
    assert (
        result["absolute_churn_rate_delta"]
        >= result["target_drift_threshold"]
    )


def test_evidently_drift_metric_structure():
    """Verify Evidently drift extraction returns numeric metrics."""

    from src.monitoring.drift import build_drift_report

    current, reference = load_monitoring_data()

    report = build_drift_report(
        current=current,
        reference=reference,
    )

    snapshot = report.run(
        current_data=current,
        reference_data=reference,
    )

    metrics = extract_drift_metrics(snapshot)

    assert "drifted_columns_count" in metrics
    assert "drifted_columns_share" in metrics

    assert metrics["drifted_columns_count"] >= 0
    assert 0.0 <= metrics["drifted_columns_share"] <= 1.0

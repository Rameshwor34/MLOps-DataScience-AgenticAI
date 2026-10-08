from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd

from src.monitoring.drift import (
    CURRENT_PATH,
    DRIFTED_PATH,
    REFERENCE_PATH,
    REPORT_DIR,
    build_drift_report,
    calculate_target_drift,
    load_monitoring_data,
)
from src.training.experiment import configure_mlflow


EXPERIMENT_NAME = "track-a-monitoring"

DRIFTED_COLUMNS_METRIC = "drifted_columns_count"
DRIFTED_COLUMNS_SHARE_METRIC = "drifted_columns_share"


def extract_drift_metrics(snapshot: object) -> dict[str, float]:
    """Extract overall drift metrics from an Evidently snapshot."""

    if not hasattr(snapshot, "dict"):
        raise TypeError(
            "Expected an Evidently Snapshot with a dict() method."
        )

    snapshot_dict = snapshot.dict()
    metrics = snapshot_dict.get("metrics", [])

    for metric in metrics:
        metric_name = metric.get("metric_name", "")

        if metric_name.startswith("DriftedColumnsCount("):
            value = metric.get("value", {})

            return {
                DRIFTED_COLUMNS_METRIC: float(
                    value.get("count", 0.0)
                ),
                DRIFTED_COLUMNS_SHARE_METRIC: float(
                    value.get("share", 0.0)
                ),
            }

    raise RuntimeError(
        "DriftedColumnsCount metric was not found "
        "in the Evidently snapshot."
    )


def calculate_custom_metrics(
    current: pd.DataFrame,
    reference: pd.DataFrame,
) -> dict[str, float]:
    """Calculate custom monitoring metrics."""

    target_result = calculate_target_drift(
        current=current,
        reference=reference,
    )

    return {
        "reference_churn_rate": float(
            target_result["reference_churn_rate"]
        ),
        "current_churn_rate": float(
            target_result["current_churn_rate"]
        ),
        "churn_rate_delta": float(
            target_result["churn_rate_delta"]
        ),
        "absolute_churn_rate_delta": float(
            target_result["absolute_churn_rate_delta"]
        ),
        "target_drift_threshold": float(
            target_result["target_drift_threshold"]
        ),
        "target_drift_detected": float(
            target_result["target_drift_detected"]
        ),
    }


def run_monitoring(
    current_path: Path,
    reference_path: Path = REFERENCE_PATH,
    report_name: str = "telco_monitoring_report.html",
    mlflow_run_name: str = "monitoring",
) -> dict[str, float | str]:
    """Run Evidently monitoring and log results to MLflow."""

    current, reference = load_monitoring_data(
        current_path=current_path,
        reference_path=reference_path,
    )

    report = build_drift_report(
        current=current,
        reference=reference,
    )

    snapshot = report.run(
        current_data=current,
        reference_data=reference,
    )

    evidently_metrics = extract_drift_metrics(snapshot)

    custom_metrics = calculate_custom_metrics(
        current=current,
        reference=reference,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = REPORT_DIR / report_name

    snapshot.save_html(str(report_path))

    # Explicitly use the project's shared SQLite MLflow backend.
    configure_mlflow()
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=mlflow_run_name):
        mlflow.log_param(
            "reference_dataset",
            str(reference_path),
        )

        mlflow.log_param(
            "current_dataset",
            str(current_path),
        )

        mlflow.log_param(
            "monitoring_type",
            "data_and_target_drift",
        )

        mlflow.log_metrics(evidently_metrics)
        mlflow.log_metrics(custom_metrics)

        mlflow.log_artifact(
            str(report_path),
            artifact_path="monitoring_reports",
        )

        run_id = mlflow.active_run().info.run_id

    return {
        **evidently_metrics,
        **custom_metrics,
        "report_path": str(report_path),
        "run_id": run_id,
    }


def print_results(
    title: str,
    results: dict[str, float | str],
) -> None:
    """Print monitoring results in a readable format."""

    print(f"\n{title}")
    print("=" * len(title))

    print(
        f"Drifted columns count: "
        f"{results['drifted_columns_count']:.0f}"
    )

    print(
        f"Drifted columns share: "
        f"{results['drifted_columns_share']:.4f}"
    )

    print(
        f"Reference churn rate: "
        f"{results['reference_churn_rate']:.4f}"
    )

    print(
        f"Current churn rate: "
        f"{results['current_churn_rate']:.4f}"
    )

    print(
        f"Churn rate delta: "
        f"{results['churn_rate_delta']:.4f}"
    )

    print(
        f"Absolute churn rate delta: "
        f"{results['absolute_churn_rate_delta']:.4f}"
    )

    print(
        f"Target drift detected: "
        f"{bool(results['target_drift_detected'])}"
    )

    print(
        f"HTML report: "
        f"{results['report_path']}"
    )

    print(
        f"MLflow run ID: "
        f"{results['run_id']}"
    )


if __name__ == "__main__":
    normal_results = run_monitoring(
        current_path=CURRENT_PATH,
        report_name="telco_monitoring_current.html",
        mlflow_run_name="monitoring_current",
    )

    print_results(
        "Current-data monitoring",
        normal_results,
    )

    drifted_results = run_monitoring(
        current_path=DRIFTED_PATH,
        report_name="telco_monitoring_drifted.html",
        mlflow_run_name="monitoring_drifted",
    )

    print_results(
        "Synthetic-drift monitoring",
        drifted_results,
    )

    assert (
        normal_results["target_drift_detected"]
        == 0.0
    )

    assert (
        drifted_results["target_drift_detected"]
        == 1.0
    )

    assert (
        drifted_results["drifted_columns_count"]
        > normal_results["drifted_columns_count"]
    )

    print("\nStep 26 monitoring pipeline: PASSED")

from __future__ import annotations

from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


MONITORING_DIR = Path("data/monitoring")
REPORT_DIR = Path("artifacts/reports")

REFERENCE_PATH = MONITORING_DIR / "reference.csv"
CURRENT_PATH = MONITORING_DIR / "current.csv"
DRIFTED_PATH = MONITORING_DIR / "drifted.csv"

TARGET_COLUMN = "Churn"

NUMERICAL_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def load_monitoring_data(
    current_path: Path = CURRENT_PATH,
    reference_path: Path = REFERENCE_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load current and reference monitoring datasets."""

    if not current_path.exists():
        raise FileNotFoundError(
            f"Current monitoring dataset not found: {current_path}"
        )

    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference monitoring dataset not found: {reference_path}"
        )

    current = pd.read_csv(current_path)
    reference = pd.read_csv(reference_path)

    return current, reference


def build_drift_report(
    current: pd.DataFrame,
    reference: pd.DataFrame,
) -> Report:
    """Build an Evidently feature-drift report."""

    monitored_columns = (
        NUMERICAL_COLUMNS
        + CATEGORICAL_COLUMNS
    )

    return Report(
        metrics=[
            DataDriftPreset(
                columns=monitored_columns,
                include_tests=True,
            )
        ],
        include_tests=True,
    )


def calculate_target_drift(
    current: pd.DataFrame,
    reference: pd.DataFrame,
) -> dict[str, float | bool]:
    """Calculate target distribution change for Churn."""

    if TARGET_COLUMN not in current.columns:
        raise ValueError(
            f"Missing target column in current data: {TARGET_COLUMN}"
        )

    if TARGET_COLUMN not in reference.columns:
        raise ValueError(
            f"Missing target column in reference data: {TARGET_COLUMN}"
        )

    reference_churn_rate = float(
        reference[TARGET_COLUMN].mean()
    )

    current_churn_rate = float(
        current[TARGET_COLUMN].mean()
    )

    churn_rate_delta = (
        current_churn_rate - reference_churn_rate
    )

    absolute_churn_rate_delta = abs(
        churn_rate_delta
    )

    # Monitoring threshold:
    # flag target drift when churn prevalence changes
    # by at least 5 percentage points.
    target_drift_threshold = 0.05

    target_drift_detected = (
        absolute_churn_rate_delta
        >= target_drift_threshold
    )

    return {
        "reference_churn_rate": reference_churn_rate,
        "current_churn_rate": current_churn_rate,
        "churn_rate_delta": churn_rate_delta,
        "absolute_churn_rate_delta": absolute_churn_rate_delta,
        "target_drift_threshold": target_drift_threshold,
        "target_drift_detected": target_drift_detected,
    }


def run_drift_report(
    current_path: Path = CURRENT_PATH,
    reference_path: Path = REFERENCE_PATH,
    output_path: Path | None = None,
) -> object:
    """Run Evidently feature-drift analysis and save HTML."""

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

    if output_path is None:
        output_path = (
            REPORT_DIR / "telco_data_drift_report.html"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot.save_html(str(output_path))

    return snapshot


def summarize_snapshot(snapshot: object) -> dict:
    """Extract a compact summary from an Evidently snapshot."""

    if not hasattr(snapshot, "dict"):
        raise TypeError(
            "Expected an Evidently Snapshot with a dict() method."
        )

    snapshot_dict = snapshot.dict()

    return {
        "has_snapshot": True,
        "top_level_keys": list(snapshot_dict.keys()),
    }


if __name__ == "__main__":
    current, reference = load_monitoring_data()

    snapshot = run_drift_report()

    target_result = calculate_target_drift(
        current=current,
        reference=reference,
    )

    print("Step 25 target drift: PASSED")

    print("\nTarget monitoring:")
    print(
        f"  Reference churn rate: "
        f"{target_result['reference_churn_rate']:.4f}"
    )
    print(
        f"  Current churn rate:   "
        f"{target_result['current_churn_rate']:.4f}"
    )
    print(
        f"  Churn rate delta:     "
        f"{target_result['churn_rate_delta']:.4f}"
    )
    print(
        f"  Absolute delta:       "
        f"{target_result['absolute_churn_rate_delta']:.4f}"
    )
    print(
        f"  Drift threshold:      "
        f"{target_result['target_drift_threshold']:.4f}"
    )
    print(
        f"  Drift detected:       "
        f"{target_result['target_drift_detected']}"
    )

    print(
        f"\nFeature drift report: "
        f"{REPORT_DIR / 'telco_data_drift_report.html'}"
    )

    print(
        f"Snapshot summary: "
        f"{summarize_snapshot(snapshot)}"
    )

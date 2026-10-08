from __future__ import annotations

import mlflow
import pandas as pd

from src.training.experiment import configure_mlflow, get_experiment


PRIMARY_METRIC = "f1"
SECONDARY_METRIC = "roc_auc"


def load_mlflow_runs() -> pd.DataFrame:
    """Load completed model runs from the Track A MLflow experiment."""

    configure_mlflow()
    experiment = get_experiment()

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=[f"metrics.{PRIMARY_METRIC} DESC"],
    )

    if runs.empty:
        raise RuntimeError("No completed MLflow runs were found.")

    return runs


def compare_models() -> pd.DataFrame:
    """Compare candidate models using MLflow-recorded validation metrics."""

    runs = load_mlflow_runs()

    required_columns = {
        "run_id",
        "tags.mlflow.runName",
        "metrics.accuracy",
        "metrics.precision",
        "metrics.recall",
        "metrics.f1",
        "metrics.roc_auc",
    }

    missing_columns = required_columns - set(runs.columns)

    if missing_columns:
        raise RuntimeError(
            f"Missing expected MLflow columns: {sorted(missing_columns)}"
        )

    comparison = runs[
        [
            "run_id",
            "tags.mlflow.runName",
            "metrics.accuracy",
            "metrics.precision",
            "metrics.recall",
            "metrics.f1",
            "metrics.roc_auc",
        ]
    ].copy()

    comparison = comparison.rename(
        columns={
            "tags.mlflow.runName": "model",
            "metrics.accuracy": "accuracy",
            "metrics.precision": "precision",
            "metrics.recall": "recall",
            "metrics.f1": "f1",
            "metrics.roc_auc": "roc_auc",
        }
    )

    comparison = comparison.sort_values(
        by=["f1", "roc_auc"],
        ascending=[False, False],
    ).reset_index(drop=True)

    comparison.insert(
        0,
        "rank",
        range(1, len(comparison) + 1),
    )

    return comparison


if __name__ == "__main__":
    comparison = compare_models()

    print("\nMLflow Model Comparison")
    print("=======================")
    print(comparison.to_string(index=False))

    best = comparison.iloc[0]

    print("\nSelection policy:")
    print(f"  Primary metric: {PRIMARY_METRIC}")
    print(f"  Secondary metric: {SECONDARY_METRIC}")

    print("\nBest validation model:")
    print(f"  Model: {best['model']}")
    print(f"  F1: {best['f1']:.4f}")
    print(f"  ROC-AUC: {best['roc_auc']:.4f}")

    print("\nModel comparison: PASSED")

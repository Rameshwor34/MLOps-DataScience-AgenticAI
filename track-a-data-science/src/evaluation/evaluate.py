from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from src.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_confusion_matrix,
)
from src.training.train import load_split, train_models


PROCESSED_DATA_DIR = Path("data/processed")


def evaluate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    """Evaluate one trained pipeline on a dataset."""

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    metrics = calculate_classification_metrics(
        y_true=y,
        y_pred=predictions,
        y_probability=probabilities,
    )

    confusion = calculate_confusion_matrix(
        y_true=y,
        y_pred=predictions,
    )

    return {
        "metrics": metrics,
        "confusion_matrix": confusion,
    }


def evaluate_all_models() -> dict[str, dict]:
    """Train candidate models and evaluate them on validation data."""

    X_validation, y_validation = load_split("validation.csv")

    trained_models = train_models()

    results: dict[str, dict] = {}

    for name, model in trained_models.items():
        print(f"\nEvaluating: {name}")

        result = evaluate_model(
            model=model,
            X=X_validation,
            y=y_validation,
        )

        results[name] = result

        print("Metrics:")

        for metric_name, value in result["metrics"].items():
            print(f"  {metric_name}: {value:.4f}")

        print("Confusion matrix:")
        print(result["confusion_matrix"])

    return results


if __name__ == "__main__":
    results = evaluate_all_models()

    print("\nEvaluation pipeline: PASSED")
    print(f"Models evaluated: {list(results)}")

from __future__ import annotations

import mlflow
from mlflow import MlflowClient

from src.evaluation.compare import compare_models
from src.training.experiment import configure_mlflow


REGISTERED_MODEL_NAME = "TelcoChurnClassifier"


def get_best_run():
    """Return the MLflow run selected by the validation model-comparison policy."""

    comparison = compare_models()

    best = comparison.iloc[0]

    return best


def register_best_model():
    """Register the selected MLflow model in the Model Registry."""

    configure_mlflow()

    best = get_best_run()

    run_id = best["run_id"]
    model_name = best["model"]

    model_uri = f"runs:/{run_id}/model"

    print("Selected model:")
    print(f"  Name: {model_name}")
    print(f"  Run ID: {run_id}")
    print(f"  F1: {best['f1']:.4f}")
    print(f"  ROC-AUC: {best['roc_auc']:.4f}")
    print(f"  Model URI: {model_uri}")

    client = MlflowClient()

    try:
        registered_model = client.get_registered_model(
            REGISTERED_MODEL_NAME
        )

        print(
            f"\nRegistry model already exists: "
            f"{registered_model.name}"
        )

    except Exception:
        print(
            f"\nCreating registered model: "
            f"{REGISTERED_MODEL_NAME}"
        )

        client.create_registered_model(
            REGISTERED_MODEL_NAME
        )

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    print("\nModel registration: PASSED")
    print(f"Registered model: {model_version.name}")
    print(f"Version: {model_version.version}")

    return model_version


if __name__ == "__main__":
    register_best_model()

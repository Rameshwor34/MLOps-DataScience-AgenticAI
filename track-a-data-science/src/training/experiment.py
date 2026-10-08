from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.sklearn

from src.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_confusion_matrix,
)
from src.evaluation.plots import save_confusion_matrix
from src.training.train import load_split, train_models


MLFLOW_DB = Path("mlflow.db")
EXPERIMENT_NAME = "track-a-telco-churn"
PLOT_DIR = Path("artifacts/plots")


def configure_mlflow() -> None:
    """Configure the local MLflow tracking backend."""

    tracking_uri = f"sqlite:///{MLFLOW_DB.resolve()}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)


def get_experiment():
    """Return the configured MLflow experiment."""

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment '{EXPERIMENT_NAME}' was not created."
        )

    return experiment


def extract_model_parameters(model) -> dict[str, object]:
    """Extract classifier parameters for MLflow logging."""

    classifier = model.named_steps["classifier"]

    parameters = {
        "model_type": classifier.__class__.__name__,
    }

    for parameter_name, parameter_value in classifier.get_params().items():
        if parameter_value is not None:
            parameters[f"model_{parameter_name}"] = str(parameter_value)

    return parameters


def evaluate_model(model, X_validation, y_validation) -> dict:
    """Evaluate one trained pipeline on the validation dataset."""

    predictions = model.predict(X_validation)
    probabilities = model.predict_proba(X_validation)[:, 1]

    metrics = calculate_classification_metrics(
        y_true=y_validation,
        y_pred=predictions,
        y_probability=probabilities,
    )

    confusion = calculate_confusion_matrix(
        y_true=y_validation,
        y_pred=predictions,
    )

    return {
        "metrics": metrics,
        "confusion_matrix": confusion,
    }


def run_experiments() -> dict[str, dict]:
    """Train, evaluate, and track all candidate models in MLflow."""

    configure_mlflow()

    X_validation, y_validation = load_split("validation.csv")
    trained_models = train_models()

    results: dict[str, dict] = {}

    for model_name, model in trained_models.items():
        print(f"\nMLflow run: {model_name}")

        with mlflow.start_run(run_name=model_name):
            parameters = extract_model_parameters(model)

            mlflow.log_params(parameters)

            result = evaluate_model(
                model=model,
                X_validation=X_validation,
                y_validation=y_validation,
            )

            mlflow.log_metrics(result["metrics"])

            confusion_path = save_confusion_matrix(
                confusion_matrix=result["confusion_matrix"],
                model_name=model_name,
                output_dir=PLOT_DIR,
            )

            mlflow.log_artifact(
                str(confusion_path),
                artifact_path="plots",
            )

            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.compose._column_transformer._RemainderColsList",
                    "sklearn.tree._tree.Tree",
                ],
            )

            results[model_name] = result

            print("Logged parameters:")
            print(f"  model_type: {parameters['model_type']}")

            print("Logged metrics:")
            for metric_name, value in result["metrics"].items():
                print(f"  {metric_name}: {value:.4f}")

            print(f"Logged confusion matrix: {confusion_path}")
            print(
                f"Run ID: {mlflow.active_run().info.run_id}"
            )

    return results


if __name__ == "__main__":
    results = run_experiments()

    print("\nMLflow experiment runs: PASSED")
    print(f"Runs created: {list(results)}")



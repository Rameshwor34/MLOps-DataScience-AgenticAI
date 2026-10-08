from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from src.data.preprocess import build_preprocessor
from src.training.models import build_models


RANDOM_STATE = 42

PROCESSED_DATA_DIR = Path("data/processed")
MODEL_DIR = Path("artifacts/models")


def load_split(
    filename: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load a processed split and separate features from target."""

    path = PROCESSED_DATA_DIR / filename
    df = pd.read_csv(path)

    if "Churn" not in df.columns:
        raise ValueError(f"Missing Churn column in {path}")

    X = df.drop(columns=["Churn"])
    y = df["Churn"].astype(int)

    return X, y


def build_pipeline(model: object) -> Pipeline:
    """Combine preprocessing and classifier into one pipeline."""

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", model),
        ]
    )


def train_models() -> dict[str, Pipeline]:
    """Train all candidate models using the training split only."""

    X_train, y_train = load_split("train.csv")

    models = build_models()
    trained_models: dict[str, Pipeline] = {}

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        print(f"\nTraining: {name}")

        pipeline = build_pipeline(model)

        pipeline.fit(X_train, y_train)

        trained_models[name] = pipeline

        print(f"Completed: {name}")

    return trained_models


if __name__ == "__main__":
    trained_models = train_models()

    print("\nTraining pipeline: PASSED")
    print(f"Models trained: {list(trained_models)}")

from __future__ import annotations

from functools import lru_cache

import mlflow
import pandas as pd
from fastapi import FastAPI

from src.serving.schemas import ChurnRequest, ChurnResponse
from src.training.experiment import configure_mlflow


MODEL_NAME = "TelcoChurnClassifier"
PRODUCTION_ALIAS = "production"

app = FastAPI(
    title="Telco Churn Prediction API",
    version="1.0.0",
    description="Prediction service backed by the MLflow production model.",
)


@lru_cache(maxsize=1)
def load_production_model():
    """Load and cache the model currently assigned to production."""

    configure_mlflow()

    model_uri = f"models:/{MODEL_NAME}@{PRODUCTION_ALIAS}"

    model = mlflow.sklearn.load_model(model_uri)

    return model


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""

    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "alias": PRODUCTION_ALIAS,
    }


@app.post("/predict", response_model=ChurnResponse)
def predict(request: ChurnRequest) -> ChurnResponse:
    """Predict customer churn using the production MLflow model."""

    model = load_production_model()

    input_data = pd.DataFrame(
        [request.model_dump()]
    )

    prediction = int(model.predict(input_data)[0])

    probability = float(
        model.predict_proba(input_data)[0, 1]
    )

    label = "Yes" if prediction == 1 else "No"

    return ChurnResponse(
        churn_prediction=prediction,
        churn_label=label,
        churn_probability=probability,
    )

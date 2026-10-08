from __future__ import annotations

from pydantic import BaseModel, Field


class ChurnRequest(BaseModel):
    """Customer features accepted by the prediction API."""

    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class ChurnResponse(BaseModel):
    """Prediction returned by the API."""

    churn_prediction: int
    churn_label: str
    churn_probability: float

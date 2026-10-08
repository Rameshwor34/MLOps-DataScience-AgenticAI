from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

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


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare raw Telco data into features and binary target."""

    data = df.copy()

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    # customerID is an identifier, not a predictive feature.
    data = data.drop(columns=[ID_COLUMN], errors="ignore")

    # TotalCharges is stored as text in the raw dataset.
    # Convert invalid/blank values into NaN so the preprocessing
    # pipeline can impute them using training-set statistics.
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"],
        errors="coerce",
    )

    # Encode target explicitly: No -> 0, Yes -> 1.
    target = data.pop(TARGET_COLUMN).map({"No": 0, "Yes": 1})

    if target.isna().any():
        raise ValueError("Target contains values other than 'Yes' or 'No'.")

    return data, target.astype(int)


def build_preprocessor() -> ColumnTransformer:
    """Build the feature preprocessing transformer."""

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_COLUMNS,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="drop",
    )


if __name__ == "__main__":
    df = pd.read_csv("data/raw/telco_churn.csv")

    X, y = prepare_features(df)

    print("Feature preparation: PASSED")
    print(f"Feature shape before encoding: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target classes: {sorted(y.unique().tolist())}")
    print(f"TotalCharges dtype: {X['TotalCharges'].dtype}")
    print(f"TotalCharges missing after conversion: {X['TotalCharges'].isna().sum()}")

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)

    print(f"Transformed feature shape: {transformed.shape}")


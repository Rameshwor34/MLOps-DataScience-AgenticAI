import pandas as pd


REQUIRED_COLUMNS = {
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
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
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
}

NUMERIC_COLUMNS = {
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
}

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

EXPECTED_TARGET_VALUES = {"Yes", "No"}


def validate_dataset(df: pd.DataFrame) -> dict:
    """Validate the Telco Churn dataset and return a structured report."""

    if df.empty:
        raise ValueError("Dataset is empty.")

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    missing_values = df.isna().sum()
    columns_with_missing_values = {
        column: int(count)
        for column, count in missing_values.items()
        if count > 0
    }

    duplicate_customer_ids = int(df[ID_COLUMN].duplicated().sum())

    unexpected_target_values = sorted(
        set(df[TARGET_COLUMN].dropna().unique()) - EXPECTED_TARGET_VALUES
    )

    total_charges_numeric = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    )

    invalid_total_charges = int(total_charges_numeric.isna().sum())

    target_distribution = (
        df[TARGET_COLUMN]
        .value_counts(dropna=False)
        .to_dict()
    )

    report = {
        "is_empty": bool(df.empty),
        "shape": tuple(df.shape),
        "column_count": int(len(df.columns)),
        "required_columns_present": True,
        "missing_values": columns_with_missing_values,
        "duplicate_customer_ids": duplicate_customer_ids,
        "invalid_total_charges": invalid_total_charges,
        "unexpected_target_values": unexpected_target_values,
        "target_distribution": {
            str(key): int(value)
            for key, value in target_distribution.items()
        },
    }

    if duplicate_customer_ids > 0:
        raise ValueError(
            f"Found {duplicate_customer_ids} duplicate customer IDs."
        )

    if unexpected_target_values:
        raise ValueError(
            "Unexpected target values found in Churn: "
            f"{unexpected_target_values}"
        )

    if invalid_total_charges > 0:
        raise ValueError(
            f"TotalCharges contains {invalid_total_charges} values "
            "that cannot be converted to numeric."
        )

    if len(target_distribution) < 2:
        raise ValueError(
            "Target column must contain at least two classes."
        )

    return report


if __name__ == "__main__":
    df = pd.read_csv("data/raw/telco_churn.csv")
    validation_report = validate_dataset(df)

    print("Dataset validation: PASSED")
    print(f"Shape: {validation_report['shape']}")
    print(
        "Target distribution:",
        validation_report["target_distribution"],
    )

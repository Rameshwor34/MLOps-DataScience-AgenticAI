from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.preprocess import prepare_features


RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

RAW_DATA_PATH = Path("data/raw/telco_churn.csv")
PROCESSED_DATA_DIR = Path("data/processed")


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Create reproducible stratified train/validation/test splits."""

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_val,
        y_train_val,
        test_size=VALIDATION_SIZE,
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )


def save_splits(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
    y_test: pd.Series,
) -> None:
    """Save raw-feature splits for reproducible inspection."""

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train = X_train.copy()
    train["Churn"] = y_train

    validation = X_validation.copy()
    validation["Churn"] = y_validation

    test = X_test.copy()
    test["Churn"] = y_test

    train.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    validation.to_csv(PROCESSED_DATA_DIR / "validation.csv", index=False)
    test.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)


def distribution(y: pd.Series) -> dict[str, float]:
    """Return class proportions."""

    return (
        y.value_counts(normalize=True)
        .sort_index()
        .round(4)
        .to_dict()
    )


if __name__ == "__main__":
    df = pd.read_csv(RAW_DATA_PATH)

    X, y = prepare_features(df)

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = split_dataset(X, y)

    save_splits(
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )

    print("Dataset splitting: PASSED")
    print(f"Total samples: {len(X)}")
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_validation)}")
    print(f"Test samples: {len(X_test)}")

    print("\nClass distributions:")
    print(f"Train:       {distribution(y_train)}")
    print(f"Validation:  {distribution(y_validation)}")
    print(f"Test:        {distribution(y_test)}")

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROCESSED_DATA_DIR = Path("data/processed")
MONITORING_DIR = Path("data/monitoring")


def create_monitoring_datasets() -> dict[str, Path]:
    """Create reference, current, and synthetic drifted datasets."""

    train_path = PROCESSED_DATA_DIR / "train.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing training split: {train_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"Missing test split: {test_path}")

    reference = pd.read_csv(train_path)
    current = pd.read_csv(test_path)

    # Create a controlled synthetic drift scenario.
    drifted = current.copy()

    # Numerical drift: increase monthly charges by 20%.
    drifted["MonthlyCharges"] = (
        drifted["MonthlyCharges"] * 1.20
    )

    # Categorical drift: make month-to-month contracts more common.
    contract_mask = drifted["Contract"] == "Month-to-month"
    drifted.loc[~contract_mask, "Contract"] = "Month-to-month"

    # Target drift: increase the churn rate in the synthetic batch.
    churn_zero_indices = drifted.index[
        drifted["Churn"] == 0
    ]

    replacement_count = min(
        int(len(drifted) * 0.10),
        len(churn_zero_indices),
    )

    if replacement_count > 0:
        drifted.loc[
            churn_zero_indices[:replacement_count],
            "Churn",
        ] = 1

    MONITORING_DIR.mkdir(parents=True, exist_ok=True)

    reference_path = MONITORING_DIR / "reference.csv"
    current_path = MONITORING_DIR / "current.csv"
    drifted_path = MONITORING_DIR / "drifted.csv"

    reference.to_csv(reference_path, index=False)
    current.to_csv(current_path, index=False)
    drifted.to_csv(drifted_path, index=False)

    return {
        "reference": reference_path,
        "current": current_path,
        "drifted": drifted_path,
    }


if __name__ == "__main__":
    paths = create_monitoring_datasets()

    print("Monitoring datasets: PASSED")

    for name, path in paths.items():
        df = pd.read_csv(path)

        print(
            f"{name}: "
            f"shape={df.shape}, "
            f"saved={path}"
        )

    reference = pd.read_csv(paths["reference"])
    current = pd.read_csv(paths["current"])
    drifted = pd.read_csv(paths["drifted"])

    print("\nChurn rates:")
    print(f"Reference: {reference['Churn'].mean():.4f}")
    print(f"Current:   {current['Churn'].mean():.4f}")
    print(f"Drifted:   {drifted['Churn'].mean():.4f}")

    print("\nMonthlyCharges means:")
    print(f"Reference: {reference['MonthlyCharges'].mean():.2f}")
    print(f"Current:   {current['MonthlyCharges'].mean():.2f}")
    print(f"Drifted:   {drifted['MonthlyCharges'].mean():.2f}")

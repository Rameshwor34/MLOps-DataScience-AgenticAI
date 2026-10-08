from __future__ import annotations

from mlflow import MlflowClient

from src.training.experiment import configure_mlflow


REGISTERED_MODEL_NAME = "TelcoChurnClassifier"
STAGING_ALIAS = "staging"
PRODUCTION_ALIAS = "production"


def get_latest_version(client: MlflowClient):
    """Return the newest registered model version."""

    versions = client.search_model_versions(
        f"name = '{REGISTERED_MODEL_NAME}'"
    )

    if not versions:
        raise RuntimeError(
            f"No versions found for {REGISTERED_MODEL_NAME}."
        )

    return max(
        versions,
        key=lambda version: int(version.version),
    )


def promote_to_staging():
    """Assign the selected registered model version to staging."""

    configure_mlflow()

    client = MlflowClient()

    version = get_latest_version(client)

    print("Registered model:")
    print(f"  Name: {REGISTERED_MODEL_NAME}")
    print(f"  Version: {version.version}")
    print(f"  Run ID: {version.run_id}")

    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        STAGING_ALIAS,
        version.version,
    )

    print("\nStaging promotion: PASSED")
    print(
        f"  {REGISTERED_MODEL_NAME}@{STAGING_ALIAS} "
        f"-> version {version.version}"
    )


def promote_staging_to_production():
    """Promote the exact staging version to production."""

    configure_mlflow()

    client = MlflowClient()

    staging_version = client.get_model_version_by_alias(
        REGISTERED_MODEL_NAME,
        STAGING_ALIAS,
    )

    print("Current staging version:")
    print(f"  Version: {staging_version.version}")
    print(f"  Run ID: {staging_version.run_id}")

    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        PRODUCTION_ALIAS,
        staging_version.version,
    )

    print("\nProduction promotion: PASSED")
    print(
        f"  {REGISTERED_MODEL_NAME}@{PRODUCTION_ALIAS} "
        f"-> version {staging_version.version}"
    )


if __name__ == "__main__":
    promote_staging_to_production()


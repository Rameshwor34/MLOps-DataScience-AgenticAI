from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def build_models() -> dict[str, object]:
    """Return the candidate classification models for Track A."""

    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }


if __name__ == "__main__":
    models = build_models()

    print("Model factory: PASSED")

    for name, model in models.items():
        print(f"{name}: {model.__class__.__name__}")

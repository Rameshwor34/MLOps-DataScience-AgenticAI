import numpy as np

from src.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_confusion_matrix,
)


def test_classification_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_probability = np.array([0.10, 0.70, 0.80, 0.90])

    metrics = calculate_classification_metrics(
        y_true,
        y_pred,
        y_probability,
    )

    assert set(metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_confusion_matrix_shape():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    matrix = calculate_confusion_matrix(
        y_true,
        y_pred,
    )

    assert matrix.shape == (2, 2)
    assert matrix.sum() == len(y_true)

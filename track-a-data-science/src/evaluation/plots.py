from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def save_confusion_matrix(
    confusion_matrix,
    model_name: str,
    output_dir: Path,
) -> Path:
    """Save a confusion matrix visualization and return its path."""

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{model_name}_confusion_matrix.png"

    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=["No Churn", "Churn"],
    )

    display.plot()

    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


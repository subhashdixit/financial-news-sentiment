"""Render evaluation charts directly from measured metrics."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Backend must be selected before pyplot import.
# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def plot_results():
    """Generate reproducible charts without reading raw applicant data."""
    report = json.loads((ROOT / "models/metrics.json").read_text())
    destination = ROOT / "docs/images"
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#f4f6f5",
        }
    )
    matrix = np.asarray(report["test"]["confusion_matrix"])
    fig, axis = plt.subplots(figsize=(7, 5))
    image = axis.imshow(matrix, cmap="Greens")
    axis.set(
        xticks=range(len(report["classes"])),
        yticks=range(len(report["classes"])),
        xticklabels=report["classes"],
        yticklabels=report["classes"],
        xlabel="Predicted category",
        ylabel="Actual category",
        title="Financial News Lens · held-out confusion matrix",
    )
    for row in range(len(report["classes"])):
        for col in range(len(report["classes"])):
            axis.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > matrix.max() / 2 else "#123a30",
            )
    fig.colorbar(image, ax=axis, label="Number of rows")
    fig.tight_layout()
    fig.savefig(destination / "confusion-matrix.png", dpi=150)
    plt.close(fig)
    candidates = report["validation_candidates"]
    fig, axis = plt.subplots(figsize=(8, 4))
    names = list(candidates)
    axis.barh(
        [name.replace("_", " ") for name in names],
        [candidates[name]["macro_f1"] for name in names],
        color="#24664e",
    )
    axis.set(
        xlim=(0, 1),
        xlabel="Validation macro F1 (higher is better)",
        title="Compare candidates before opening the test results",
    )
    fig.tight_layout()
    fig.savefig(destination / "model-comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_results()

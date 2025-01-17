import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def visualize(all_labels, all_predictions, output_path="visualizations"):
    """Visualize predictions vs. true labels."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot Predicted vs. True labels
    plt.figure(figsize=(8, 6))
    plt.scatter(all_labels, all_predictions, alpha=0.5)
    plt.plot([0, 1], [0, 1], color="red", linestyle="--", label="Ideal Fit")
    plt.xlabel("True Labels")
    plt.ylabel("Predicted Labels")
    plt.title("Predicted vs True Labels")
    plt.legend()
    plt.savefig(output_dir / "predicted_vs_true.png")
    plt.close()

    # Histogram of errors
    errors = np.array(all_predictions) - np.array(all_labels)
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=30, alpha=0.7, color="blue", edgecolor="black")
    plt.title("Histogram of Prediction Errors")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.savefig(output_dir / "error_histogram.png")
    plt.close()

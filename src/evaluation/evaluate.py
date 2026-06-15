"""
evaluate.py — Compute detection metrics and generate evaluation plots.

Runs a trained YOLOv8 model on the test set and produces:
  • mAP@0.5, mAP@0.5:0.95, precision, recall (per class and overall)
  • Confusion matrix heatmap
  • Precision-Recall curves
  • Per-class bar charts

These artefacts go into a results/ directory for easy inclusion in the
competition report / README.

Usage:
    python -m src.evaluation.evaluate \
        --weights models/best.pt \
        --data configs/malaria.yaml \
        --output_dir results
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ultralytics import YOLO

# Use non-interactive backend so plots save on headless servers
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Class names (must match configs/malaria.yaml)
# ---------------------------------------------------------------------------
CLASS_NAMES = ["red_blood_cell", "ring", "trophozoite", "schizont", "gametocyte"]


def run_validation(
    weights: str,
    data_config: str,
    device: str = "cpu",
    imgsz: int = 640,
    batch_size: int = 16,
    split: str = "test",
) -> Any:
    """Run YOLO validation on the specified split.

    Returns the Ultralytics Results object which contains all metrics.
    We intentionally run with verbose=True so the console shows per-class AP
    for quick sanity checks during development.
    """
    model = YOLO(weights)
    results = model.val(
        data=data_config,
        split=split,
        device=device,
        imgsz=imgsz,
        batch=batch_size,
        verbose=True,
    )
    return results


def extract_metrics(results: Any) -> Dict[str, Any]:
    """Extract key metrics from Ultralytics validation results.

    Returns a flat dictionary suitable for JSON serialisation and
    display in the Streamlit app.
    """
    box = results.box

    metrics = {
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "precision_mean": float(box.mp),
        "recall_mean": float(box.mr),
        "per_class": {},
    }

    # Per-class breakdown
    for i, name in enumerate(CLASS_NAMES):
        if i < len(box.ap50()):
            metrics["per_class"][name] = {
                "AP50": float(box.ap50()[i]),
                "precision": float(box.p[i]) if i < len(box.p) else None,
                "recall": float(box.r[i]) if i < len(box.r) else None,
            }

    return metrics


def plot_confusion_matrix(
    results: Any,
    output_path: Path,
    class_names: List[str] = CLASS_NAMES,
) -> None:
    """Generate and save a confusion matrix heatmap.

    The confusion matrix from YOLO includes a 'background' class
    (false positives / false negatives). We plot the full matrix
    because seeing where the model confuses parasites with RBCs
    is clinically important.
    """
    # Ultralytics stores the confusion matrix in results.confusion_matrix
    cm = results.confusion_matrix
    if cm is None or not hasattr(cm, "matrix"):
        print("  WARNING: Confusion matrix not available (too few predictions?).")
        return

    matrix = cm.matrix  # Shape: (nc+1, nc+1) — includes background class
    labels = class_names + ["background"]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.5,
        linecolor="grey",
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Detection Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path = output_path / "confusion_matrix.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved confusion matrix → {save_path}")


def plot_per_class_metrics(
    metrics: Dict[str, Any],
    output_path: Path,
) -> None:
    """Bar chart showing AP50, precision, and recall per class.

    This visualisation quickly reveals which parasite stages the model
    struggles with (usually schizonts and gametocytes due to rarity).
    """
    per_class = metrics.get("per_class", {})
    if not per_class:
        print("  WARNING: No per-class metrics to plot.")
        return

    names = list(per_class.keys())
    ap50_vals = [per_class[n].get("AP50", 0) or 0 for n in names]
    prec_vals = [per_class[n].get("precision", 0) or 0 for n in names]
    rec_vals = [per_class[n].get("recall", 0) or 0 for n in names]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, ap50_vals, width, label="AP@0.5", color="#2196F3")
    ax.bar(x, prec_vals, width, label="Precision", color="#4CAF50")
    ax.bar(x + width, rec_vals, width, label="Recall", color="#FF9800")

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class Detection Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    save_path = output_path / "per_class_metrics.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved per-class metrics → {save_path}")


def plot_summary_card(
    metrics: Dict[str, Any],
    output_path: Path,
) -> None:
    """Summary card with overall metrics, suitable for the README."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")

    text_lines = [
        f"mAP@0.5:        {metrics['mAP50']:.4f}",
        f"mAP@0.5:0.95:   {metrics['mAP50_95']:.4f}",
        f"Precision (avg): {metrics['precision_mean']:.4f}",
        f"Recall (avg):    {metrics['recall_mean']:.4f}",
    ]
    text = "\n".join(text_lines)

    ax.text(
        0.5, 0.5, text,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="center",
        horizontalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#E3F2FD", edgecolor="#1976D2"),
    )
    ax.set_title("Overall Detection Performance", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()

    save_path = output_path / "summary_card.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved summary card → {save_path}")


def evaluate(
    weights: str = "models/best.pt",
    data_config: str = "configs/malaria.yaml",
    output_dir: str = "results",
    device: str = "cpu",
    imgsz: int = 640,
    batch_size: int = 16,
    split: str = "test",
) -> Dict[str, Any]:
    """End-to-end evaluation pipeline.

    1. Run validation on the test split.
    2. Extract metrics.
    3. Generate all plots.
    4. Save metrics as JSON.

    Returns:
        Dictionary of computed metrics.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MALARIA DETECTION — MODEL EVALUATION")
    print("=" * 60)
    print(f"  Weights: {weights}")
    print(f"  Split:   {split}")
    print("=" * 60)

    # Step 1: Run validation
    results = run_validation(weights, data_config, device, imgsz, batch_size, split)

    # Step 2: Extract metrics
    metrics = extract_metrics(results)

    # Step 3: Generate plots
    print("\nGenerating evaluation plots ...")
    plot_confusion_matrix(results, output_path)
    plot_per_class_metrics(metrics, output_path)
    plot_summary_card(metrics, output_path)

    # Step 4: Save metrics JSON
    metrics_path = output_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics JSON → {metrics_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  mAP@0.5:        {metrics['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95:   {metrics['mAP50_95']:.4f}")
    print(f"  Precision (avg): {metrics['precision_mean']:.4f}")
    print(f"  Recall (avg):    {metrics['recall_mean']:.4f}")
    print("=" * 60)

    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained YOLOv8 malaria detection model."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="models/best.pt",
        help="Path to trained model weights.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="configs/malaria.yaml",
        help="Path to YOLO data config YAML.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory for evaluation outputs (default: results).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device: 'cpu' or GPU index.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on (default: test).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (default: 16).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        weights=args.weights,
        data_config=args.data,
        output_dir=args.output_dir,
        device=args.device,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        split=args.split,
    )

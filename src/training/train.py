"""
train.py — YOLOv8 training script for malaria parasite detection.

Wraps the Ultralytics YOLO API with competition-friendly defaults:
  • Configurable model size (nano → xlarge) via CLI
  • Automatic mixed-precision on GPU, graceful CPU fallback
  • Cosine LR scheduler + mosaic augmentation (YOLO defaults)
  • Saves best + last weights to models/

Usage:
    # Quick training run (CPU, nano model, 10 epochs for testing)
    python -m src.training.train \
        --data configs/malaria.yaml \
        --model yolov8n.pt \
        --epochs 10 \
        --batch_size 8

    # Full training (GPU, medium model, 100 epochs)
    python -m src.training.train \
        --data configs/malaria.yaml \
        --model yolov8m.pt \
        --epochs 100 \
        --batch_size 16 \
        --device 0
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Model size presets — maps friendly names to Ultralytics weight files.
# Pre-trained on COCO, which gives a strong initialisation even for
# medical images because the early conv layers learn generic features
# (edges, textures) that transfer well.
# ---------------------------------------------------------------------------
MODEL_PRESETS = {
    "nano": "yolov8n.pt",
    "small": "yolov8s.pt",
    "medium": "yolov8m.pt",
    "large": "yolov8l.pt",
    "xlarge": "yolov8x.pt",
}


def train(
    data_config: str = "configs/malaria.yaml",
    model: str = "yolov8n.pt",
    epochs: int = 50,
    batch_size: int = 16,
    imgsz: int = 640,
    device: str = "cpu",
    project: str = "runs/train",
    name: str = "malaria_detection",
    patience: int = 15,
    lr0: float = 0.01,
    resume: bool = False,
    workers: int = 4,
) -> str:
    """Train a YOLOv8 model on the malaria dataset.

    This function configures and launches an Ultralytics training run.
    Key design decisions:
      • patience=15 gives early stopping headroom so we don't waste compute
        on a converged model, but also don't stop too aggressively on noisy
        validation loss.
      • imgsz=640 balances resolution (parasites are small) with memory.
        If you have >16 GB VRAM, consider 1024.
      • mosaic augmentation (Ultralytics default) is extremely effective
        for detection because it forces the model to see objects at many scales.

    Args:
        data_config: Path to YOLO dataset YAML config.
        model: Pre-trained weights file or model size name.
        epochs: Total training epochs.
        batch_size: Batch size (reduce if OOM on GPU or slow on CPU).
        imgsz: Input image size.
        device: 'cpu', '0', '0,1', etc.
        project: Output directory for training artifacts.
        name: Experiment name (subdirectory under project).
        patience: Early stopping patience (epochs without improvement).
        lr0: Initial learning rate.
        resume: Resume from last checkpoint.
        workers: DataLoader workers (set 0 on Windows if issues).

    Returns:
        Path to the best model weights file.
    """
    # Resolve model size shortcuts
    if model.lower() in MODEL_PRESETS:
        model = MODEL_PRESETS[model.lower()]

    print("=" * 60)
    print("MALARIA DETECTION — YOLOv8 TRAINING")
    print("=" * 60)
    print(f"  Model:       {model}")
    print(f"  Data config: {data_config}")
    print(f"  Epochs:      {epochs}")
    print(f"  Batch size:  {batch_size}")
    print(f"  Image size:  {imgsz}")
    print(f"  Device:      {device}")
    print(f"  LR:          {lr0}")
    print("=" * 60)

    # Initialise model (downloads pre-trained weights if not cached)
    yolo_model = YOLO(model)

    # Launch training
    yolo_model.train(
        data=data_config,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        patience=patience,
        lr0=lr0,
        resume=resume,
        workers=workers,
        # --- Augmentation tuning for microscopy ---
        # Disable vertical flip (blood smears have no orientation bias,
        # but keeping default horizontal flip is fine).
        flipud=0.5,
        # Mosaic probability — keep high for small-object detection
        mosaic=1.0,
        # Mild colour jitter — stain normalisation already handles hue shifts
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.3,
        # Save best and last checkpoints
        save=True,
        save_period=-1,  # Save only best + last (not every N epochs)
        exist_ok=True,
    )

    # Copy best weights to the models/ directory for easy access
    best_weights = Path(project) / name / "weights" / "best.pt"
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    if best_weights.exists():
        import shutil
        dest = models_dir / "best.pt"
        shutil.copy2(best_weights, dest)
        print(f"\n✅ Best weights saved to {dest}")
        return str(dest)
    else:
        print("\n⚠️ best.pt not found; training may not have completed.")
        return ""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for malaria parasite detection."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="configs/malaria.yaml",
        help="Path to YOLO data config YAML.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Pre-trained model (e.g., yolov8n.pt, yolov8s.pt, or 'nano', 'small', etc.).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (default: 16). Reduce for CPU or limited VRAM.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device: 'cpu', '0' (GPU 0), '0,1' (multi-GPU).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Output project directory.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="malaria_detection",
        help="Experiment name.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience (default: 15).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Initial learning rate (default: 0.01).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="DataLoader workers (default: 4, use 0 on Windows if errors).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_config=args.data,
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        lr0=args.lr,
        resume=args.resume,
        workers=args.workers,
    )

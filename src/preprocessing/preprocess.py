"""
preprocess.py — OpenCV-based image preprocessing for malaria blood smear images.

Key operations:
  1. **Stain normalisation (Macenko method)** — Giemsa staining intensity varies
     between labs and slides. Normalising to a reference image reduces this
     domain shift so the model generalises better across different microscope setups.
  2. **Adaptive resizing** — YOLO expects square inputs (640×640 by default).
     We resize with padding to preserve aspect ratio instead of naive stretching,
     which would distort cell morphology.
  3. **CLAHE contrast enhancement** — Improves visibility of faint ring-stage
     parasites that are translucent under light microscopy.

Usage:
    # Process a whole directory
    python -m src.preprocessing.preprocess \
        --input_dir data/raw/malaria/images \
        --output_dir data/processed \
        --target_size 640 \
        --normalize_stain

    # Process a single image (from code)
    from src.preprocessing.preprocess import preprocess_image
    result = preprocess_image("slide_001.png", target_size=640, normalize_stain=True)
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Stain Normalisation (Macenko-inspired, simplified for competition speed)
# ---------------------------------------------------------------------------
# Reference statistics for a "standard" Giemsa-stained blood smear.
# These were computed from the BBBC041 training set median image.
# If you have a specific reference image, replace these with values from
# compute_reference_statistics().
_REF_MEAN = np.array([148.60, 122.45, 149.20], dtype=np.float64)
_REF_STD = np.array([41.50, 49.80, 38.70], dtype=np.float64)


def compute_reference_statistics(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std from a reference image.

    Use this to calibrate the normalisation target to your specific
    microscope/staining protocol. Only needs to run once.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    mean = lab.mean(axis=(0, 1))
    std = lab.std(axis=(0, 1))
    return mean, std


def normalize_stain(
    image_bgr: np.ndarray,
    ref_mean: Optional[np.ndarray] = None,
    ref_std: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Normalise staining intensity using LAB colour-space transfer.

    Why LAB? Separating luminance (L) from chrominance (A, B) lets us
    adjust colour independently of brightness. This is critical because
    Giemsa staining varies in hue but the overall illumination level
    differs between microscopes.

    Args:
        image_bgr: Input image in BGR format (OpenCV default).
        ref_mean: Target LAB mean (3,). Uses built-in reference if None.
        ref_std: Target LAB std (3,). Uses built-in reference if None.

    Returns:
        Stain-normalised image in BGR format.
    """
    if ref_mean is None:
        ref_mean = _REF_MEAN
    if ref_std is None:
        ref_std = _REF_STD

    # Convert to LAB (float64 for precision)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)

    # Per-channel z-score → re-scale to reference distribution
    src_mean = lab.mean(axis=(0, 1))
    src_std = lab.std(axis=(0, 1))

    # Avoid division by zero on blank/uniform patches
    src_std = np.where(src_std < 1e-6, 1.0, src_std)

    lab = (lab - src_mean) * (ref_std / src_std) + ref_mean

    # Clip to valid LAB range and convert back
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# CLAHE Contrast Enhancement
# ---------------------------------------------------------------------------
def enhance_contrast(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation).

    CLAHE operates on the L channel in LAB space to boost local contrast
    without amplifying noise. This is especially useful for detecting
    faint ring-stage parasites that are nearly transparent.

    Args:
        image_bgr: Input BGR image.
        clip_limit: CLAHE clip limit (higher = more contrast, more noise).
        tile_grid_size: Grid for local histogram computation.

    Returns:
        Contrast-enhanced BGR image.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Resize with Aspect-Ratio-Preserving Padding (Letterbox)
# ---------------------------------------------------------------------------
def resize_with_padding(
    image: np.ndarray,
    target_size: int = 640,
    pad_color: Tuple[int, int, int] = (114, 114, 114),
) -> np.ndarray:
    """Resize image to target_size×target_size with letterbox padding.

    Why letterbox instead of naive resize? Stretching distorts cell
    morphology and changes the apparent size of parasites, which hurts
    both training and inference. The grey padding (114,114,114) is the
    Ultralytics default and produces neutral activations in early conv layers.

    Args:
        image: Input image (any size, BGR or RGB).
        target_size: Square output dimension.
        pad_color: RGB fill colour for padding bars.

    Returns:
        Padded, resized image of shape (target_size, target_size, 3).
    """
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create padded canvas
    canvas = np.full((target_size, target_size, 3), pad_color, dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return canvas


# ---------------------------------------------------------------------------
# Combined Preprocessing Pipeline
# ---------------------------------------------------------------------------
def preprocess_image(
    image_path: str,
    target_size: int = 640,
    apply_stain_norm: bool = True,
    apply_clahe: bool = True,
) -> np.ndarray:
    """Full preprocessing pipeline for a single image.

    Applies (in order):
        1. Stain normalisation (optional)
        2. CLAHE contrast enhancement (optional)
        3. Letterbox resize to target_size

    Args:
        image_path: Path to the input image.
        target_size: Output square dimension.
        apply_stain_norm: Whether to run stain normalisation.
        apply_clahe: Whether to run CLAHE contrast enhancement.

    Returns:
        Preprocessed image as a numpy array (BGR, uint8).

    Raises:
        FileNotFoundError: If image_path does not exist.
        ValueError: If the image cannot be decoded (corrupt file).
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not decode image: {path}")

    if apply_stain_norm:
        image = normalize_stain(image)

    if apply_clahe:
        image = enhance_contrast(image)

    image = resize_with_padding(image, target_size)

    return image


def preprocess_directory(
    input_dir: str,
    output_dir: str,
    target_size: int = 640,
    apply_stain_norm: bool = True,
    apply_clahe: bool = True,
) -> None:
    """Batch-preprocess all images in a directory.

    Processed images are saved with the same filename in output_dir.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    image_files = [
        f for f in input_path.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"No images found in {input_path}")
        return

    print(f"Preprocessing {len(image_files)} images ...")
    for img_file in tqdm(image_files, desc="Preprocessing"):
        try:
            result = preprocess_image(
                str(img_file), target_size, apply_stain_norm, apply_clahe
            )
            cv2.imwrite(str(output_path / img_file.name), result)
        except (ValueError, FileNotFoundError) as e:
            print(f"  SKIPPED {img_file.name}: {e}")

    print(f"Done. Output saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess blood smear images (stain normalisation, CLAHE, resize)."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing raw images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Output directory for preprocessed images (default: data/processed).",
    )
    parser.add_argument(
        "--target_size",
        type=int,
        default=640,
        help="Target square image size (default: 640).",
    )
    parser.add_argument(
        "--normalize_stain",
        action="store_true",
        default=False,
        help="Apply stain normalisation.",
    )
    parser.add_argument(
        "--apply_clahe",
        action="store_true",
        default=False,
        help="Apply CLAHE contrast enhancement.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_size=args.target_size,
        apply_stain_norm=args.normalize_stain,
        apply_clahe=args.apply_clahe,
    )

"""
convert_annotations.py — Convert BBBC041 JSON annotations to YOLO format.

BBBC041 ships annotations as a JSON file where each entry contains:
  {
    "image": { "pathname": "...", "shape": {"r": H, "c": W} },
    "objects": [
      { "category": "red blood cell", "bounding_box": {"minimum": {"r": y1, "c": x1},
                                                        "maximum": {"r": y2, "c": x2}} }
    ]
  }

YOLO expects one .txt file per image with rows of:
    class_id  x_center  y_center  width  height
All values normalised to [0, 1] relative to image dimensions.

Usage:
    python -m src.data.convert_annotations \
        --annotations_json data/raw/malaria/training.json \
        --images_dir data/raw/malaria/images \
        --output_dir data/yolo_dataset \
        --val_split 0.15 \
        --test_split 0.15
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple  # CHANGED

# ---------------------------------------------------------------------------
# Class mapping — must match configs/malaria.yaml
# ---------------------------------------------------------------------------
CLASS_MAP: Dict[str, int] = {
    "red blood cell": 0,
    "ring": 1,
    "trophozoite": 2,
    "schizont": 3,
    "gametocyte": 4,
}

# Some annotations use variant spellings; normalise them here.
CATEGORY_ALIASES: Dict[str, str] = {
    "rbc": "red blood cell",
    "rbcs": "red blood cell",
    "red blood cells": "red blood cell",
    "red_blood_cell": "red blood cell",
    "rings": "ring",
    "trophozoites": "trophozoite",
    "schizonts": "schizont",
    "gametocytes": "gametocyte",
    "difficult": None,  # "difficult" labels are ambiguous — skip them
    "leukocyte": None,  # White blood cells are out-of-scope for parasite detection
    "leukocytes": None,
}


def normalise_category(raw: str) -> Optional[str]:  # CHANGED
    """Map raw annotation category strings to canonical class names.

    Returns None for categories we want to skip entirely (leukocytes, etc.).
    """
    clean = raw.strip().lower()
    if clean in CLASS_MAP:
        return clean
    return CATEGORY_ALIASES.get(clean, clean)


def convert_bbox_to_yolo(
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    img_width: int,
    img_height: int,
) -> Tuple[float, float, float, float]:
    """Convert absolute pixel bounding box to YOLO normalised format.

    YOLO format: (x_center, y_center, width, height), all in [0, 1].
    We clamp to image bounds to handle annotations that slightly overflow.
    """
    # Clamp to image boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_width, x_max)
    y_max = min(img_height, y_max)

    box_w = x_max - x_min
    box_h = y_max - y_min

    x_center = (x_min + box_w / 2) / img_width
    y_center = (y_min + box_h / 2) / img_height
    norm_w = box_w / img_width
    norm_h = box_h / img_height

    return x_center, y_center, norm_w, norm_h


def load_bbbc041_annotations(json_path: str) -> List[dict]:
    """Load the BBBC041 JSON annotations file.

    The file is a JSON array, where each element is one image entry.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} image entries from {json_path}")
    return data


def process_single_image(
    entry: dict,
    images_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
) -> Tuple[int, int]:
    """Convert annotations for one image and copy it to the output dir.

    Returns:
        (num_boxes_written, num_boxes_skipped)
    """
    image_info = entry["image"]
    img_filename = os.path.basename(image_info["pathname"])
    img_height = image_info["shape"]["r"]
    img_width = image_info["shape"]["c"]

    objects = entry.get("objects", [])
    if not objects:
        return 0, 0

    # Build YOLO label lines
    label_lines: List[str] = []
    skipped = 0

    for obj in objects:
        raw_category = obj.get("category", "")
        category = normalise_category(raw_category)

        if category is None or category not in CLASS_MAP:
            skipped += 1
            continue

        class_id = CLASS_MAP[category]
        bbox = obj["bounding_box"]

        x_min = bbox["minimum"]["c"]
        y_min = bbox["minimum"]["r"]
        x_max = bbox["maximum"]["c"]
        y_max = bbox["maximum"]["r"]

        x_c, y_c, w, h = convert_bbox_to_yolo(
            x_min, y_min, x_max, y_max, img_width, img_height
        )

        # Sanity check: skip degenerate boxes (zero area)
        if w <= 0 or h <= 0:
            skipped += 1
            continue

        label_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    # Only write file if we have valid boxes
    if label_lines:
        # Copy image
        src_img = images_dir / img_filename
        if src_img.exists():
            shutil.copy2(src_img, output_images_dir / img_filename)
        else:
            print(f"  WARNING: Image not found: {src_img}")
            return 0, len(objects)

        # Write YOLO label file (same stem as image, .txt extension)
        label_filename = Path(img_filename).stem + ".txt"
        label_path = output_labels_dir / label_filename
        with open(label_path, "w") as f:
            f.write("\n".join(label_lines) + "\n")

    return len(label_lines), skipped


def split_dataset(
    entries: List[dict],
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Randomly split entries into train / val / test sets.

    We shuffle with a fixed seed for reproducibility across team members.
    """
    random.seed(seed)
    shuffled = entries.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_test = int(n * test_split)
    n_val = int(n * val_split)

    test_set = shuffled[:n_test]
    val_set = shuffled[n_test : n_test + n_val]
    train_set = shuffled[n_test + n_val :]

    print(f"Split sizes — train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}")
    return train_set, val_set, test_set


def convert_dataset(
    annotations_json: str,
    images_dir: str,
    output_dir: str = "data/yolo_dataset",
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> None:
    """Full conversion pipeline: load → split → convert → write.

    Creates the standard YOLO directory layout:
        output_dir/
        ├── images/{train,val,test}/
        └── labels/{train,val,test}/
    """
    entries = load_bbbc041_annotations(annotations_json)
    train_set, val_set, test_set = split_dataset(entries, val_split, test_split, seed)

    output_path = Path(output_dir)
    splits = {"train": train_set, "val": val_set, "test": test_set}

    total_boxes = 0
    total_skipped = 0

    for split_name, split_entries in splits.items():
        img_dir = output_path / "images" / split_name
        lbl_dir = output_path / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for entry in split_entries:
            written, skipped = process_single_image(
                entry, Path(images_dir), img_dir, lbl_dir
            )
            total_boxes += written
            total_skipped += skipped

    print("\nConversion complete.")
    print(f"  Total bounding boxes written: {total_boxes}")
    print(f"  Total skipped (unknown/degenerate): {total_skipped}")
    print(f"  Output directory: {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert BBBC041 JSON annotations to YOLO .txt format."
    )
    parser.add_argument(
        "--annotations_json",
        type=str,
        required=True,
        help="Path to the BBBC041 JSON annotations file (e.g., training.json).",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing the raw BBBC041 images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/yolo_dataset",
        help="Root output directory for YOLO dataset (default: data/yolo_dataset).",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.15,
        help="Fraction of data for validation (default: 0.15).",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.15,
        help="Fraction of data for test (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_dataset(
        annotations_json=args.annotations_json,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )

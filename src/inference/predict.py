"""
predict.py — Run inference on a single image or batch of images.

Loads a trained YOLOv8 model and returns structured detection results:
bounding boxes, class labels, confidence scores, and an annotated image.

This module is used by:
  • The Streamlit app (app/streamlit_app.py) for interactive demos
  • CLI batch processing for generating submission files
  • The evaluation pipeline for qualitative inspection

Usage:
    # Single image
    python -m src.inference.predict \
        --weights models/best.pt \
        --source path/to/image.png \
        --save_dir results/predictions

    # Folder of images
    python -m src.inference.predict \
        --weights models/best.pt \
        --source path/to/images_folder/ \
        --save_dir results/predictions \
        --conf 0.3
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Class names — must match configs/malaria.yaml
# ---------------------------------------------------------------------------
CLASS_NAMES = ["ring", "trophozoite", "schizont", "gametocyte", "red_blood_cell"]  # CHANGED: Order matches malaria.yaml

# Parasite classes (everything except healthy RBCs)
PARASITE_CLASSES = {"ring", "trophozoite", "schizont", "gametocyte"}

# Colours for bounding box visualisation (BGR for OpenCV)
CLASS_COLORS = {
    "red_blood_cell": (200, 200, 200),  # Light grey — don't distract from parasites
    "ring": (0, 0, 255),                # Red — most common stage, needs to pop
    "trophozoite": (0, 165, 255),       # Orange
    "schizont": (0, 255, 255),          # Yellow
    "gametocyte": (255, 0, 255),        # Magenta — distinctive for rare stage
}

# CHANGED: Uncertainty classification thresholds for clinical safety
# Only detections with confidence in [LOW, HIGH] are flagged as "uncertain".
# Detections below LOW that still pass the user's confidence slider are drawn normally.
UNCERTAINTY_THRESHOLD_LOW = 0.35    # Lower bound for uncertain tier
UNCERTAINTY_THRESHOLD_HIGH = 0.55   # Upper bound — above this is "confident"
UNCERTAIN_COLOR = (0, 255, 255)     # Yellow (BGR) for uncertain detections


@dataclass
class Detection:
    """Single detected object."""
    class_name: str
    class_id: int
    confidence: float
    bbox_xyxy: List[float]  # [x1, y1, x2, y2] in pixels
    bbox_xywh: List[float]  # [x_center, y_center, width, height] in pixels


@dataclass
class PredictionResult:
    """Aggregated prediction results for one image."""
    image_path: str
    detections: List[Detection] = field(default_factory=list)
    annotated_image: Optional[np.ndarray] = None
    total_rbc: int = 0
    total_parasites: int = 0
    parasitemia_pct: float = 0.0

    def compute_parasitemia(self) -> None:
        """Estimate parasitemia percentage.

        Parasitemia = (infected RBCs / total RBCs) × 100
        We count parasite detections as proxies for infected RBCs.
        This is a simplification — in reality, one RBC can harbour
        multiple ring-stage parasites, but for a competition demo this
        gives a reasonable estimate.
        """
        self.total_rbc = sum(
            1 for d in self.detections if d.class_name == "red_blood_cell"
        )
        self.total_parasites = sum(
            1 for d in self.detections if d.class_name in PARASITE_CLASSES
        )

        # Total cells ≈ healthy RBCs + infected cells
        total_cells = self.total_rbc + self.total_parasites
        if total_cells > 0:
            self.parasitemia_pct = (self.total_parasites / total_cells) * 100
        else:
            self.parasitemia_pct = 0.0

    def summary(self) -> Dict[str, any]:
        """Return a JSON-serialisable summary."""
        return {
            "image": self.image_path,
            "total_detections": len(self.detections),
            "total_rbc": self.total_rbc,
            "total_parasites": self.total_parasites,
            "parasitemia_pct": round(self.parasitemia_pct, 2),
            "per_class_counts": self._per_class_counts(),
        }

    def _per_class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for d in self.detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return counts


class MalariaDetector:
    """High-level wrapper around the trained YOLO model.

    Provides a clean API for the Streamlit app and CLI:
        detector = MalariaDetector("models/best.pt")
        result = detector.predict("slide.png", conf=0.25)
        annotated = result.annotated_image
    """

    def __init__(self, weights_path: str, device: str = "cpu"):
        """Load the model once; reuse for multiple predictions.

        Args:
            weights_path: Path to the .pt weights file.
            device: 'cpu' or GPU index string.
        """
        self.model = YOLO(weights_path)
        self.device = device
        print(f"Loaded model from {weights_path} (device={device})")

    def predict(
        self,
        image_source: str | np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        annotate: bool = True,
    ) -> PredictionResult:
        """Run detection on a single image.

        Args:
            image_source: File path (str) or numpy array (BGR).
            conf: Minimum confidence threshold.
            iou: NMS IoU threshold.
            imgsz: Inference image size.
            annotate: If True, generate an annotated image with boxes drawn.

        Returns:
            PredictionResult with detections and (optionally) annotated image.
        """
        # Run YOLO inference
        results = self.model.predict(
            source=image_source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
        )

        result = results[0]  # Single image → single result

        # Parse detections
        detections: List[Detection] = []
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
                conf_val = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                xywh = boxes.xywh[i].cpu().numpy().tolist()

                detections.append(Detection(
                    class_name=cls_name,
                    class_id=cls_id,
                    confidence=conf_val,
                    bbox_xyxy=xyxy,
                    bbox_xywh=xywh,
                ))

        # Build annotated image
        annotated_img = None
        if annotate:
            annotated_img = self._draw_detections(result, detections)

        # Determine image path
        img_path = image_source if isinstance(image_source, str) else "<numpy_array>"

        pred_result = PredictionResult(
            image_path=img_path,
            detections=detections,
            annotated_image=annotated_img,
        )
        pred_result.compute_parasitemia()

        return pred_result

    def _draw_detections(
        self,
        yolo_result: any,
        detections: List[Detection],
    ) -> np.ndarray:
        """Draw bounding boxes with class-specific colours.

        We draw our own boxes instead of using the Ultralytics plot()
        method because:
          1. We want clinically meaningful colour coding (red for infected)
          2. We need control over label placement for the demo
        """
        img = yolo_result.orig_img.copy()

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]

            # CHANGED: Classify detection into uncertainty tiers for clinical safety.
            # Gate on BOTH bounds: only 0.35 <= conf <= 0.55 is "uncertain".
            # Detections below 0.35 that passed the slider are drawn normally.
            is_uncertain = (
                UNCERTAINTY_THRESHOLD_LOW <= det.confidence <= UNCERTAINTY_THRESHOLD_HIGH
            )

            if is_uncertain:
                # CHANGED: Uncertain tier — yellow box, thicker border, generic label
                color = UNCERTAIN_COLOR
                thickness = 3
                label = f"INCONCLUSIVE {det.confidence:.2f}"
            else:
                # Confident tier (>0.55) or sub-threshold (<0.35) — normal class color
                color = CLASS_COLORS.get(det.class_name, (255, 255, 255))
                thickness = 2 if det.class_name in PARASITE_CLASSES else 1
                label = f"{det.class_name} {det.confidence:.2f}"

            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # Draw label background + text
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                img, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
            )

        return img

    def predict_batch(
        self,
        source_dir: str,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        save_dir: Optional[str] = None,
    ) -> List[PredictionResult]:
        """Run inference on all images in a directory.

        Optionally saves annotated images to save_dir.
        """
        source_path = Path(source_dir)
        image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        image_files = sorted(
            f for f in source_path.iterdir()
            if f.suffix.lower() in image_extensions
        )

        if not image_files:
            print(f"No images found in {source_path}")
            return []

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        results: List[PredictionResult] = []
        for img_file in image_files:
            pred = self.predict(str(img_file), conf, iou, imgsz)
            results.append(pred)

            if save_dir and pred.annotated_image is not None:
                save_path = Path(save_dir) / f"pred_{img_file.name}"
                cv2.imwrite(str(save_path), pred.annotated_image)

        print(f"Processed {len(results)} images.")
        return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 malaria detection inference."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="models/best.pt",
        help="Path to trained model weights.",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to a single image or directory of images.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/predictions",
        help="Directory to save annotated images.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold (default: 0.45).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device: 'cpu' or GPU index.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    detector = MalariaDetector(args.weights, device=args.device)

    source = Path(args.source)
    if source.is_dir():
        results = detector.predict_batch(
            str(source), args.conf, args.iou, args.imgsz, args.save_dir
        )
        for r in results:
            print(r.summary())
    else:
        result = detector.predict(str(source), args.conf, args.iou, args.imgsz)
        print(result.summary())

        if result.annotated_image is not None:
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            save_path = Path(args.save_dir) / f"pred_{source.name}"
            cv2.imwrite(str(save_path), result.annotated_image)
            print(f"Annotated image saved to {save_path}")

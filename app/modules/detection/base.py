"""
Base classes for Detection Modules
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class Detection:
    """Single detected object."""
    class_name: str
    class_id: int
    confidence: float
    bbox_xyxy: List[float]
    bbox_xywh: List[float]


@dataclass
class BaseResult:
    """Base result class for detection modules."""
    image_path: str
    detections: List[Detection] = field(default_factory=list)
    annotated_image: Optional[np.ndarray] = None
    inference_time_sec: float = 0.0

    def summary(self) -> Dict:
        return {
            "image": self.image_path,
            "total_detections": len(self.detections),
            "inference_time_sec": round(self.inference_time_sec, 4),
            "per_class_counts": self._per_class_counts(),
        }

    def _per_class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for d in self.detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return counts


class BaseDetector:
    """Base class for all detectors."""

    CLASS_NAMES: List[str] = []
    CLASS_COLORS: Dict[str, tuple] = {}
    TARGET_CLASSES: set = set()
    UNCERTAINTY_LOW: float = 0.35
    UNCERTAINTY_HIGH: float = 0.45

    def __init__(self, weights_path: str, device: str = "cpu"):
        from ultralytics import YOLO
        self.model = YOLO(weights_path)
        self.device = device

    def predict(self, image_source, conf=0.25, iou=0.45, imgsz=640, annotate=True):
        raise NotImplementedError

    def _draw_detections(self, yolo_result, detections: List[Detection]) -> np.ndarray:
        import cv2
        img = yolo_result.orig_img.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
            is_uncertain = self.UNCERTAINTY_LOW <= det.confidence <= self.UNCERTAINTY_HIGH
            if is_uncertain:
                color = (0, 255, 255)
                thickness = 3
                label = f"INCONCLUSIVE {det.confidence:.2f}"
                text_color = (0, 0, 0)
            else:
                color = self.CLASS_COLORS.get(det.class_name, (255, 255, 255))
                is_target = det.class_name in self.TARGET_CLASSES
                thickness = 2 if is_target else 1
                label = f"{det.class_name} {det.confidence:.2f}"
                text_color = (255, 255, 255)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)
        return img
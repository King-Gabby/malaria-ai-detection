"""
X-ray Module — Chest Pathology Detection
Uses YOLOv8n for detection.
Supports CheXpert and NIH ChestX-ray datasets.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO


@dataclass
class XRayDetection:
    """Single detected pathology in X-ray."""
    class_name: str
    class_id: int
    confidence: float
    bbox_xyxy: List[float]
    bbox_xywh: List[float]
    laterality: str = "unknown"  # left, right, bilateral


@dataclass
class XRayResult:
    """Aggregated X-ray analysis results."""
    image_path: str
    detections: List[XRayDetection] = field(default_factory=list)
    annotated_image: Optional[np.ndarray] = None
    inference_time_sec: float = 0.0

    def summary(self) -> Dict:
        return {
            "image": self.image_path,
            "total_detections": len(self.detections),
            "pneumonia_count": sum(1 for d in self.detections if d.class_name == "pneumonia"),
            "tb_count": sum(1 for d in self.detections if d.class_name == "tb"),
            "cardiomegaly_count": sum(1 for d in self.detections if d.class_name == "cardiomegaly"),
            "inference_time_sec": round(self.inference_time_sec, 4),
            "per_class_counts": self._per_class_counts(),
        }

    def _per_class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for d in self.detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return counts


class XRayDetector:
    """High-level wrapper for X-ray analysis."""

    CLASS_NAMES = ["pneumonia", "tb", "cardiomegaly", "normal"]
    CLASS_COLORS = {
        "pneumonia": (0, 0, 255),      # Red
        "tb": (0, 165, 255),           # Orange
        "cardiomegaly": (255, 0, 255), # Magenta
        "normal": (200, 200, 200),     # Grey
    }

    def __init__(self, weights_path: str, device: str = "cpu"):
        self.model = YOLO(weights_path)
        self.device = device
        print(f"Loaded X-ray detector from {weights_path} (device={device})")

    def predict(
        self,
        image_source: str | np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        annotate: bool = True,
    ) -> XRayResult:
        start_time = time.perf_counter()

        results = self.model.predict(
            source=image_source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
        )

        detections: List[XRayDetection] = []
        det_result = results[0]
        boxes = det_result.boxes

        if boxes is not None and len(boxes) > 0:
            img_w = det_result.orig_img.shape[1]
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                cls_name = self.CLASS_NAMES[cls_id] if cls_id < len(self.CLASS_NAMES) else f"class_{cls_id}"
                conf_val = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                xywh = boxes.xywh[i].cpu().numpy().tolist()

                # Determine laterality based on bbox center
                cx = xywh[0]
                laterality = "left" if cx < img_w / 2 else "right"

                detections.append(XRayDetection(
                    class_name=cls_name,
                    class_id=cls_id,
                    confidence=conf_val,
                    bbox_xyxy=xyxy,
                    bbox_xywh=xywh,
                    laterality=laterality,
                ))

        annotated_img = None
        if annotate:
            annotated_img = self._draw_detections(det_result, detections)

        img_path = image_source if isinstance(image_source, str) else "<numpy_array>"

        return XRayResult(
            image_path=img_path,
            detections=detections,
            annotated_image=annotated_img,
            inference_time_sec=time.perf_counter() - start_time,
        )

    def _draw_detections(self, yolo_result, detections: List[XRayDetection]) -> np.ndarray:
        img = yolo_result.orig_img.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
            color = self.CLASS_COLORS.get(det.class_name, (255, 255, 255))
            thickness = 2
            label = f"{det.class_name} {det.confidence:.2f} ({det.laterality})"

            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def predict_batch(
        self,
        source_dir: str,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        save_dir: Optional[str] = None,
    ) -> List[XRayResult]:
        source_path = Path(source_dir)
        image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dcm"}
        image_files = sorted(f for f in source_path.iterdir() if f.suffix.lower() in image_extensions)

        if not image_files:
            print(f"No images found in {source_path}")
            return []

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        results: List[XRayResult] = []
        for img_file in image_files:
            pred = self.predict(str(img_file), conf, iou, imgsz)
            results.append(pred)
            if save_dir and pred.annotated_image is not None:
                save_path = Path(save_dir) / f"pred_{img_file.name}"
                cv2.imwrite(str(save_path), pred.annotated_image)

        print(f"Processed {len(results)} X-ray images.")
        return results


@st.cache_resource
def load_xray_model(weights_path: str, device: str = "cpu") -> XRayDetector | None:
    """Cached model loader for Streamlit - LOCAL FILES ONLY."""
    if not Path(weights_path).exists():
        print(f"X-ray model weights not found locally: {weights_path}")
        return None
    
    try:
        return XRayDetector(weights_path, device)
    except Exception as e:
        print(f"Failed to load X-ray model: {e}")
        return None
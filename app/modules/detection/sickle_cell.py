"""
Sickle Cell Disease Detection Module
Uses YOLOv8n for detecting Normal RBC, Sickle Cell, Target Cell, Spherocyte.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO


CLASS_NAMES = ["normal_rbc", "sickle_cell", "target_cell", "spherocyte"]
ABNORMAL_CLASSES = {"sickle_cell", "target_cell", "spherocyte"}

CLASS_COLORS = {
    "normal_rbc": (200, 200, 200),
    "sickle_cell": (0, 0, 255),
    "target_cell": (0, 165, 255),
    "spherocyte": (255, 0, 255),
}

UNCERTAINTY_THRESHOLD_LOW = 0.35
UNCERTAINTY_THRESHOLD_HIGH = 0.45
UNCERTAIN_COLOR = (0, 255, 255)


@dataclass
class Detection:
    class_name: str
    class_id: int
    confidence: float
    bbox_xyxy: List[float]
    bbox_xywh: List[float]


@dataclass
class SickleCellResult:
    image_path: str
    detections: List[Detection] = field(default_factory=list)
    annotated_image: Optional[np.ndarray] = None
    total_normal: int = 0
    total_abnormal: int = 0
    sickle_count: int = 0
    target_count: int = 0
    spherocyte_count: int = 0
    sickle_percentage: float = 0.0
    inference_time_sec: float = 0.0

    def compute_metrics(self) -> None:
        self.total_normal = sum(1 for d in self.detections if d.class_name == "normal_rbc")
        self.sickle_count = sum(1 for d in self.detections if d.class_name == "sickle_cell")
        self.target_count = sum(1 for d in self.detections if d.class_name == "target_cell")
        self.spherocyte_count = sum(1 for d in self.detections if d.class_name == "spherocyte")
        self.total_abnormal = self.sickle_count + self.target_count + self.spherocyte_count

        total = self.total_normal + self.total_abnormal
        if total > 0:
            self.sickle_percentage = (self.total_abnormal / total) * 100
        else:
            self.sickle_percentage = 0.0

    def summary(self) -> Dict:
        return {
            "image": self.image_path,
            "total_detections": len(self.detections),
            "normal_rbc": self.total_normal,
            "sickle_cell": self.sickle_count,
            "target_cell": self.target_count,
            "spherocyte": self.spherocyte_count,
            "abnormal_percentage": round(self.sickle_percentage, 2),
            "inference_time_sec": round(self.inference_time_sec, 4),
            "per_class_counts": self._per_class_counts(),
        }

    def _per_class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for d in self.detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return counts


class SickleCellDetector:
    def __init__(self, weights_path: str, device: str = "cpu"):
        self.model = YOLO(weights_path)
        self.device = device
        print(f"Loaded sickle cell model from {weights_path} (device={device})")

    def predict(
        self,
        image_source: str | np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        annotate: bool = True,
    ) -> SickleCellResult:
        start_time = time.perf_counter()
        results = self.model.predict(
            source=image_source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
        )
        inference_time_sec = time.perf_counter() - start_time

        result = results[0]
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

        annotated_img = None
        if annotate:
            annotated_img = self._draw_detections(result, detections)

        img_path = image_source if isinstance(image_source, str) else "<numpy_array>"

        pred_result = SickleCellResult(
            image_path=img_path,
            detections=detections,
            annotated_image=annotated_img,
            inference_time_sec=inference_time_sec,
        )
        pred_result.compute_metrics()
        return pred_result

    def _draw_detections(self, yolo_result, detections: List[Detection]) -> np.ndarray:
        img = yolo_result.orig_img.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]

            is_uncertain = UNCERTAINTY_THRESHOLD_LOW <= det.confidence <= UNCERTAINTY_THRESHOLD_HIGH
            if is_uncertain:
                color = UNCERTAIN_COLOR
                thickness = 3
                label = f"INCONCLUSIVE {det.confidence:.2f}"
                text_color = (0, 0, 0)
            else:
                color = CLASS_COLORS.get(det.class_name, (255, 255, 255))
                thickness = 2 if det.class_name in ABNORMAL_CLASSES else 1
                label = f"{det.class_name} {det.confidence:.2f}"
                text_color = (255, 255, 255)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)
        return img

    def predict_batch(
        self,
        source_dir: str,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        save_dir: Optional[str] = None,
    ) -> List[SickleCellResult]:
        source_path = Path(source_dir)
        image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        image_files = sorted(f for f in source_path.iterdir() if f.suffix.lower() in image_extensions)

        if not image_files:
            print(f"No images found in {source_path}")
            return []

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        results: List[SickleCellResult] = []
        for img_file in image_files:
            pred = self.predict(str(img_file), conf, iou, imgsz)
            results.append(pred)
            if save_dir and pred.annotated_image is not None:
                save_path = Path(save_dir) / f"pred_{img_file.name}"
                cv2.imwrite(str(save_path), pred.annotated_image)

        print(f"Processed {len(results)} images.")
        return results


@st.cache_resource
def load_sickle_cell_model(weights_path: str, device: str = "cpu") -> SickleCellDetector | None:
    """Cached model loader for Streamlit - LOCAL FILES ONLY."""
    if not Path(weights_path).exists():
        print(f"Sickle cell model weights not found locally: {weights_path}")
        return None
    
    try:
        return SickleCellDetector(weights_path, device)
    except Exception as e:
        print(f"Failed to load sickle cell model: {e}")
        return None
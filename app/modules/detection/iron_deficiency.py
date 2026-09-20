"""
Iron Deficiency Anemia Detection Module
Uses YOLOv8n for detecting Microcyte, Hypochromic, Normal RBC, Pencil Cell.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO


CLASS_NAMES = ["microcyte", "hypochromic", "normal_rbc", "pencil_cell"]
ABNORMAL_CLASSES = {"microcyte", "hypochromic", "pencil_cell"}

CLASS_COLORS = {
    "normal_rbc": (200, 200, 200),
    "microcyte": (0, 0, 255),
    "hypochromic": (0, 165, 255),
    "pencil_cell": (255, 0, 255),
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
class IronDeficiencyResult:
    image_path: str
    detections: List[Detection] = field(default_factory=list)
    annotated_image: Optional[np.ndarray] = None
    total_cells: int = 0
    normal_count: int = 0
    microcyte_count: int = 0
    hypochromic_count: int = 0
    pencil_count: int = 0
    abnormal_percentage: float = 0.0
    inference_time_sec: float = 0.0

    def compute_metrics(self) -> None:
        self.normal_count = sum(1 for d in self.detections if d.class_name == "normal_rbc")
        self.microcyte_count = sum(1 for d in self.detections if d.class_name == "microcyte")
        self.hypochromic_count = sum(1 for d in self.detections if d.class_name == "hypochromic")
        self.pencil_count = sum(1 for d in self.detections if d.class_name == "pencil_cell")
        total_abnormal = self.microcyte_count + self.hypochromic_count + self.pencil_count
        self.total_cells = self.normal_count + total_abnormal

        if self.total_cells > 0:
            self.abnormal_percentage = (total_abnormal / self.total_cells) * 100
        else:
            self.abnormal_percentage = 0.0

    def summary(self) -> Dict:
        return {
            "image": self.image_path,
            "total_detections": len(self.detections),
            "normal_rbc": self.normal_count,
            "microcyte": self.microcyte_count,
            "hypochromic": self.hypochromic_count,
            "pencil_cell": self.pencil_count,
            "abnormal_percentage": round(self.abnormal_percentage, 2),
            "inference_time_sec": round(self.inference_time_sec, 4),
            "per_class_counts": self._per_class_counts(),
        }

    def _per_class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for d in self.detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return counts


class IronDeficiencyDetector:
    def __init__(self, weights_path: str, device: str = "cpu"):
        self.model = YOLO(weights_path)
        self.device = device
        print(f"Loaded iron deficiency model from {weights_path} (device={device})")

    def predict(
        self,
        image_source: str | np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        annotate: bool = True,
    ) -> IronDeficiencyResult:
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

        pred_result = IronDeficiencyResult(
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
    ) -> List[IronDeficiencyResult]:
        source_path = Path(source_dir)
        image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        image_files = sorted(f for f in source_path.iterdir() if f.suffix.lower() in image_extensions)

        if not image_files:
            print(f"No images found in {source_path}")
            return []

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        results: List[IronDeficiencyResult] = []
        for img_file in image_files:
            pred = self.predict(str(img_file), conf, iou, imgsz)
            results.append(pred)
            if save_dir and pred.annotated_image is not None:
                save_path = Path(save_dir) / f"pred_{img_file.name}"
                cv2.imwrite(str(save_path), pred.annotated_image)

        print(f"Processed {len(results)} images.")
        return results


@st.cache_resource
def load_iron_deficiency_model(weights_path: str, device: str = "cpu") -> IronDeficiencyDetector | None:
    try:
        return IronDeficiencyDetector(weights_path, device)
    except Exception as e:
        print(f"Failed to load iron deficiency model: {e}")
        return None
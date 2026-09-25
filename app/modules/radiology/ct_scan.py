"""
CT Scan Module — Lung Nodule & Pathology Detection
Uses YOLOv8n for detection.
Supports LIDC and COVID-CT datasets.
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
class CTDetection:
    """Single detected object in CT scan."""
    class_name: str
    class_id: int
    confidence: float
    bbox_xyxy: List[float]
    bbox_xywh: List[float]
    slice_idx: int = 0
    volume_mm3: float = 0.0
    diameter_mm: float = 0.0


@dataclass
class CTResult:
    """Aggregated CT analysis results."""
    image_path: str
    detections: List[CTDetection] = field(default_factory=list)
    annotated_image: Optional[np.ndarray] = None
    total_nodule_volume_mm3: float = 0.0
    inference_time_sec: float = 0.0
    slice_thickness_mm: float = 1.0
    pixel_spacing_mm: float = 1.0

    def compute_volumes(self) -> None:
        """Compute nodule volumes from detections."""
        for det in self.detections:
            if det.class_name == "nodule":
                w, h = det.bbox_xywh[2], det.bbox_xywh[3]
                # Approximate as sphere: diameter = mean of width/height in mm
                det.diameter_mm = ((w + h) / 2) * self.pixel_spacing_mm
                # Sphere volume = 4/3 * pi * r^3
                r = det.diameter_mm / 2
                det.volume_mm3 = (4/3) * np.pi * (r ** 3)
                self.total_nodule_volume_mm3 += det.volume_mm3

    def summary(self) -> Dict:
        return {
            "image": self.image_path,
            "total_detections": len(self.detections),
            "nodule_count": sum(1 for d in self.detections if d.class_name == "nodule"),
            "consolidation_count": sum(1 for d in self.detections if d.class_name == "consolidation"),
            "effusion_count": sum(1 for d in self.detections if d.class_name == "effusion"),
            "total_nodule_volume_mm3": round(self.total_nodule_volume_mm3, 2),
            "inference_time_sec": round(self.inference_time_sec, 4),
            "per_class_counts": self._per_class_counts(),
        }

    def _per_class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for d in self.detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return counts


class CTScanDetector:
    """High-level wrapper for CT scan analysis."""

    CLASS_NAMES = ["nodule", "consolidation", "effusion", "normal"]
    CLASS_COLORS = {
        "nodule": (0, 0, 255),        # Red
        "consolidation": (0, 165, 255),  # Orange
        "effusion": (0, 255, 255),    # Yellow
        "normal": (200, 200, 200),    # Grey
    }

    def __init__(self, weights_path: str, device: str = "cpu"):
        self.model = YOLO(weights_path)
        self.device = device
        print(f"Loaded CT detector from {weights_path} (device={device})")

    def predict(
        self,
        image_source: str | np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        annotate: bool = True,
        slice_thickness_mm: float = 1.0,
        pixel_spacing_mm: float = 1.0,
    ) -> CTResult:
        start_time = time.perf_counter()

        results = self.model.predict(
            source=image_source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
        )

        detections: List[CTDetection] = []
        det_result = results[0]
        boxes = det_result.boxes

        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                cls_name = self.CLASS_NAMES[cls_id] if cls_id < len(self.CLASS_NAMES) else f"class_{cls_id}"
                conf_val = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                xywh = boxes.xywh[i].cpu().numpy().tolist()

                detections.append(CTDetection(
                    class_name=cls_name,
                    class_id=cls_id,
                    confidence=conf_val,
                    bbox_xyxy=xyxy,
                    bbox_xywh=xywh,
                ))

        annotated_img = None
        if annotate:
            annotated_img = self._draw_detections(det_result, detections)

        img_path = image_source if isinstance(image_source, str) else "<numpy_array>"

        result = CTResult(
            image_path=img_path,
            detections=detections,
            annotated_image=annotated_img,
            inference_time_sec=time.perf_counter() - start_time,
            slice_thickness_mm=slice_thickness_mm,
            pixel_spacing_mm=pixel_spacing_mm,
        )
        result.compute_volumes()
        return result

    def _draw_detections(self, yolo_result, detections: List[CTDetection]) -> np.ndarray:
        img = yolo_result.orig_img.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
            color = self.CLASS_COLORS.get(det.class_name, (255, 255, 255))
            thickness = 2
            label = f"{det.class_name} {det.confidence:.2f}"
            if det.class_name == "nodule" and det.diameter_mm > 0:
                label += f" ({det.diameter_mm:.1f} mm)"

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
    ) -> List[CTResult]:
        source_path = Path(source_dir)
        image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dcm"}
        image_files = sorted(f for f in source_path.iterdir() if f.suffix.lower() in image_extensions)

        if not image_files:
            print(f"No images found in {source_path}")
            return []

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        results: List[CTResult] = []
        for img_file in image_files:
            pred = self.predict(str(img_file), conf, iou, imgsz)
            results.append(pred)
            if save_dir and pred.annotated_image is not None:
                save_path = Path(save_dir) / f"pred_{img_file.name}"
                cv2.imwrite(str(save_path), pred.annotated_image)

        print(f"Processed {len(results)} CT images.")
        return results


@st.cache_resource
def load_ct_model(weights_path: str, device: str = "cpu") -> CTScanDetector | None:
    """Cached model loader for Streamlit - LOCAL FILES ONLY."""
    if not Path(weights_path).exists():
        print(f"CT model weights not found locally: {weights_path}")
        return None
    
    try:
        return CTScanDetector(weights_path, device)
    except Exception as e:
        print(f"Failed to load CT model: {e}")
        return None
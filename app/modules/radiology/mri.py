"""
MRI Module — Brain Tumor Detection & Segmentation
Uses YOLOv8n for detection and UNet for segmentation.
Supports BraTS and fastMRI datasets.
"""

import time
import streamlit as st
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class MRIDetection:
    """Single detected object in MRI."""
    class_name: str
    class_id: int
    confidence: float
    bbox_xyxy: List[float]
    bbox_xywh: List[float]
    slice_idx: int = 0
    volume_mm3: float = 0.0


@dataclass
class MRIResult:
    """Aggregated MRI analysis results."""
    image_path: str
    detections: List[MRIDetection] = field(default_factory=list)
    annotated_image: Optional[np.ndarray] = None
    segmentation_mask: Optional[np.ndarray] = None
    total_tumor_volume_mm3: float = 0.0
    total_edema_volume_mm3: float = 0.0
    inference_time_sec: float = 0.0
    slice_thickness_mm: float = 1.0
    pixel_spacing_mm: float = 1.0

    def compute_volumes(self) -> None:
        """Compute tumor and edema volumes from segmentation."""
        if self.segmentation_mask is not None:
            tumor_pixels = np.sum(self.segmentation_mask == 1)
            edema_pixels = np.sum(self.segmentation_mask == 2)
            voxel_volume = self.slice_thickness_mm * (self.pixel_spacing_mm ** 2)
            self.total_tumor_volume_mm3 = tumor_pixels * voxel_volume
            self.total_edema_volume_mm3 = edema_pixels * voxel_volume

    def summary(self) -> Dict:
        """Return JSON-serializable summary."""
        return {
            "image": self.image_path,
            "total_detections": len(self.detections),
            "tumor_volume_mm3": round(self.total_tumor_volume_mm3, 2),
            "edema_volume_mm3": round(self.total_edema_volume_mm3, 2),
            "inference_time_sec": round(self.inference_time_sec, 4),
            "per_class_counts": self._per_class_counts(),
        }

    def _per_class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for d in self.detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return counts


class MRIDetector:
    """High-level wrapper for MRI analysis (detection + segmentation)."""

    CLASS_NAMES = ["tumor", "edema", "normal_tissue"]
    CLASS_COLORS = {
        "tumor": (0, 0, 255),      # Red
        "edema": (0, 165, 255),    # Orange
        "normal_tissue": (200, 200, 200),  # Grey
    }

    def __init__(self, weights_path: str, seg_weights_path: str = None, device: str = "cpu"):
        self.detector = YOLO(weights_path)
        self.segmenter = YOLO(seg_weights_path) if seg_weights_path else None
        self.device = device
        print(f"Loaded MRI detector from {weights_path} (device={device})")
        if self.segmenter:
            print(f"Loaded MRI segmenter from {seg_weights_path}")

    def predict(
        self,
        image_source: str | np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        annotate: bool = True,
        slice_thickness_mm: float = 1.0,
        pixel_spacing_mm: float = 1.0,
    ) -> MRIResult:
        """Run detection and optional segmentation on MRI slice/volume."""
        start_time = time.perf_counter()

        # Run detection
        det_results = self.detector.predict(
            source=image_source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
        )

        detections: List[MRIDetection] = []
        det_result = det_results[0]
        boxes = det_result.boxes

        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                cls_name = self.CLASS_NAMES[cls_id] if cls_id < len(self.CLASS_NAMES) else f"class_{cls_id}"
                conf_val = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                xywh = boxes.xywh[i].cpu().numpy().tolist()

                # Estimate volume for this detection
                w, h = xywh[2], xywh[3]
                area_mm2 = w * h * (pixel_spacing_mm ** 2)
                volume_mm3 = area_mm2 * slice_thickness_mm

                detections.append(MRIDetection(
                    class_name=cls_name,
                    class_id=cls_id,
                    confidence=conf_val,
                    bbox_xyxy=xyxy,
                    bbox_xywh=xywh,
                    volume_mm3=volume_mm3,
                ))

        # Run segmentation if available
        segmentation_mask = None
        if self.segmenter:
            seg_results = self.segmenter.predict(
                source=image_source,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=self.device,
                verbose=False,
            )
            if seg_results[0].masks is not None:
                segmentation_mask = seg_results[0].masks.data[0].cpu().numpy().astype(np.uint8)

        # Build annotated image
        annotated_img = None
        if annotate:
            annotated_img = self._draw_detections(det_result, detections)

        img_path = image_source if isinstance(image_source, str) else "<numpy_array>"

        result = MRIResult(
            image_path=img_path,
            detections=detections,
            annotated_image=annotated_img,
            segmentation_mask=segmentation_mask,
            inference_time_sec=time.perf_counter() - start_time,
            slice_thickness_mm=slice_thickness_mm,
            pixel_spacing_mm=pixel_spacing_mm,
        )
        result.compute_volumes()
        return result

    def _draw_detections(self, yolo_result, detections: List[MRIDetection]) -> np.ndarray:
        img = yolo_result.orig_img.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
            color = self.CLASS_COLORS.get(det.class_name, (255, 255, 255))
            thickness = 2
            label = f"{det.class_name} {det.confidence:.2f} ({det.volume_mm3:.1f} mm³)"

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
    ) -> List[MRIResult]:
        source_path = Path(source_dir)
        image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dcm", ".nii", ".nii.gz"}
        image_files = sorted(f for f in source_path.iterdir() if f.suffix.lower() in image_extensions)

        if not image_files:
            print(f"No images found in {source_path}")
            return []

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        results: List[MRIResult] = []
        for img_file in image_files:
            pred = self.predict(str(img_file), conf, iou, imgsz)
            results.append(pred)
            if save_dir and pred.annotated_image is not None:
                save_path = Path(save_dir) / f"pred_{img_file.name}"
                cv2.imwrite(str(save_path), pred.annotated_image)

        print(f"Processed {len(results)} MRI images.")
        return results


@st.cache_resource
def load_mri_model(weights_path: str, seg_weights_path: str = None, device: str = "cpu") -> MRIDetector | None:
    """Cached model loader for Streamlit."""
    try:
        return MRIDetector(weights_path, seg_weights_path, device)
    except Exception as e:
        print(f"Failed to load MRI model: {e}")
        return None
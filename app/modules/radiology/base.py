# RaphaID AI - Radiology Base Module

import streamlit as st
import numpy as np
import cv2
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class RadiologyResult:
    """Standardized radiology detection/segmentation result"""
    detections: List[Dict]  # List of {class_name, confidence, bbox_xyxy, bbox_xywh}
    segmentation_masks: Optional[Dict]  # For segmentation tasks
    annotated_image: Optional[np.ndarray]
    inference_time_sec: float
    image_path: str
    per_class_counts: Dict[str, int]
    measurements: Dict[str, float]  # Volume, diameter, etc.


class BaseRadiologyModel(ABC):
    """Base class for all radiology models"""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.class_names = []
        self.confidence_threshold = 0.25
        self.iou_threshold = 0.45
    
    @abstractmethod
    def load_model(self) -> bool:
        """Load the model. Returns True if successful."""
        pass
    
    @abstractmethod
    def predict(self, image: np.ndarray, conf: float = None, iou: float = None) -> RadiologyResult:
        """Run inference on image. Returns RadiologyResult."""
        pass
    
    def get_severity(self, finding_count: int) -> tuple:
        """Get severity based on findings"""
        if finding_count == 0:
            return "No Findings", "🟢"
        elif finding_count <= 2:
            return "Few Findings", "🟡"
        elif finding_count <= 5:
            return "Multiple Findings", "🟠"
        else:
            return "Numerous Findings", "🔴"


class RadiologyUI:
    """Shared UI components for radiology modules"""
    
    def __init__(self, modality_key: str, modality_name: str, modality_icon: str):
        self.modality_key = modality_key
        self.modality_name = modality_name
        self.modality_icon = modality_icon
    
    def render_patient_intake(self):
        """Render patient intake form"""
        with st.expander("🏥 Patient Information", expanded=True):
            st.caption("Details entered here are stored temporarily for this session only.")
            
            pt_col1, pt_col2, pt_col3 = st.columns(3)
            
            with pt_col1:
                p_name = st.text_input("Patient Name", key=f"pt_name_{self.modality_key}")
                st.session_state.setdefault("patient_details", {})["name"] = p_name
                p_age = st.number_input("Age (years)", min_value=0, max_value=120, value=0, step=1, key=f"pt_age_{self.modality_key}")
                st.session_state.setdefault("patient_details", {})["age"] = p_age if p_age > 0 else None
            
            with pt_col2:
                p_sex = st.selectbox("Sex", options=["", "Male", "Female", "Other"], key=f"pt_sex_{self.modality_key}")
                st.session_state.setdefault("patient_details", {})["sex"] = p_sex
                p_id = st.text_input("Patient / Study ID", key=f"pt_id_{self.modality_key}")
                st.session_state.setdefault("patient_details", {})["patient_id"] = p_id
            
            with pt_col3:
                p_clinician = st.text_input("Requesting Clinician", key=f"pt_clinician_{self.modality_key}")
                st.session_state.setdefault("patient_details", {})["clinician"] = p_clinician
                p_facility = st.text_input("Health Facility", key=f"pt_facility_{self.modality_key}")
                st.session_state.setdefault("patient_details", {})["facility"] = p_facility
            
            p_notes = st.text_area("Clinical Indication", height=80, key=f"pt_notes_{self.modality_key}")
            st.session_state.setdefault("patient_details", {})["notes"] = p_notes
    
    def render_image_upload(self, sample_images: Dict[str, str] = None):
        """Render DICOM/image upload"""
        uploaded_file = st.file_uploader(
            f"📤 Upload {self.modality_name} image (DICOM, PNG, JPG)",
            type=["dcm", "dicom", "png", "jpg", "jpeg", "tif", "tiff"],
            key=f"uploader_{self.modality_key}",
        )
        
        if sample_images:
            st.markdown("**Or try a sample:**")
            sample_cols = st.columns(len(sample_images))
            for col, (label, path) in zip(sample_cols, sample_images.items()):
                with col:
                    if st.button(f"🖼️ {label}", key=f"sample_{self.modality_key}_{label}"):
                        st.session_state[f"sample_image_{self.modality_key}"] = path
        
        sample_key = f"sample_image_{self.modality_key}"
        has_image = uploaded_file is not None or sample_key in st.session_state
        
        if has_image:
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                # Handle DICOM
                if uploaded_file.name.lower().endswith(('.dcm', '.dicom')):
                    import pydicom
                    ds = pydicom.dcmread(io.BytesIO(file_bytes))
                    image_bgr = ds.pixel_array
                    if len(image_bgr.shape) == 2:
                        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
                    else:
                        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR)
                else:
                    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                image_name = uploaded_file.name
            else:
                sample_path = Path(st.session_state[sample_key])
                image_bgr = cv2.imread(str(sample_path))
                image_name = sample_path.name
            
            if image_bgr is not None:
                image_bgr = self._downscale_if_needed(image_bgr)
            
            return image_bgr, image_name, has_image
        
        return None, None, False
    
    def _downscale_if_needed(self, image_bgr: np.ndarray, max_dimension: int = 1024) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        longest_side = max(h, w)
        if longest_side <= max_dimension:
            return image_bgr
        scale = max_dimension / longest_side
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    def render_analyse_button(self) -> bool:
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            return st.button(
                f"🔬 Analyse {self.modality_name}",
                use_container_width=True,
                type="primary",
                key=f"analyse_{self.modality_key}",
            )
    
    def render_results(self, result: RadiologyResult):
        """Render standardized results display"""
        st.markdown("---")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        total_findings = sum(result.per_class_counts.values())
        severity_text, severity_icon = self.get_severity(total_findings)
        
        with col1:
            st.metric("Total Findings", total_findings)
        with col2:
            st.metric("Classes Detected", len([k for k, v in result.per_class_counts.items() if v > 0]))
        with col3:
            st.metric("Inference Time", f"{result.inference_time_sec*1000:.0f}ms")
        with col4:
            st.metric("Severity", f"{severity_icon} {severity_text}")
        
        # Images
        st.markdown("---")
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.markdown("#### 📷 Original Image")
            st.info("Original image would be shown here")
        with img_col2:
            st.markdown("#### 🎯 Detection Results")
            if result.annotated_image is not None:
                annotated_rgb = cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_container_width=True)
            else:
                st.info("No detections to display")
        
        # Findings table
        st.markdown("---")
        st.markdown("### 📋 Findings Summary")
        
        if result.per_class_counts:
            for class_name, count in sorted(result.per_class_counts.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    st.markdown(f"- **{class_name.replace('_', ' ').title()}**: {count} detection(s)")
        else:
            st.info("No findings detected")
        
        # Measurements
        if result.measurements:
            st.markdown("### 📐 Measurements")
            for key, value in result.measurements.items():
                st.markdown(f"- **{key.replace('_', ' ').title()}**: {value:.2f}")


import io  # for DICOM handling
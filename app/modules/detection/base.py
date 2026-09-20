# RaphaID AI - Detection Base Module

import streamlit as st
import numpy as np
import cv2
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class DetectionResult:
    """Standardized detection result across all diseases"""
    detections: List[Dict]  # List of {class_name, confidence, bbox_xyxy, bbox_xywh}
    total_cells: int
    total_targets: int  # Parasites, sickle cells, blasts, etc.
    target_percentage: float  # Parasitemia, sickle cell %, blast %, etc.
    severity: str
    annotated_image: Optional[np.ndarray]
    inference_time_sec: float
    image_path: str
    per_class_counts: Dict[str, int]


class BaseDetector(ABC):
    """Base class for all detection models"""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.class_names = []
        self.target_classes = []  # Classes that count as "targets" (parasites, sickle cells, etc.)
        self.normal_class = ""  # Normal/healthy class (RBC, normal lymphocyte, etc.)
        self.confidence_threshold = 0.25
        self.iou_threshold = 0.45
        self.uncertainty_low = 0.35
        self.uncertainty_high = 0.45
    
    @abstractmethod
    def load_model(self) -> bool:
        """Load the model. Returns True if successful."""
        pass
    
    @abstractmethod
    def predict(self, image: np.ndarray, conf: float = None, iou: float = None, annotate: bool = True) -> DetectionResult:
        """Run inference on image. Returns DetectionResult."""
        pass
    
    def get_severity(self, percentage: float) -> tuple:
        """Get severity classification based on percentage. Override in subclasses."""
        if percentage == 0:
            return "None Detected", "🟢"
        elif percentage < 1:
            return "Low", "🟡"
        elif percentage < 5:
            return "Moderate", "🟠"
        elif percentage < 10:
            return "High", "🔴"
        else:
            return "Severe", "🚨"
    
    def is_uncertain(self, confidence: float) -> bool:
        """Check if detection is in uncertainty range"""
        return self.uncertainty_low <= confidence <= self.uncertainty_high
    
    def count_uncertain(self, detections: List[Dict]) -> tuple:
        """Count uncertain vs confident target detections"""
        uncertain = 0
        confident = 0
        for d in detections:
            if d["class_name"] in self.target_classes:
                if self.is_uncertain(d["confidence"]):
                    uncertain += 1
                elif d["confidence"] > self.uncertainty_high:
                    confident += 1
        return uncertain, confident


class DetectionUI:
    """Shared UI components for detection modules"""
    
    def __init__(self, disease_key: str, disease_name: str, disease_icon: str):
        self.disease_key = disease_key
        self.disease_name = disease_name
        self.disease_icon = disease_icon
    
    def render_patient_intake(self):
        """Render patient intake form - shared across all detection modules"""
        with st.expander("🏥 Patient Information", expanded=True):
            st.caption(
                "Details entered here are stored temporarily for this session only. "
                "They are never saved to any server or database. All fields are optional."
            )
            
            pt_col1, pt_col2, pt_col3 = st.columns(3)
            
            with pt_col1:
                p_name = st.text_input(
                    "Patient Name",
                    value=st.session_state.get("patient_details", {}).get("name", ""),
                    placeholder="e.g. Femi Okoro",
                    key=f"pt_name_{self.disease_key}",
                )
                st.session_state.setdefault("patient_details", {})["name"] = p_name
                
                p_age = st.number_input(
                    "Age (years)",
                    min_value=0,
                    max_value=120,
                    value=st.session_state.get("patient_details", {}).get("age", 0) or 0,
                    step=1,
                    key=f"pt_age_{self.disease_key}",
                    help="Enter 0 if age is unknown",
                )
                st.session_state.setdefault("patient_details", {})["age"] = p_age if p_age > 0 else None
            
            with pt_col2:
                p_sex = st.selectbox(
                    "Sex (optional)",
                    options=["", "Male", "Female", "Other / Prefer not to say"],
                    index=["", "Male", "Female", "Other / Prefer not to say"].index(
                        st.session_state.get("patient_details", {}).get("sex", "")
                    ) if st.session_state.get("patient_details", {}).get("sex", "") in ["", "Male", "Female", "Other / Prefer not to say"] else 0,
                    key=f"pt_sex_{self.disease_key}",
                )
                st.session_state.setdefault("patient_details", {})["sex"] = p_sex
                
                p_id = st.text_input(
                    "Patient / Sample ID",
                    value=st.session_state.get("patient_details", {}).get("patient_id", ""),
                    placeholder="e.g. LAB-2026-00142",
                    key=f"pt_id_{self.disease_key}",
                    help="Hospital or laboratory identifier used in report filename",
                )
                st.session_state.setdefault("patient_details", {})["patient_id"] = p_id
            
            with pt_col3:
                p_clinician = st.text_input(
                    "Requesting Clinician",
                    value=st.session_state.get("patient_details", {}).get("clinician", ""),
                    placeholder="e.g. Dr. O.I Olayemi",
                    key=f"pt_clinician_{self.disease_key}",
                )
                st.session_state.setdefault("patient_details", {})["clinician"] = p_clinician
                
                p_facility = st.text_input(
                    "Health Facility",
                    value=st.session_state.get("patient_details", {}).get("facility", ""),
                    placeholder="e.g. University College Hospital Ibadan",
                    key=f"pt_facility_{self.disease_key}",
                )
                st.session_state.setdefault("patient_details", {})["facility"] = p_facility
            
            p_notes = st.text_area(
                "Clinical Notes (Optional)",
                value=st.session_state.get("patient_details", {}).get("notes", ""),
                placeholder="Brief clinical history or presenting complaints...",
                height=80,
                max_chars=250,
                key=f"pt_notes_{self.disease_key}",
            )
            st.session_state.setdefault("patient_details", {})["notes"] = p_notes
    
    def render_image_upload(self, sample_images: Dict[str, str] = None):
        """Render image upload with sample options"""
        uploaded_file = st.file_uploader(
            f"📤 Upload a {self.disease_name.lower()} microscopy image",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
            help="Supported formats: PNG, JPG, TIFF, BMP",
            key=f"uploader_{self.disease_key}",
        )
        
        # Clear sample if new file uploaded
        if uploaded_file is not None:
            sample_key = f"sample_image_{self.disease_key}"
            if sample_key in st.session_state:
                del st.session_state[sample_key]
        
        if sample_images:
            st.markdown("**Or try a sample image:**")
            sample_cols = st.columns(len(sample_images))
            for col, (label, path) in zip(sample_cols, sample_images.items()):
                with col:
                    if st.button(f"🔬 {label}", key=f"sample_{self.disease_key}_{label}"):
                        st.session_state[f"sample_image_{self.disease_key}"] = path
        
        # Check for selected sample
        sample_key = f"sample_image_{self.disease_key}"
        has_image = uploaded_file is not None or sample_key in st.session_state
        
        if has_image:
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                image_name = uploaded_file.name
            else:
                sample_path = Path(st.session_state[sample_key])
                image_bgr = cv2.imread(str(sample_path))
                image_name = sample_path.name
            
            if image_bgr is not None:
                # Downscale if needed
                image_bgr = self._downscale_if_needed(image_bgr)
            
            return image_bgr, image_name, has_image
        
        return None, None, False
    
    def _downscale_if_needed(self, image_bgr: np.ndarray, max_dimension: int = 1600) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        longest_side = max(h, w)
        if longest_side <= max_dimension:
            return image_bgr
        scale = max_dimension / longest_side
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    def render_analyse_button(self, key_suffix: str = "") -> bool:
        """Render the analyse button"""
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            return st.button(
                f"🔬 Analyse {self.disease_name}",
                use_container_width=True,
                type="primary",
                key=f"analyse_{self.disease_key}_{key_suffix}",
            )
    
    def render_results(self, result: DetectionResult, uncertain_count: int = 0):
        """Render standardized results display"""
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Cells", result.total_cells)
        with col2:
            st.metric(f"{self.disease_name} Targets", result.total_targets)
        with col3:
            st.metric("Percentage", f"{result.target_percentage:.2f}%")
        with col4:
            severity_text, severity_icon = self.get_severity(result.target_percentage)
            st.metric("Severity", f"{severity_icon} {severity_text}")
        
        # Image comparison
        st.markdown("---")
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.markdown("#### 📷 Original Image")
            # Would need original image passed separately
            st.info("Original image display")
        with img_col2:
            st.markdown("#### 🎯 Detection Results")
            if result.annotated_image is not None:
                annotated_rgb = cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_container_width=True)
            else:
                st.info("No detections to display")
        
        # Clinical summary card
        st.markdown("---")
        st.markdown(f"### 🩺 AI Screening Summary - {self.disease_name}")
        
        # Dominant stage/type
        dominant = "None"
        if result.per_class_counts:
            target_counts = {k: v for k, v in result.per_class_counts.items() if k in self.target_classes}
            if target_counts:
                dominant = max(target_counts, key=target_counts.get).replace("_", " ").title()
        
        severity_text, severity_icon = self.get_severity(result.target_percentage)
        
        # Recommendation based on severity
        if result.target_percentage == 0:
            rec_str = f"No {self.disease_name.lower()} detected. Routine confirmation recommended."
            rec_style = "background-color: #1f3a2b; color: #10B981; border-left: 5px solid #10B981;"
            rec_icon = "✅"
        elif result.target_percentage < 1:
            rec_str = f"Low {self.disease_name.lower()} burden. Microscopy review advised."
            rec_style = "background-color: #3b3a1a; color: #F59E0B; border-left: 5px solid #F59E0B;"
            rec_icon = "🟡"
        elif result.target_percentage < 5:
            rec_str = f"Moderate {self.disease_name.lower()} burden. Clinical correlation recommended."
            rec_style = "background-color: #3a2510; color: #F97316; border-left: 5px solid #F97316;"
            rec_icon = "🟠"
        else:
            rec_str = f"High {self.disease_name.lower()} burden. Immediate clinical review advised."
            rec_style = "background-color: #421818; color: #EF4444; border-left: 5px solid #EF4444;"
            rec_icon = "🚨"
        
        if uncertain_count > 0:
            rec_str = f"{uncertain_count} detection(s) require human verification."
            rec_style = "background-color: #e94560; color: #1a1a2e; border-left: 5px solid #d4af37;"
            rec_icon = "⚠️"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1F2937, #374151); border: 1px solid #374151; border-radius: 12px; padding: 1.5rem; color: white;">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1.5rem;">
                <div>
                    <div style="font-size: 0.85rem; color: #9CA3AF;">Targets Detected</div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: #EF4444;">{result.total_targets}</div>
                </div>
                <div>
                    <div style="font-size: 0.85rem; color: #9CA3AF;">Dominant Type</div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: #10B981;">{dominant}</div>
                </div>
                <div>
                    <div style="font-size: 0.85rem; color: #9CA3AF;">Percentage</div>
                    <div style="font-size: 1.8rem; font-weight: 700; color: #10B981;">{result.target_percentage:.2f}%</div>
                </div>
                <div>
                    <div style="font-size: 0.85rem; color: #9CA3AF;">Severity</div>
                    <div style="font-size: 1.2rem; font-weight: 700; margin-top: 0.5rem;">{severity_icon} {severity_text}</div>
                </div>
            </div>
            <div style="{rec_style} padding: 1rem; border-radius: 8px; font-weight: 600; display: flex; align-items: center; gap: 0.8rem;">
                <span style="font-size: 1.5rem;">{rec_icon}</span>
                <span>{rec_str}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.caption(f"⏱️ Inference completed in {result.inference_time_sec:.2f} seconds (CPU)")
    
    def render_download_buttons(self, result: DetectionResult, image_name: str, 
                                patient_details: dict, report_meta: dict, 
                                uncertain_count: int, verified_data: dict = None):
        """Render download buttons for PDF, CSV, and annotated image"""
        st.markdown("---")
        st.markdown("### 📥 Download Reports")
        
        dl_col1, dl_col2, dl_col3 = st.columns(3)
        
        with dl_col1:
            if st.button("📄 Generate PDF Report", key=f"pdf_{self.disease_key}", use_container_width=True):
                st.info("PDF generation - implement with disease-specific template")
        
        with dl_col2:
            if st.button("📊 Download CSV Data", key=f"csv_{self.disease_key}", use_container_width=True):
                st.info("CSV export - implement with detection data")
        
        with dl_col3:
            if st.button("🖼️ Download Annotated Image", key=f"img_{self.disease_key}", use_container_width=True):
                st.info("Annotated image export")
"""
RaphaID AI — Offline Multi-Disease Diagnostic Tool
Main Streamlit application integrating Detection, Radiology, and Chatbot modules.
"""

# ---------------------------------------------------------------------------
# CRITICAL FIX: PyTorch 2.1+ torch.classes compatibility
# Must be applied BEFORE any torch/ultralytics imports
# ---------------------------------------------------------------------------
import torch
import torch._classes as _tc

# Monkey patch _ClassNamespace to gracefully handle __path__/__file__ access
# This prevents: "Tried to instantiate class '__path__.__file__', but it does not exist!"
_original_namespace_getattr = _tc._ClassNamespace.__getattr__

def _safe_namespace_getattr(self, attr):
    try:
        return _original_namespace_getattr(self, attr)
    except RuntimeError as e:
        error_str = str(e)
        # Handle __path__.__file__ and similar access patterns that fail in PyTorch 2.1+
        if '__path__' in error_str or '__file__' in error_str:
            # Return empty list for __path__ access (common pattern for namespace packages)
            # Return empty string for __file__ access
            return [] if '__path__' in self.name else ''
        raise

_tc._ClassNamespace.__getattr__ = _safe_namespace_getattr
# ---------------------------------------------------------------------------

import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Add project root to path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Imports from new module structure
# ---------------------------------------------------------------------------
from app.components.theme import apply_theme, get_theme_colors
from app.components.navigation import render_navigation
from app.components.footer import render_footer, update_session_metrics

from app.modules.detection import (
    MalariaDetector,
    SickleCellDetector,
    ALLDetector,
    IronDeficiencyDetector,
    load_malaria_model,
    load_sickle_cell_model,
    load_all_model,
    load_iron_deficiency_model,
)
from app.modules.radiology import (
    MRIDetector,
    CTScanDetector,
    XRayDetector,
    load_mri_model,
    load_ct_model,
    load_xray_model,
)
from app.modules.chatbot import render_chatbot_ui

from src.utils.quality_check import assess_image_quality
from src.utils.report_generator import (
    generate_mri_report,
    generate_ct_report,
    generate_xray_report,
    generate_csv_report_radiology,
)
from src.utils.session_manager import get_session_manager


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DETECTION_SUBMODULES = {
    "malaria": "🦠 Malaria",
    "sickle_cell": "🩸 Sickle Cell",
    "all": "🧬 ALL Leukemia",
    "iron_deficiency": "🩸 Iron Deficiency",
}

RADIOLOGY_SUBMODULES = {
    "mri": "🧠 MRI Brain",
    "ct": "🫁 CT Chest",
    "xray": "🩻 X-ray Chest",
}

MODEL_PATHS = {
    "detection": {
        "malaria": "models/detection/malaria_yolov8n.pt",
        "sickle_cell": "models/detection/sickle_cell_yolov8n.pt",
        "all": "models/detection/all_yolov8n.pt",
        "iron_deficiency": "models/detection/iron_deficiency_yolov8n.pt",
    },
    "radiology": {
        "mri": "models/radiology/mri_yolov8n.pt",
        "ct": "models/radiology/ct_yolov8n.pt",
        "xray": "models/radiology/xray_yolov8n.pt",
    },
}

# Default model configs per submodule
DETECTION_CONFIGS = {
    "malaria": {
        "classes": ["ring", "trophozoite", "schizont", "gametocyte", "red_blood_cell"],
        "parasite_classes": {"ring", "trophozoite", "schizont", "gametocyte"},
        "uncertainty_low": 0.35,
        "uncertainty_high": 0.45,
    },
    "sickle_cell": {
        "classes": ["normal_rbc", "sickle_cell", "target_cell", "spherocyte"],
        "abnormal_classes": {"sickle_cell", "target_cell", "spherocyte"},
        "uncertainty_low": 0.35,
        "uncertainty_high": 0.45,
    },
    "all": {
        "classes": ["lymphoblast", "normal_lymphocyte", "smudge_cell"],
        "blast_classes": {"lymphoblast"},
        "uncertainty_low": 0.35,
        "uncertainty_high": 0.45,
    },
    "iron_deficiency": {
        "classes": ["microcyte", "hypochromic", "normal_rbc", "pencil_cell"],
        "abnormal_classes": {"microcyte", "hypochromic", "pencil_cell"},
        "uncertainty_low": 0.35,
        "uncertainty_high": 0.45,
    },
}


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def _downscale_if_needed(image_bgr: np.ndarray, max_dimension: int = 1600) -> np.ndarray:
    """Downscale image if too large to prevent OOM."""
    h, w = image_bgr.shape[:2]
    longest_side = max(h, w)
    if longest_side <= max_dimension:
        return image_bgr
    scale = max_dimension / longest_side
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _is_uncertain(confidence: float, low: float = 0.35, high: float = 0.45) -> bool:
    return low <= confidence <= high


def _count_tiers(detections, parasite_classes, low=0.35, high=0.45):
    uncertain = confident = 0
    for d in detections:
        if d.class_name not in parasite_classes:
            continue
        if low <= d.confidence <= high:
            uncertain += 1
        elif d.confidence > high:
            confident += 1
    return uncertain, confident


def _draw_filtered(image_bgr, detections, config, class_colors, submodule: str):
    """Draw detections with uncertainty highlighting, optionally filtering classes."""
    img = image_bgr.copy()
    show_all = st.session_state.get(f"show_all_{submodule}", True)

    for det in detections:
        # Skip normal/healthy classes if filter is on
        if not show_all:
            skip_classes = {"red_blood_cell", "normal_rbc", "normal_lymphocyte", "normal"}
            if det.class_name in skip_classes:
                continue

        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
        is_uncertain_det = _is_uncertain(det.confidence, config["uncertainty_low"], config["uncertainty_high"])

        if is_uncertain_det:
            color = (0, 255, 255)  # Yellow
            thickness = 3
            label = f"INCONCLUSIVE {det.confidence:.2f}"
            text_color = (0, 0, 0)
        else:
            color = class_colors.get(det.class_name, (255, 255, 255))
            is_target = (
                det.class_name in config.get("parasite_classes", set()) or
                det.class_name in config.get("abnormal_classes", set()) or
                det.class_name in config.get("blast_classes", set())
            )
            thickness = 2 if is_target else 1
            label = f"{det.class_name} {det.confidence:.2f}"
            text_color = (255, 255, 255)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)
    return img


def _render_detection_gallery(image_bgr, detections, config, title="Detection Close-ups"):
    """Render zoomable gallery of detection crops."""
    target_classes = config.get("parasite_classes") or config.get("abnormal_classes") or config.get("blast_classes", set())
    target_dets = [d for d in detections if d.class_name in target_classes]

    if not target_dets:
        st.info(f"No target detections to display for {title.lower()}.")
        return

    st.markdown(f"### 🔬 {title}")
    st.caption("Zoomed crops for visual verification")

    img_h, img_w = image_bgr.shape[:2]
    gallery_items = []

    for det in target_dets:
        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue

        pad_x, pad_y = max(1, int(bw * 0.1)), max(1, int(bh * 0.1))
        cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        cx2, cy2 = min(img_w, x2 + pad_x), min(img_h, y2 + pad_y)

        crop = image_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        crop = cv2.resize(crop, (150, 150), interpolation=cv2.INTER_CUBIC)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        uncertain = _is_uncertain(det.confidence, config["uncertainty_low"], config["uncertainty_high"])
        gallery_items.append({
            "image": crop,
            "class_name": det.class_name,
            "confidence": det.confidence,
            "is_uncertain": uncertain,
        })

    if not gallery_items:
        return

    # Sort: uncertain first
    gallery_items.sort(key=lambda x: (not x["is_uncertain"], -x["confidence"]))

    # Render in grid
    max_initial = 12
    initial = gallery_items[:max_initial]
    remaining = gallery_items[max_initial:]

    def _render_grid(items):
        cols = st.columns(4)
        for i, item in enumerate(items):
            with cols[i % 4]:
                st.image(item["image"], use_container_width=True)
                name = item["class_name"].replace("_", " ").title()
                st.markdown(f"**{name}**")
                st.caption(f"Conf: {item['confidence']:.1%}")
                st.caption("⚠️ Needs review" if item["is_uncertain"] else "✓ High confidence")

    _render_grid(initial)

    if remaining:
        expanded = st.session_state.get("gallery_expanded", False)
        if expanded:
            _render_grid(remaining)
        _, c, _ = st.columns([2, 1, 2])
        with c:
            if expanded:
                if st.button("🔼 Show less", key="gallery_collapse", use_container_width=True):
                    st.session_state.gallery_expanded = False
                    st.rerun()
            else:
                if st.button(f"🔽 Show all {len(gallery_items)}", key="gallery_expand", use_container_width=True):
                    st.session_state.gallery_expanded = True
                    st.rerun()


def _render_patient_intake():
    """Render patient information form in sidebar/expander."""
    if "patient_details" not in st.session_state:
        st.session_state.patient_details = {
            "name": "", "age": None, "sex": "", "patient_id": "",
            "clinician": "", "facility": "", "notes": "",
        }

    with st.expander("🏥 Patient Information", expanded=True):
        st.caption("Session-only storage. Not saved to server. All fields optional.")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.session_state.patient_details["name"] = st.text_input(
                "Patient Name", value=st.session_state.patient_details["name"],
                placeholder="e.g. Femi Okoro", key="pt_name"
            )
            age_val = st.number_input("Age", 0, 120,
                value=st.session_state.patient_details["age"] or 0,
                key="pt_age", help="0 = unknown")
            st.session_state.patient_details["age"] = age_val if age_val > 0 else None

        with c2:
            sex_opts = ["", "Male", "Female", "Other / Prefer not to say"]
            curr_sex = st.session_state.patient_details["sex"]
            idx = sex_opts.index(curr_sex) if curr_sex in sex_opts else 0
            st.session_state.patient_details["sex"] = st.selectbox(
                "Sex", sex_opts, index=idx, key="pt_sex"
            )
            st.session_state.patient_details["patient_id"] = st.text_input(
                "Patient / Sample ID", value=st.session_state.patient_details["patient_id"],
                placeholder="e.g. LAB-2026-00142", key="pt_id"
            )

        with c3:
            st.session_state.patient_details["clinician"] = st.text_input(
                "Requesting Clinician", value=st.session_state.patient_details["clinician"],
                placeholder="e.g. Dr. O.I Olayemi", key="pt_clinician"
            )
            st.session_state.patient_details["facility"] = st.text_input(
                "Health Facility", value=st.session_state.patient_details["facility"],
                placeholder="e.g. UCH Ibadan", key="pt_facility"
            )

        st.session_state.patient_details["notes"] = st.text_area(
            "Clinical Notes", value=st.session_state.patient_details["notes"],
            placeholder="Brief history...", max_chars=250, height=80, key="pt_notes"
        )

        if st.button("🗑 Clear", key="clear_patient"):
            st.session_state.patient_details = {k: ("" if k != "age" else None) for k in st.session_state.patient_details}
            st.rerun()


def _render_detection_module(submodule: str):
    """Render the Detection module UI for a specific disease."""
    config = DETECTION_CONFIGS[submodule]
    model_path = MODEL_PATHS["detection"][submodule]

    # Module-specific config
    if submodule == "malaria":
        detector_loader = load_malaria_model
        class_colors = {
            "red_blood_cell": (200, 200, 200), "ring": (0, 0, 255),
            "trophozoite": (0, 165, 255), "schizont": (0, 255, 255), "gametocyte": (255, 0, 255),
        }
        metric_keys = ["total_rbc", "total_parasites", "parasitemia_pct"]
    elif submodule == "sickle_cell":
        detector_loader = load_sickle_cell_model
        class_colors = {
            "normal_rbc": (200, 200, 200), "sickle_cell": (0, 0, 255),
            "target_cell": (0, 165, 255), "spherocyte": (255, 0, 255),
        }
        metric_keys = ["total_normal", "total_abnormal", "sickle_percentage"]
    elif submodule == "all":
        detector_loader = load_all_model
        class_colors = {
            "normal_lymphocyte": (200, 200, 200), "lymphoblast": (0, 0, 255),
            "smudge_cell": (0, 165, 255),
        }
        metric_keys = ["total_cells", "blast_count", "blast_percentage"]
    else:  # iron_deficiency
        detector_loader = load_iron_deficiency_model
        class_colors = {
            "normal_rbc": (200, 200, 200), "microcyte": (0, 0, 255),
            "hypochromic": (0, 165, 255), "pencil_cell": (255, 0, 255),
        }
        metric_keys = ["total_cells", "normal_count", "abnormal_percentage"]

    # Sidebar settings
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ⚙️ Detection Settings")
        conf = st.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05, key=f"conf_{submodule}")
        iou = st.slider("NMS IoU", 0.1, 0.9, 0.45, 0.05, key=f"iou_{submodule}")
        st.checkbox(
            "Show all classes (incl. healthy)", value=True, key=f"show_all_{submodule}"
        )

    # Load model
    detector = detector_loader(model_path)
    if detector is None:
        st.error(f"❌ Model not found at `{model_path}`. Train or download weights first.")
        st.info("Expected: `models/detection/{submodule}_yolov8n.pt`")
        return

    # Patient intake
    _render_patient_intake()

    # Image upload
    st.markdown("---")
    uploaded = st.file_uploader(
        "📤 Upload microscopy image",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        key=f"uploader_{submodule}"
    )

    # Sample images
    st.markdown("**Or try a sample:**")
    s1, s2, s3 = st.columns(3)
    samples = {"Sample 1": "app/samples/infected_sample.jpg",
               "Sample 2": "app/samples/mixed_sample.jpg",
               "Sample 3": "app/samples/healthy_sample.jpg"}
    for i, (label, path) in enumerate(samples.items()):
        with [s1, s2, s3][i]:
            if st.button(f"🔬 {label}", key=f"sample_{submodule}_{i}"):
                st.session_state[f"sample_{submodule}"] = path

    has_image = uploaded or f"sample_{submodule}" in st.session_state
    if has_image:
        if uploaded:
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_name = uploaded.name
        else:
            img = cv2.imread(st.session_state[f"sample_{submodule}"])
            img_name = Path(st.session_state[f"sample_{submodule}"]).name

        if img is None:
            st.error("Could not decode image.")
            return

        img = _downscale_if_needed(img)

        # Quality check
        quality = assess_image_quality(img, "microscopy")
        if not quality["passed"]:
            for issue in quality["issues"]:
                if issue["severity"] == "error":
                    st.error(f"🚫 {issue['message']} ({issue['detail']})")
            st.stop()
        elif quality["has_warnings"]:
            for issue in quality["issues"]:
                st.warning(f"⚠️ {issue['message']} ({issue['detail']})")

        # Run inference
        if st.button("🔬 Analyse Slide", type="primary", key=f"analyse_{submodule}", use_container_width=True):
            with st.spinner("Running detection..."):
                result = detector.predict(img, conf=conf, iou=iou, annotate=True)
            st.session_state[f"result_{submodule}"] = result
            st.session_state[f"img_{submodule}"] = img
            st.session_state[f"img_name_{submodule}"] = img_name
            st.rerun()

    # Display results
    result_key = f"result_{submodule}"
    if result_key in st.session_state:
        result = st.session_state[result_key]
        img = st.session_state[f"img_{submodule}"]
        img_name = st.session_state[f"img_name_{submodule}"]

        # Metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Cells", getattr(result, metric_keys[0], len(result.detections)))
        with m2:
            st.metric("Abnormal/Parasites", getattr(result, metric_keys[1], 0))
        with m3:
            st.metric("Percentage", f"{getattr(result, metric_keys[2], 0):.2f}%")

        # Images
        c1, c2 = st.columns(2)
        with c1:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
        with c2:
            display_img = _draw_filtered(img, result.detections, config, class_colors, submodule)
            st.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), caption="Detections", use_container_width=True)

        # Uncertainty check
        parasite_classes = config.get("parasite_classes") or config.get("abnormal_classes") or config.get("blast_classes", set())
        unc, conf_cnt = _count_tiers(result.detections, parasite_classes, config["uncertainty_low"], config["uncertainty_high"])
        if unc > 0:
            st.warning(f"⚠️ {unc} detection(s) need human verification (confidence 35-45%)")

        # Gallery
        _render_detection_gallery(img, result.detections, config)

        # Downloads
        st.markdown("---")
        st.markdown("### 📥 Download Reports")
        d1, d2, d3 = st.columns(3)
        with d1:
            # PDF would need disease-specific generator
            st.info("PDF report generation per-disease coming soon")
        with d2:
            # CSV
            import csv, io
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["class", "confidence", "x1", "y1", "x2", "y2"])
            for d in result.detections:
                writer.writerow([d.class_name, f"{d.confidence:.4f}", *[f"{v:.1f}" for v in d.bbox_xyxy]])
            st.download_button("📊 CSV", output.getvalue(),
                             f"detections_{Path(img_name).stem}.csv", "text/csv")
        with d3:
            if result.annotated_image is not None:
                pil_img = Image.fromarray(cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB))
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                st.download_button("🖼️ Image", buf.getvalue(),
                                 f"annotated_{Path(img_name).stem}.png", "image/png")

        # Update session metrics
        update_session_metrics(submodule, {"total_detections": len(result.detections), "positive": getattr(result, metric_keys[1], 0) > 0})


def _render_radiology_module(submodule: str):
    """Render the Radiology module UI for a specific modality."""
    model_path = MODEL_PATHS["radiology"][submodule]

    if submodule == "mri":
        detector_loader = load_mri_model
        report_gen = generate_mri_report
        modality_name = "MRI Brain"
    elif submodule == "ct":
        detector_loader = load_ct_model
        report_gen = generate_ct_report
        modality_name = "CT Chest"
    else:  # xray
        detector_loader = load_xray_model
        report_gen = generate_xray_report
        modality_name = "Chest X-ray"

    with st.sidebar:
        st.markdown("---")
        st.markdown("### ⚙️ Radiology Settings")
        conf = st.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05, key=f"conf_{submodule}")
        iou = st.slider("NMS IoU", 0.1, 0.9, 0.45, 0.05, key=f"iou_{submodule}")

    detector = detector_loader(model_path)
    if detector is None:
        st.error(f"❌ Model not found at `{model_path}`")
        st.info(f"Expected: `models/radiology/{submodule}_yolov8n.pt`")
        return

    _render_patient_intake()

    st.markdown("---")
    uploaded = st.file_uploader(
        f"📤 Upload {modality_name} image (PNG, JPG, DICOM)",
        type=["png", "jpg", "jpeg", "tif", "tiff", "dcm"],
        key=f"uploader_{submodule}"
    )

    has_image = uploaded
    if has_image:
        if uploaded.name.lower().endswith(".dcm"):
            import pydicom
            ds = pydicom.dcmread(uploaded)
            pixel_array = ds.pixel_array
            if pixel_array.dtype != np.uint8:
                pixel_array = cv2.normalize(pixel_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            img = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2BGR) if len(pixel_array.shape) == 2 else pixel_array
        else:
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_name = uploaded.name

        if img is None:
            st.error("Could not decode image.")
            return

        img = _downscale_if_needed(img)

        # Quality check
        quality = assess_image_quality(img, submodule)
        if not quality["passed"]:
            for issue in quality["issues"]:
                if issue["severity"] == "error":
                    st.error(f"🚫 {issue['message']}")
            st.stop()
        elif quality["has_warnings"]:
            for issue in quality["issues"]:
                st.warning(f"⚠️ {issue['message']}")

        if st.button(f"🔬 Analyse {modality_name}", type="primary", key=f"analyse_{submodule}", use_container_width=True):
            with st.spinner(f"Analyzing {modality_name}..."):
                if submodule == "mri":
                    result = detector.predict(img, conf=conf, iou=iou, annotate=True)
                elif submodule == "ct":
                    result = detector.predict(img, conf=conf, iou=iou, annotate=True)
                else:
                    result = detector.predict(img, conf=conf, iou=iou, annotate=True)
            st.session_state[f"result_{submodule}"] = result
            st.session_state[f"img_{submodule}"] = img
            st.session_state[f"img_name_{submodule}"] = img_name
            st.rerun()

    result_key = f"result_{submodule}"
    if result_key in st.session_state:
        result = st.session_state[result_key]
        img = st.session_state[f"img_{submodule}"]
        img_name = st.session_state[f"img_name_{submodule}"]

        # Results summary
        st.markdown(f"### 📋 {modality_name} Analysis Results")
        summary = result.summary()
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Detections", summary["total_detections"])
        with m2:
            if submodule == "mri":
                st.metric("Tumor Volume", f"{summary.get('tumor_volume_mm3', 0):.1f} mm³")
            elif submodule == "ct":
                st.metric("Nodule Count", summary.get("nodule_count", 0))
            else:
                st.metric("Pneumonia Foci", summary.get("pneumonia_count", 0))
        with m3:
            st.metric("Inference Time", f"{summary['inference_time_sec']:.2f}s")

        # Images
        c1, c2 = st.columns(2)
        with c1:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
        with c2:
            if result.annotated_image is not None:
                st.image(cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB),
                        caption="Detections", use_container_width=True)

        # Detection table
        st.markdown("### 📊 Detection Summary")
        counts = result._per_class_counts() if hasattr(result, '_per_class_counts') else {}
        if counts:
            df = pd.DataFrame([{"Class": k.replace("_", " ").title(), "Count": v} for k, v in counts.items()])
            st.table(df)

        # Downloads
        st.markdown("---")
        st.markdown("### 📥 Download Reports")
        d1, d2, d3 = st.columns(3)
        with d1:
            try:
                pdf_bytes = report_gen(
                    result,
                    patient_details=st.session_state.get("patient_details"),
                    report_meta=st.session_state.get("report_meta"),
                )
                st.download_button("📄 PDF Report", pdf_bytes,
                                 f"{submodule}_report_{Path(img_name).stem}.pdf", "application/pdf")
            except Exception as e:
                st.error(f"PDF error: {e}")
        with d2:
            csv_data = generate_csv_report_radiology(result, submodule)
            st.download_button("📊 CSV", csv_data,
                             f"{submodule}_detections_{Path(img_name).stem}.csv", "text/csv")
        with d3:
            if result.annotated_image is not None:
                pil_img = Image.fromarray(cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB))
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                st.download_button("🖼️ Image", buf.getvalue(),
                                 f"annotated_{Path(img_name).stem}.png", "image/png")

        update_session_metrics(submodule, {"total_detections": summary["total_detections"], "positive": summary["total_detections"] > 0})


def _render_dashboard():
    """Render the main dashboard."""
    colors = get_theme_colors()

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0d0d1a, #1a1a2e, #0f3460);
                border-radius: 14px; padding: 1.2rem 1.8rem; margin-bottom: 1rem;
                border: 1px solid rgba(255,255,255,0.12); text-align: center;">
        <h2 style="color: #FFFFFF; margin-bottom: 0.2rem; font-size: 1.8rem;">
           🩺 RaphaID AI
        </h2>
        <p style="color: #D8DEE9; margin: 0 0 0.3rem 0; font-size: 0.95rem;">
            Offline Multi-Disease Diagnostic Tool
        </p>
        <p style="color: #9AA4B2; margin: 0 0 1rem 0; font-size: 0.78rem;">
            YOLOv8n · WHO Workflow · Human Verification
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Quick actions
    st.markdown("### 🚀 Quick Actions")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🔬 New Malaria Diagnosis", use_container_width=True, type="primary"):
            st.session_state["current_module"] = "detection"
            st.session_state["detection_submodule"] = "malaria"
            st.rerun()
    with c2:
        if st.button("🏥 New Radiology Scan", use_container_width=True):
            st.session_state["current_module"] = "radiology"
            st.session_state["radiology_submodule"] = "mri"
            st.rerun()
    with c3:
        if st.button("🤖 Medical Assistant", use_container_width=True):
            st.session_state["current_module"] = "chatbot"
            st.rerun()

    # Quick facts
    st.markdown("### 📋 Quick Facts")
    facts = [
        ("🧠", "Models", "YOLOv8n (quantized)"),
        ("🦠", "Detection", "4 Diseases"),
        ("🏥", "Radiology", "3 Modalities"),
        ("🤖", "Chatbot", "RAG + LangGraph"),
        ("⚡", "Inference", "< 500ms CPU"),
        ("💾", "RAM Target", "< 6GB"),
        ("🔒", "Offline", "Fully Air-gapped"),
        ("📄", "Reports", "PDF + CSV"),
    ]
    for row_start in range(0, len(facts), 4):
        cols = st.columns(4)
        for col, (icon, label, value) in zip(cols, facts[row_start:row_start+4]):
            with col:
                st.markdown(f"""
                <div style="background: #16213e; border: 1px solid rgba(255,255,255,0.12);
                            border-radius: 10px; padding: 0.8rem; text-align: center;">
                    <div style="font-size: 1.3rem;">{icon}</div>
                    <div style="font-size: 0.68rem; color: #9AA4B2; text-transform: uppercase; margin-top: 0.35rem;">{label}</div>
                    <div style="font-size: 1rem; font-weight: 700; color: #FFFFFF; margin-top: 0.25rem;">{value}</div>
                </div>
                """, unsafe_allow_html=True)

    # Clinical workflow
    st.markdown("### 🔄 Clinical Workflow")
    st.markdown("""
    <div style="background: #16213e; border: 1px solid rgba(255,255,255,0.12);
                border-radius: 12px; padding: 1rem; display: flex; flex-wrap: wrap; gap: 0.5rem; justify-content: space-between;">
        <div style="flex: 1; min-width: 120px; text-align: center;">
            <div style="font-size: 1.5rem;">🧫</div>
            <div style="font-weight: 600;">Upload</div>
            <div style="font-size: 0.75rem; color: #9AA4B2;">Sample/Image</div>
        </div>
        <div style="font-size: 1.2rem; color: #64ffda; align-self: center;">→</div>
        <div style="flex: 1; min-width: 120px; text-align: center;">
            <div style="font-size: 1.5rem;">🔍</div>
            <div style="font-weight: 600;">Quality</div>
            <div style="font-size: 0.75rem; color: #9AA4B2;">Auto-Check</div>
        </div>
        <div style="font-size: 1.2rem; color: #64ffda; align-self: center;">→</div>
        <div style="flex: 1; min-width: 120px; text-align: center;">
            <div style="font-size: 1.5rem;">🧠</div>
            <div style="font-weight: 600;">Detect</div>
            <div style="font-size: 0.75rem; color: #9AA4B2;">AI Inference</div>
        </div>
        <div style="font-size: 1.2rem; color: #64ffda; align-self: center;">→</div>
        <div style="flex: 1; min-width: 120px; text-align: center;">
            <div style="font-size: 1.5rem;">👨‍⚕️</div>
            <div style="font-weight: 600;">Verify</div>
            <div style="font-size: 0.75rem; color: #9AA4B2;">Clinician Review</div>
        </div>
        <div style="font-size: 1.2rem; color: #64ffda; align-self: center;">→</div>
        <div style="flex: 1; min-width: 120px; text-align: center;">
            <div style="font-size: 1.5rem;">📄</div>
            <div style="font-weight: 600;">Report</div>
            <div style="font-size: 0.75rem; color: #9AA4B2;">PDF/CSV Export</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Disclaimer
    st.warning(
        "Disclaimer: For research use and decision support only. Not intended as a "
        "standalone diagnostic tool. All findings require confirmation "
        "by a qualified specialist."
    )

    # Footer
    render_footer()


def main():
    # Page config
    st.set_page_config(
        page_title="RaphaID AI",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Apply theme
    apply_theme()

    # Session state initialization
    defaults = {
        "current_module": "dashboard",
        "detection_submodule": "malaria",
        "radiology_submodule": "mri",
        "report_meta": {
            "report_number": f"RPT-{datetime.now().strftime('%Y%m%d')}-{np.random.randint(10000, 99999)}",
            "study_id": f"STD-{np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 2).tolist()[0]}{np.random.randint(100, 999)}",
            "date": datetime.now().strftime("%d %B %Y"),
            "time": datetime.now().strftime("%H:%M"),
        },
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Initialize session manager
    session_mgr = get_session_manager()
    if "session_data" not in st.session_state:
        st.session_state.session_data = {
            "session_id": datetime.now().strftime("%Y%m%d%H%M%S"),
            "start_time": datetime.now().strftime("%H:%M:%S"),
            "current_module": "dashboard",
            "analysis_count": 0,
            "metrics": {"total_detections": 0, "positive_cases": 0, "verifications": 0, "reports_generated": 0},
            "analytics": {"malaria": 0, "sickle_cell": 0, "all": 0, "iron_deficiency": 0, "mri": 0, "ct": 0, "xray": 0},
        }

    # Navigation callback
    def on_module_change(module):
        st.session_state["current_module"] = module
        st.session_state.session_data["current_module"] = module

    # Render sidebar navigation
    render_navigation(st.session_state["current_module"], on_module_change)

    # Route to module
    module = st.session_state["current_module"]

    if module == "dashboard":
        _render_dashboard()

    elif module == "detection":
        submodule = st.session_state.get("detection_submodule", "malaria")
        st.markdown(f"### {DETECTION_SUBMODULES[submodule]} Detection")
        _render_detection_module(submodule)

    elif module == "radiology":
        submodule = st.session_state.get("radiology_submodule", "mri")
        st.markdown(f"### {RADIOLOGY_SUBMODULES[submodule]} Analysis")
        _render_radiology_module(submodule)

    elif module == "chatbot":
        render_chatbot_ui()


if __name__ == "__main__":
    main()
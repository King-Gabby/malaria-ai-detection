"""
streamlit_app.py — Interactive Malaria Detection Demo

Upload a blood smear microscopy image → AI-powered parasite detection →
annotated image + parasitemia estimate + downloadable PDF/CSV report.

Designed for the competition demo day: visually polished, shows clinical
relevance, and handles edge cases gracefully.

Usage:
    streamlit run app/streamlit_app.py -- --weights models/best.pt

REQUIRES: A trained model weights file (models/best.pt).
"""

import csv
import io
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Add project root to path so we can import our modules
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import MalariaDetector, PredictionResult, PARASITE_CLASSES


# ---------------------------------------------------------------------------
# PDF Report Generation
# ---------------------------------------------------------------------------
def generate_pdf_report(
    result: PredictionResult,
    annotated_img: np.ndarray,
) -> bytes:
    """Generate a downloadable PDF report with detection results.

    The PDF includes:
      • Header with timestamp and file info
      • Parasitemia percentage (large, bold)
      • Per-class detection counts table
      • Annotated image snapshot

    Returns:
        PDF file as bytes for Streamlit download button.
    """
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # --- Header ---
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 12, "Malaria Detection Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.cell(0, 8, f"Image: {Path(result.image_path).name}", ln=True, align="C")
    pdf.ln(8)

    # --- Parasitemia ---
    pdf.set_font("Helvetica", "B", 16)
    severity = classify_severity(result.parasitemia_pct)
    pdf.cell(0, 10, f"Estimated Parasitemia: {result.parasitemia_pct:.2f}%", ln=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Severity: {severity}", ln=True)
    pdf.ln(6)

    # --- Detection summary table ---
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Detection Summary", ln=True)
    pdf.set_font("Helvetica", "", 10)

    counts = result._per_class_counts()
    pdf.cell(90, 7, "Class", border=1, align="C")
    pdf.cell(40, 7, "Count", border=1, align="C", ln=True)

    for cls_name, count in counts.items():
        pdf.cell(90, 7, cls_name, border=1)
        pdf.cell(40, 7, str(count), border=1, align="C", ln=True)

    pdf.cell(90, 7, "TOTAL CELLS", border=1, fill=True)
    pdf.cell(40, 7, str(result.total_rbc + result.total_parasites), border=1, align="C", ln=True)
    pdf.ln(6)

    # --- Annotated image ---
    if annotated_img is not None:
        # Save image to temp file for FPDF
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            Image.fromarray(img_rgb).save(tmp.name)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Annotated Image", ln=True)
            pdf.image(tmp.name, w=180)

    # --- Disclaimer ---
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 5,
        "DISCLAIMER: This is an AI-assisted screening tool for research and "
        "educational purposes only. It is NOT a certified medical diagnostic device. "
        "All findings must be confirmed by a qualified microscopist."
    )

    return pdf.output()


def generate_csv_report(result: PredictionResult) -> str:
    """Generate a CSV string with per-detection rows."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "class", "confidence", "x1", "y1", "x2", "y2",
        "x_center", "y_center", "width", "height",
    ])
    for d in result.detections:
        writer.writerow([
            d.class_name,
            f"{d.confidence:.4f}",
            *[f"{v:.1f}" for v in d.bbox_xyxy],
            *[f"{v:.1f}" for v in d.bbox_xywh],
        ])
    return output.getvalue()


# ---------------------------------------------------------------------------
# Severity Classification (WHO guidelines, simplified)
# ---------------------------------------------------------------------------
def classify_severity(parasitemia_pct: float) -> str:
    """Classify malaria severity based on parasitemia percentage.

    Based on WHO treatment guidelines:
      • <1%: Low parasitemia (uncomplicated malaria)
      • 1-5%: Moderate
      • >5%: High / severe (consider IV artesunate)
    """
    if parasitemia_pct == 0:
        return "🟢 No parasites detected"
    elif parasitemia_pct < 1:
        return "🟡 Low parasitemia (< 1%)"
    elif parasitemia_pct < 5:
        return "🟠 Moderate parasitemia (1–5%)"
    else:
        return "🔴 High parasitemia (> 5%) — SEVERE"


# ---------------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------------
def main():
    # --- Page config ---
    st.set_page_config(
        page_title="🦟 Malaria Detection System",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # --- Custom CSS for a polished look ---
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            padding: 1.5rem 0;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            border-radius: 12px;
            margin-bottom: 2rem;
            color: white;
        }
        .main-header h1 {
            color: #e94560;
            font-size: 2.2rem;
            margin-bottom: 0.3rem;
        }
        .main-header p {
            color: #a8b2d1;
            font-size: 1rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            border: 1px solid #233554;
            border-radius: 10px;
            padding: 1.2rem;
            text-align: center;
            color: white;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #64ffda;
        }
        .metric-label {
            font-size: 0.85rem;
            color: #8892b0;
            margin-top: 0.3rem;
        }
        .severity-badge {
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-weight: 600;
            font-size: 1.1rem;
            text-align: center;
        }
        .stButton > button {
            background: linear-gradient(135deg, #e94560, #c23152);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(233, 69, 96, 0.4);
        }
        .detection-table {
            width: 100%;
            border-collapse: collapse;
        }
        .detection-table th {
            background: #16213e;
            color: #64ffda;
            padding: 0.6rem;
        }
        .detection-table td {
            padding: 0.5rem;
            border-bottom: 1px solid #233554;
        }
    </style>
    """, unsafe_allow_html=True)

    # --- Header ---
    st.markdown("""
    <div class="main-header">
        <h1>🔬 AI-Powered Malaria Detection</h1>
        <p>YOLOv8 Object Detection • P. vivax Parasite Identification • Blood Smear Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Sidebar: Settings ---
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        weights_path = st.text_input(
            "Model Weights Path",
            value="models/best.pt",
            help="Path to trained YOLOv8 .pt file",
        )

        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.25,
            step=0.05,
            help="Minimum detection confidence. Lower = more detections (more false positives).",
        )

        iou_threshold = st.slider(
            "NMS IoU Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.45,
            step=0.05,
            help="Non-Maximum Suppression threshold. Lower = fewer overlapping boxes.",
        )

        show_rbc = st.checkbox(
            "Show Red Blood Cells",
            value=False,
            help="Toggle visibility of healthy RBC detections (reduces visual clutter).",
        )

        # CHANGED: Added Model Performance metrics card to sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Model Performance")
        metrics_col1, metrics_col2 = st.sidebar.columns(2)
        with metrics_col1:
            st.metric("mAP@50", "—", help="Updated after training")
            st.metric("Precision", "—", help="Updated after training")
        with metrics_col2:
            st.metric("Recall", "—", help="Updated after training")
            st.metric("F1 Score", "—", help="Updated after training")
        st.sidebar.caption("Metrics populated after model training.")

        st.markdown("---")
        st.markdown("### 📊 About")
        st.markdown(
            "This tool detects malaria parasites (*P. vivax*) in blood smear "
            "microscopy images using a YOLOv8 object detection model trained "
            "on the BBBC041 dataset."
        )
        st.markdown(
            "**Stages detected:** Ring, Trophozoite, Schizont, Gametocyte"
        )
        st.markdown("---")
        st.caption(
            "⚠️ For research and educational use only. "
            "Not a certified medical diagnostic device."
        )

    # --- Main content ---
    uploaded_file = st.file_uploader(
        "📤 Upload a blood smear microscopy image",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        help="Supported formats: PNG, JPG, TIFF, BMP",
    )

    # CHANGED: Clear sample image selection if a new file is uploaded
    if uploaded_file is not None:
        if "sample_image" in st.session_state:
            del st.session_state["sample_image"]

    # CHANGED: Added sample images button section
    st.markdown("**Or try a sample image:**")
    sample_col1, sample_col2, sample_col3 = st.columns(3)
    with sample_col1:
        if st.button("🔬 Sample 1 — Infected"):
            st.session_state["sample_image"] = "app/samples/infected_sample.jpg"
    with sample_col2:
        if st.button("🔬 Sample 2 — Mixed"):
            st.session_state["sample_image"] = "app/samples/mixed_sample.jpg"
    with sample_col3:
        if st.button("🔬 Sample 3 — Healthy"):
            st.session_state["sample_image"] = "app/samples/healthy_sample.jpg"

    # CHANGED: Added How It Works section
    st.markdown("---")
    st.markdown("### ⚙️ How It Works")
    how_col1, how_col2, how_col3 = st.columns(3)
    with how_col1:
        st.markdown("#### 1️⃣ Upload")
        st.markdown("Upload a PNG, JPG, TIFF, or BMP blood smear microscopy image.")
    with how_col2:
        st.markdown("#### 2️⃣ Detect")
        st.markdown("YOLOv8 identifies and localises parasite stages with bounding boxes.")
    with how_col3:
        st.markdown("#### 3️⃣ Report")
        st.markdown("Get parasite count, parasitemia %, WHO severity classification, and a downloadable report.")
    st.markdown("---")

    # CHANGED: Modify inference trigger to check either uploaded file or selected sample image
    has_image = uploaded_file is not None or "sample_image" in st.session_state
    if has_image:
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image_name = uploaded_file.name
        else:
            sample_path = st.session_state["sample_image"]
            image_bgr = cv2.imread(sample_path)
            image_name = Path(sample_path).name

        if image_bgr is None:
            st.error("❌ Could not decode the uploaded image. Please try a different file.")
            return

        # Load model (cached to avoid reloading on every interaction)
        detector = load_model(weights_path)

        if detector is None:
            st.error(
                f"❌ Model weights not found at `{weights_path}`. "
                "Please train a model first or update the weights path."
            )
            return

        # Run inference
        with st.spinner("🔍 Analyzing blood smear ..."):
            result = detector.predict(
                image_bgr,
                conf=conf_threshold,
                iou=iou_threshold,
                annotate=True,
            )
            # Override image path for display
            result.image_path = image_name

        # --- Results Layout ---
        st.markdown("---")

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{result.total_rbc + result.total_parasites}</div>
                <div class="metric-label">Total Cells</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{result.total_parasites}</div>
                <div class="metric-label">Parasites Detected</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{result.parasitemia_pct:.2f}%</div>
                <div class="metric-label">Parasitemia</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            severity = classify_severity(result.parasitemia_pct)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="font-size:1.4rem;">{severity}</div>
                <div class="metric-label">Severity</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Image comparison
        img_col1, img_col2 = st.columns(2)

        with img_col1:
            st.markdown("#### 📷 Original Image")
            original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            st.image(original_rgb, use_container_width=True)

        with img_col2:
            st.markdown("#### 🎯 Detection Results")
            if result.annotated_image is not None:
                # Optionally filter out RBC boxes for cleaner visualisation
                if not show_rbc:
                    display_img = _draw_filtered(image_bgr, result)
                else:
                    display_img = result.annotated_image
                annotated_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_container_width=True)
            else:
                st.info("No detections to display.")

        # --- Detection details table ---
        st.markdown("---")
        st.markdown("### 📋 Detection Details")

        if result.detections:
            # Build a summary by class
            counts = result._per_class_counts()
            table_data = []
            for cls_name, count in sorted(counts.items()):
                avg_conf = np.mean([
                    d.confidence for d in result.detections
                    if d.class_name == cls_name
                ])
                is_parasite = "🦠 Yes" if cls_name in PARASITE_CLASSES else "🔴 No"
                table_data.append({
                    "Class": cls_name.replace("_", " ").title(),
                    "Count": count,
                    "Avg Confidence": f"{avg_conf:.3f}",
                    "Parasite?": is_parasite,
                })

            st.table(table_data)
        else:
            st.info("No objects detected at the current confidence threshold.")

        # --- Download section ---
        st.markdown("---")
        st.markdown("### 📥 Download Reports")

        dl_col1, dl_col2, dl_col3 = st.columns(3)

        with dl_col1:
            # PDF Report
            try:
                pdf_bytes = generate_pdf_report(result, result.annotated_image)
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"malaria_report_{Path(image_name).stem}.pdf",
                    mime="application/pdf",
                )
            except ImportError:
                st.warning("Install `fpdf2` for PDF generation: `pip install fpdf2`")

        with dl_col2:
            # CSV Report
            csv_data = generate_csv_report(result)
            st.download_button(
                label="📊 Download CSV Data",
                data=csv_data,
                file_name=f"detections_{Path(image_name).stem}.csv",
                mime="text/csv",
            )

        with dl_col3:
            # Annotated Image
            if result.annotated_image is not None:
                img_rgb = cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                st.download_button(
                    label="🖼️ Download Annotated Image",
                    data=buf.getvalue(),
                    file_name=f"annotated_{Path(image_name).stem}.png",
                    mime="image/png",
                )

    else:
        # Empty state
        st.markdown(
            """
            <div style="text-align: center; padding: 4rem 2rem; color: #8892b0;">
                <h3>👆 Upload a blood smear image to get started</h3>
                <p>Supports PNG, JPG, TIFF, and BMP formats.</p>
                <p style="font-size: 0.85rem; margin-top: 1rem;">
                    Tip: Use images from the BBBC041 dataset for best results.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # CHANGED: Added team credit to the bottom of the main page
    st.markdown(
        "<p style='text-align:center; color: gray; font-size: 0.8em;'>"
        "Built by Team Devions · NACOS UI × DATICAN Competition 2026"
        "</p>",
        unsafe_allow_html=True
    )


def _draw_filtered(
    image_bgr: np.ndarray,
    result: PredictionResult,
) -> np.ndarray:
    """Redraw detections showing only parasites (no healthy RBCs)."""
    from src.inference.predict import CLASS_COLORS

    img = image_bgr.copy()
    for det in result.detections:
        if det.class_name == "red_blood_cell":
            continue  # Skip healthy cells

        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
        color = CLASS_COLORS.get(det.class_name, (255, 255, 255))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            img, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return img


@st.cache_resource
def load_model(weights_path: str) -> MalariaDetector | None:
    """Load the detector model with Streamlit caching.

    st.cache_resource ensures the model is loaded once and shared
    across all users/sessions, avoiding repeated 200+ MB loads.
    """
    if not Path(weights_path).exists():
        return None
    return MalariaDetector(weights_path, device="cpu")


if __name__ == "__main__":
    main()

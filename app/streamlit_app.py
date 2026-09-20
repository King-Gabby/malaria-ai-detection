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
import time #CHANGED: Added for
from datetime import datetime
from pathlib import Path


import cv2
import numpy as np
import pandas as pd  # CHANGED: Added for batch summary table
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Add project root to path so we can import our modules
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import (  # CHANGED: Extended imports
    MalariaDetector,
    PredictionResult,
    PARASITE_CLASSES,
    UNCERTAINTY_THRESHOLD_LOW,   # CHANGED
    UNCERTAINTY_THRESHOLD_HIGH,  # CHANGED
    UNCERTAIN_COLOR,             # CHANGED
)


# ---------------------------------------------------------------------------
# PDF Report Generation
# ---------------------------------------------------------------------------
# CHANGE 3 — Updated PDF signature with patient details
def generate_pdf_report(
    result: PredictionResult,
    annotated_img: np.ndarray,
    uncertain_count: int = 0,
    patient_details: dict = None,
    report_meta: dict = None,
    scan_status: str = "",
    verified_data: dict = None,
) -> bytes:
    """Generate a downloadable PDF report with detection results.

    The PDF includes:
      • Enhanced clinical header with patient information
      • Parasitemia percentage (large, bold)
      • Clinical Interpretation
      • Stage-Specific Clinical Notes
      • Recommendations
      • Per-class detection counts table
      • Annotated image snapshot

    Returns:
        PDF file as bytes for Streamlit download button.
    """
    from fpdf import FPDF

    # FIX 2 — Use clinician-verified parasitemia when available,
    # otherwise fall back to raw model output
    if verified_data and (
        verified_data.get('accepted_uncertain', 0) +
        verified_data.get('rejected_uncertain', 0)
    ) > 0:
        display_parasitemia = verified_data['verified_parasitemia_pct']
        display_parasite_count = verified_data['verified_parasites']
        display_rbc_count = verified_data['verified_rbc']
        is_verified = True
    else:
        display_parasitemia = result.parasitemia_pct
        display_parasite_count = result.total_parasites
        display_rbc_count = result.total_rbc
        is_verified = False

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # CHANGE 4 — Enhanced clinical PDF header

    # ── Title bar ──  (FIX 4: add [Clinician-Verified] when applicable)
    title_text = "Laboratory AI Screening Report"
    if is_verified:
        title_text += "  [Clinician-Verified]"

    pdf.set_fill_color(15, 52, 96)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 14, title_text,
             ln=True, align="C", fill=True)
    pdf.set_font("Helvetica", "", 8)
    pdf.cell(0, 6,
             "Malaria Parasite Detection · YOLOv8 Object Detection · "
             "BBBC041 Dataset · Team Devions · NACOS UI x DATICAN 2026",
             ln=True, align="C", fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # ── Report metadata row ──
    if report_meta:
        pdf.set_font("Helvetica", "", 8)
        pdf.set_fill_color(245, 247, 250)
        meta_line = (
            f"Report No: {report_meta.get('report_number', 'N/A')}     "
            f"Study ID: {report_meta.get('study_id', 'N/A')}     "
            f"Date: {report_meta.get('date', '')}  {report_meta.get('time', '')}     "
            f"Image: {Path(result.image_path).name}"
        )
        pdf.cell(0, 7, meta_line, ln=True, fill=True, align="C")
        pdf.ln(4)

    # ── Patient information table ──
    if patient_details and any(
        v for v in patient_details.values() if v
    ):
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Patient Information", ln=True)
        pdf.ln(1)

        col_label = 52
        col_value = 128
        row_h = 7

        def _pdf_row(label, value, bold_value=False):
            if not value and value != 0:
                return
            pdf.set_fill_color(237, 242, 247)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(col_label, row_h, f"  {label}",
                     border=1, fill=True)
            pdf.set_fill_color(255, 255, 255)
            pdf.set_font(
                "Helvetica", "B" if bold_value else "", 9
            )
            pdf.cell(col_value, row_h,
                     f"  {strip_emoji_for_pdf(str(value))}",
                     border=1, fill=True, ln=True)

        if patient_details.get("name"):
            _pdf_row("Patient Name",
                     patient_details["name"], bold_value=True)
        if patient_details.get("patient_id"):
            _pdf_row("Patient / Sample ID",
                     patient_details["patient_id"])

        age_sex_parts = []
        if patient_details.get("age"):
            age_sex_parts.append(
                f"{patient_details['age']} years"
            )
        if patient_details.get("sex"):
            age_sex_parts.append(patient_details["sex"])
        if age_sex_parts:
            _pdf_row("Age / Sex", "  |  ".join(age_sex_parts))

        if patient_details.get("clinician"):
            _pdf_row("Requesting Clinician",
                     patient_details["clinician"])
        if patient_details.get("facility"):
            _pdf_row("Health Facility",
                     patient_details["facility"])

        # Scan status
        if scan_status:
            _pdf_row("AI Screening Status",
                     scan_status, bold_value=True)

        if patient_details.get("notes"):
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 6, "Clinical Notes:", ln=True)
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(
                0, 5,
                f"  {strip_emoji_for_pdf(patient_details['notes'])}"
            )

        pdf.ln(6)

    else:
        # Fallback minimal header when no patient details provided
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(
            0, 7,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ln=True, align="C"
        )
        pdf.cell(
            0, 7,
            f"Image: {Path(result.image_path).name}",
            ln=True, align="C"
        )
        pdf.ln(6)

    # --- Parasitemia ---
    pdf.set_font("Helvetica", "B", 16)
    severity = classify_severity(display_parasitemia)
    label = "Verified Parasitemia" if is_verified else "Estimated Parasitemia"
    pdf.cell(0, 10, strip_emoji_for_pdf(f"{label}: {display_parasitemia:.2f}%"), ln=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, strip_emoji_for_pdf(f"Severity: {strip_emoji_for_pdf(severity)}"), ln=True)
    pdf.ln(6)

    # SECTION 1 — Clinical Interpretation
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, strip_emoji_for_pdf("Clinical Interpretation"), ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(2)

    p_counts = {k: v for k, v in result._per_class_counts().items() if k in PARASITE_CLASSES}

    if display_parasite_count == 0:
        interpretation = (
            "No malaria parasites were detected in this blood smear image. "
            "While this result is encouraging, a negative AI screening result "
            "does not exclude malaria infection. Clinical correlation and "
            "repeat microscopy are recommended if symptoms persist."
        )
    elif display_parasitemia < 1:
        dominant = max(p_counts, key=p_counts.get).replace("_", " ") if p_counts else "None"
        interpretation = (
            f"Low parasitemia detected ({display_parasitemia:.2f}%). "
            f"The predominant parasite stage identified is {dominant}. "
            "At this parasitemia level, the patient may present with mild "
            "symptoms. Microscopy confirmation is advised before initiating "
            "treatment."
        )
    elif display_parasitemia < 5:
        dominant = max(p_counts, key=p_counts.get).replace("_", " ") if p_counts else "None"
        interpretation = (
            f"Moderate parasitemia detected ({display_parasitemia:.2f}%). "
            f"The predominant parasite stage is {dominant}, suggesting active "
            "infection. WHO guidelines recommend prompt microscopy confirmation "
            "and initiation of appropriate antimalarial therapy. "
            "Clinical monitoring is advised."
        )
    else:
        dominant = max(p_counts, key=p_counts.get).replace("_", " ") if p_counts else "None"
        interpretation = (
            f"High parasitemia detected ({display_parasitemia:.2f}%). "
            f"The predominant stage is {dominant}. This exceeds the WHO severe "
            "malaria threshold of 5%. Immediate clinical review is strongly "
            "advised. Consider IV artesunate per WHO severe malaria protocol. "
            "This AI result must be confirmed by urgent microscopy."
        )

    pdf.multi_cell(0, 6, strip_emoji_for_pdf(interpretation))
    pdf.ln(4)

    # SECTION 2 — Stage-Specific Notes
    if display_parasite_count > 0:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 9, strip_emoji_for_pdf("Stage-Specific Clinical Notes"), ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.ln(2)

        notes_mapping = {
            "ring": (
                "Ring stage (early trophozoite): Most common early-infection stage. "
                "Indicates recent or active RBC invasion."
            ),
            "trophozoite": (
                "Trophozoite stage: Active feeding stage within RBC. "
                "Indicates established infection requiring treatment."
            ),
            "schizont": (
                "Schizont stage: Replicative stage. Presence may indicate "
                "higher severity — schizonts in peripheral blood are associated "
                "with more serious infection."
            ),
            "gametocyte": (
                "Gametocyte stage: Sexual stage transmissible to mosquitoes. "
                "Patient may be infectious. Consider epidemiological implications."
            )
        }

        # Filter to parasite classes with count > 0 and sort by count descending
        active_parasites = {k: v for k, v in p_counts.items() if v > 0}
        sorted_parasites = sorted(active_parasites.items(), key=lambda item: item[1], reverse=True)

        for class_name, count in sorted_parasites:
            note = notes_mapping.get(class_name, "")
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, strip_emoji_for_pdf(f"  {class_name.title()} ({count} detected):"), ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6, strip_emoji_for_pdf(f"    {note}"))
            pdf.ln(2)

    # SECTION 3 — Recommendations
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, strip_emoji_for_pdf("Recommendations"), ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(2)

    recommendations = []
    recommendations.append("Confirm findings with Giemsa-stained thick and thin blood smear microscopy.")

    if display_parasite_count > 0:
        recommendations.append("Initiate antimalarial treatment per national/WHO guidelines after microscopy confirmation.")

    if uncertain_count > 0:
        recommendations.append(
            f"{uncertain_count} detection(s) were flagged as low-confidence by the AI model. "
            "These require particular attention during microscopy review."
        )

    if display_parasitemia >= 5:
        recommendations.append(
            "Parasitemia exceeds WHO severe malaria threshold (5%). "
            "Consider IV artesunate and hospital admission per severe malaria protocol."
        )

    if 0 < display_parasitemia < 5:
        recommendations.append("Monitor patient response to treatment and repeat blood smear at 24 and 48 hours.")

    recommendations.append(
        "This report is AI-generated and must not be used as the sole basis "
        "for clinical decisions. Always correlate with clinical presentation."
    )

    for i, rec in enumerate(recommendations, 1):
        pdf.multi_cell(0, 6, strip_emoji_for_pdf(f"  {i}. {rec}"))
        pdf.ln(1)
    pdf.ln(4)

    # --- Detection summary table ---
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, strip_emoji_for_pdf("Detection Summary"), ln=True)
    pdf.set_font("Helvetica", "", 10)

    # FIX 2 — Use verified per-class counts when clinician decisions exist
    if is_verified and verified_data.get('verified_class_counts'):
        counts = verified_data['verified_class_counts']
    else:
        counts = result._per_class_counts()
    pdf.cell(90, 7, strip_emoji_for_pdf("Class"), border=1, align="C")
    pdf.cell(40, 7, strip_emoji_for_pdf("Count"), border=1, align="C", ln=True)

    for cls_name, count in counts.items():
        pdf.cell(90, 7, strip_emoji_for_pdf(cls_name), border=1)
        pdf.cell(40, 7, strip_emoji_for_pdf(str(count)), border=1, align="C", ln=True)

    pdf.cell(90, 7, strip_emoji_for_pdf("TOTAL CELLS"), border=1, fill=True)
    pdf.cell(40, 7, strip_emoji_for_pdf(str(display_rbc_count + display_parasite_count)), border=1, align="C", ln=True, fill=True)
    pdf.ln(2)

    # FIX 3 — Note when counts reflect clinician verification
    if is_verified:
        pdf.set_font("Helvetica", "I", 8)
        pdf.cell(
            0, 6,
            strip_emoji_for_pdf(
                f"Note: Counts reflect clinician verification. "
                f"{verified_data['accepted_uncertain']} accepted, "
                f"{verified_data['rejected_uncertain']} rejected, "
                f"{verified_data['pending_uncertain']} pending review."
            ),
            ln=True
        )
        pdf.ln(2)
    else:
        pdf.ln(4)

    # --- Annotated image ---
    if annotated_img is not None:
        # Save image to temp file for FPDF
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            Image.fromarray(img_rgb).save(tmp.name)
            pdf.set_font("Helvetica", "B", 12)
            # FIX 3 — Transparent caption when annotation shows unverified boxes
            image_caption = "Annotated Image"
            if is_verified:
                image_caption += (
                    " (shows original AI detections — see verified counts "
                    "above for clinician-reviewed totals)"
                )
            pdf.cell(0, 8, strip_emoji_for_pdf(image_caption), ln=True)
            pdf.image(tmp.name, w=180)
            if is_verified:
                pdf.set_font("Helvetica", "I", 8)
                pdf.ln(2)
                pdf.multi_cell(
                    0, 5,
                    strip_emoji_for_pdf(
                        "Note: Bounding boxes above reflect all AI-proposed "
                        "detections including any subsequently rejected by "
                        "clinician review. Refer to the verified Detection "
                        "Summary table above for the clinically confirmed count."
                    )
                )

    # --- Disclaimer ---
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 5, strip_emoji_for_pdf(
        "DISCLAIMER: This is an AI-assisted screening tool for research and "
        "educational purposes only. It is NOT a certified medical diagnostic device. "
        "All findings must be confirmed by a qualified microscopist."
    ))

    return bytes(pdf.output())


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
        return "🔴 High parasitemia (> 5%) = SEVERE"

def strip_emoji_for_pdf(text: str) -> str:
    """Remove emoji and non-Latin-1 characters for PDF compatibility.

    fpdf2's default core fonts (helvetica) only support Latin-1 encoding.
    This strips emoji while preserving the readable text content.
    """
    return text.encode("latin-1", errors="ignore").decode("latin-1").strip()


# ---------------------------------------------------------------------------
# Helper — downscale oversized uploads to prevent OOM on constrained deploys
# ---------------------------------------------------------------------------
def _downscale_if_needed(image_bgr: np.ndarray, max_dimension: int = 1600) -> np.ndarray:
    """Downscale an image if its longest side exceeds max_dimension.

    Uploaded images (especially phone camera photos) can be far
    larger than needed for inference, consuming excessive memory on
    memory-constrained deployments. This caps the longest dimension
    while preserving aspect ratio, using high-quality interpolation
    for downscaling (INTER_AREA is optimal for shrinking).

    Args:
        image_bgr: BGR image array from cv2.imdecode or cv2.imread.
        max_dimension: Maximum allowed size for the longest side.
                       Default 1600 matches BBBC041 native resolution
                       (1600x1200) so no quality is lost on standard
                       microscopy images.

    Returns:
        The original array unchanged if already within limits,
        otherwise a proportionally downscaled copy.
    """
    h, w = image_bgr.shape[:2]
    longest_side = max(h, w)

    if longest_side <= max_dimension:
        return image_bgr

    scale = max_dimension / longest_side
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

# ---------------------------------------------------------------------------
# CHANGED: Helper — classify a single detection's uncertainty tier
# ---------------------------------------------------------------------------
def _is_uncertain(confidence: float) -> bool:
    """Return True if the detection falls in the uncertain tier (0.35–0.45)."""
    return UNCERTAINTY_THRESHOLD_LOW <= confidence <= UNCERTAINTY_THRESHOLD_HIGH


# ---------------------------------------------------------------------------
# CHANGED: Helper — count uncertainty tiers for a PredictionResult
# ---------------------------------------------------------------------------
def _count_tiers(result: PredictionResult):
    """Count uncertain vs confident parasite detections.

    Returns (uncertain_count, confident_parasite_count).
    Only non-RBC detections are considered.
    """
    uncertain = 0
    confident = 0
    for d in result.detections:
        if d.class_name not in PARASITE_CLASSES:
            continue
        if _is_uncertain(d.confidence):
            uncertain += 1
        elif d.confidence > UNCERTAINTY_THRESHOLD_HIGH:
            confident += 1
        # Detections below UNCERTAINTY_THRESHOLD_LOW are neither tier;
        # they passed the slider but are not part of uncertainty flagging.
    return uncertain, confident


# ---------------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------------
def show_splash_screen():  #CHANGED
    # PART 1
    """Display a beautiful 2.5-second splash screen with dark theme, animations, and preloader."""
    splash_html = """
    <style>
        /* Splash Screen Container */
        .splash-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 50%, #0f3460 100%);
            background-image: radial-gradient(
                circle, 
                rgba(100, 255, 218, 0.03) 2px, 
                transparent 2px
            ), linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 50%, #0f3460 100%);
            background-size: 40px 40px, 100% 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 9999;
            animation: fadeIn 0.6s ease-in-out;
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
        }

        @keyframes fadeOut {
            from {
                opacity: 1;
            }
            to {
                opacity: 0;
            }
        }

        @keyframes slideUp {
            from {
                transform: translateY(30px);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }

        @keyframes pulse {
            0%, 100% {
                transform: scale(1);
            }
            50% {
                transform: scale(1.05);
            }
        }

        @keyframes rotate {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }

        @keyframes slideProgress {
            0% {
                width: 0%;
            }
            100% {
                width: 100%;
            }
        }

        /* Splash Card */
        .splash-card {
            background: linear-gradient(135deg, rgba(26, 26, 46, 0.95) 0%, rgba(15, 52, 96, 0.95) 100%);
            border: 2px solid rgba(100, 255, 218, 0.2);
            border-radius: 20px;
            padding: 3.5rem 3rem;
            max-width: 420px;
            text-align: center;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.6),
                        inset 0 1px 0 rgba(100, 255, 218, 0.1);
            animation: slideUp 0.8s ease-out;
            backdrop-filter: blur(10px);
        }

        /* Logo Circle */
        .logo-circle {
            width: 100px;
            height: 100px;
            background: linear-gradient(135deg, #e94560 0%, #ff6b7a 100%);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 1.5rem;
            font-size: 4rem;
            box-shadow: 0 10px 40px rgba(233, 69, 96, 0.3);
            animation: pulse 2s ease-in-out infinite;
        }

        /* Title */
        .splash-title {
            font-size: 1.8rem;
            font-weight: 800;
            color: #ffffff;
            margin: 1rem 0 0.8rem;
            letter-spacing: -0.02em;
            line-height: 1.3;
        }

        /* Subtitle */
        .splash-subtitle {
            font-size: 0.95rem;
            color: #8892b0;
            margin-bottom: 0.3rem;
            font-weight: 500;
            letter-spacing: 0.01em;
        }

        /* Status Text */
        .splash-status {
            font-size: 0.9rem;
            color: #64ffda;
            margin: 1.5rem 0 1.2rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.6rem;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background: #64ffda;
            border-radius: 50%;
            animation: pulse 1.5s ease-in-out infinite;
        }


        /* Preloader Bar Container */
        .preloader-container {
            width: 100%;
            height: 4px;
            background: rgba(100, 255, 218, 0.1);
            border-radius: 2px;
            overflow: hidden;
            margin-top: 1.2rem;
            box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.3);
        }

        /* Animated Progress Bar */
        .preloader-bar {
            height: 100%;
            background: linear-gradient(90deg, #64ffda 0%, #00d4ff 50%, #64ffda 100%);
            border-radius: 2px;
            animation: slideProgress 2.5s cubic-bezier(0.25, 0.46, 0.45, 0.94) forwards;
            box-shadow: 0 0 10px rgba(100, 255, 218, 0.6);
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .splash-card {
                padding: 2.5rem 2rem;
                max-width: 340px;
            }

            .splash-title {
                font-size: 1.5rem;
            }

            .logo-circle {
                width: 80px;
                height: 80px;
                font-size: 3.2rem;
            }
        }

        @media (max-width: 480px) {
            .splash-card {
                padding: 2rem 1.5rem;
                max-width: 90%;
            }

            .splash-title {
                font-size: 1.3rem;
            }

            .splash-status {
                font-size: 0.85rem;
            }

            .logo-circle {
                width: 70px;
                height: 70px;
                font-size: 2.8rem;
            }
        }
    </style>

    <div class="splash-overlay">
        <div class="splash-card">
            <div class="logo-circle">🔬</div>
            <h1 class="splash-title">🩺 RaphaID AI</h1>
            <p class="splash-subtitle">Offline Multi-Disease Diagnostic Tool</p>
            <div class="splash-status">
                <span class="status-dot"></span>
                <span id="status-text">Loading detection model...</span>
            </div>
            <div class="preloader-container">
                <div class="preloader-bar"></div>
            </div>
        </div>
    </div>

    <script>
        const messages = [
            "Preparing quality assessment...",
            "Initialising YOLOv8 engine...",
            "Loading WHO classification thresholds...",
            "Ready."
        ];
        let index = 0;
        const statusText = document.getElementById("status-text");
        const interval = setInterval(() => {
            if (index < messages.length) {
                statusText.textContent = messages[index];
                index++;
            } else {
                clearInterval(interval);
            }
        }, 500);
    </script>
    """

    # Display splash screen HTML  #CHANGED
    st.markdown(splash_html, unsafe_allow_html=True)
    
    # Keep placeholder for 2.5 seconds #CHANGED
    splash_placeholder = st.empty()
    time.sleep(2.5)
    
    # Clear splash screen #CHANGED
    st.rerun()



# PART 3
def render_dashboard():
    # FIX 4 — Hero banner redesign (using global contrast colors from FIX 1)
    st.markdown("""
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
    
    _, cta_col, _ = st.columns([0.8, 1.4, 0.8])
    with cta_col:
        if st.button(" Start Clinical Diagnosis ", use_container_width=True, 
                     type="primary"):
            st.session_state["current_page"] = "diagnosis"
            st.rerun()

    # FIX 5 — Quick Facts: card grid replacing all tables
    st.markdown("### 📋 Quick Facts")
    fact_items = [
        ("🧠", "Model", "YOLOv8n"),
        ("🩸", "Dataset", "BBBC041"),
        ("🔬", "Species", "P. vivax"),
        ("🎯", "mAP@0.5", "63.1%"),
        ("📐", "Precision", "58.4%"),
        ("📊", "Recall", "67.9%"),
        ("⚖️", "F1 Score", "62.8%"),
        ("⚡", "Inference", "140ms/img"),
        ("🔍", "Detection", "5-class"),
        ("🏥", "WHO Classification", "Enabled"),
        ("👨⚕️", "Human Verification", "Enabled"),
        ("📄", "Clinical Reports", "PDF + CSV"),
    ]
    
    # 4 columns grid, reduced vertical padding and spacing (FIX 3)
    num_cols = 4
    for row_start in range(0, len(fact_items), num_cols):
        row_items = fact_items[row_start:row_start + num_cols]
        cols = st.columns(num_cols)
        for col, (icon, label, value) in zip(cols, row_items):
            with col:
                st.markdown(f"""
                <div style="background: #16213e; 
                            border: 1px solid rgba(255,255,255,0.12); 
                            border-radius: 10px; padding: 0.8rem 0.7rem; 
                            text-align: center; margin-bottom: 0.7rem; 
                            min-height: 85px; display: flex; 
                            flex-direction: column; justify-content: center;">
                    <div style="font-size: 1.3rem;">{icon}</div>
                    <div style="font-size: 0.68rem; color: #9AA4B2; 
                                text-transform: uppercase; letter-spacing: 0.05em; 
                                margin-top: 0.35rem; font-weight: 600;">{label}</div>
                    <div style="font-size: 1rem; font-weight: 700; 
                                color: #FFFFFF; margin-top: 0.25rem;">{value}</div>
                </div>
                """, unsafe_allow_html=True)

    # FIX 6 — Clinical Workflow: horizontal wrapping flow card
    st.markdown("### 🔄 Clinical Workflow")
    st.markdown("""
    <div style="background: #16213e; border: 1px solid rgba(255,255,255,0.12); 
                border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 1rem;
                display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 0.8rem;">
        <div style="text-align: center; flex: 1; min-width: 95px;">
            <div style="font-size: 1.5rem; margin-bottom: 0.2rem;">🧫</div>
            <div style="font-size: 0.85rem; font-weight: 600; color: #FFFFFF;">Upload</div>
            <div style="font-size: 0.68rem; color: #9AA4B2;">Submit Smear</div>
        </div>
        <div style="font-size: 1.2rem; color: #64ffda; font-weight: bold; text-align: center;">→</div>
        <div style="text-align: center; flex: 1; min-width: 95px;">
            <div style="font-size: 1.5rem; margin-bottom: 0.2rem;">🔍</div>
            <div style="font-size: 0.85rem; font-weight: 600; color: #FFFFFF;">Quality</div>
            <div style="font-size: 0.68rem; color: #9AA4B2;">Assess Image</div>
        </div>
        <div style="font-size: 1.2rem; color: #64ffda; font-weight: bold; text-align: center;">→</div>
        <div style="text-align: center; flex: 1; min-width: 95px;">
            <div style="font-size: 1.5rem; margin-bottom: 0.2rem;">🧠</div>
            <div style="font-size: 0.85rem; font-weight: 600; color: #FFFFFF;">Detect</div>
            <div style="font-size: 0.68rem; color: #9AA4B2;">YOLOv8 AI Scan</div>
        </div>
        <div style="font-size: 1.2rem; color: #64ffda; font-weight: bold; text-align: center;">→</div>
        <div style="text-align: center; flex: 1; min-width: 95px;">
            <div style="font-size: 1.5rem; margin-bottom: 0.2rem;">👨‍⚕️</div>
            <div style="font-size: 0.85rem; font-weight: 600; color: #FFFFFF;">Verify</div>
            <div style="font-size: 0.68rem; color: #9AA4B2;">Clinical Review</div>
        </div>
        <div style="font-size: 1.2rem; color: #64ffda; font-weight: bold; text-align: center;">→</div>
        <div style="text-align: center; flex: 1; min-width: 95px;">
            <div style="font-size: 1.5rem; margin-bottom: 0.2rem;">📄</div>
            <div style="font-size: 0.85rem; font-weight: 600; color: #FFFFFF;">Report</div>
            <div style="font-size: 0.68rem; color: #9AA4B2;">PDF & CSV</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key Features card grid (using contrast palette, FIX 3 compact heights)
    st.markdown("### ✓ Key Features")
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    with feat_col1:
        st.markdown("""
        <div style="background: #16213e; border: 1px solid rgba(255,255,255,0.12);
                    border-radius: 10px; padding: 0.8rem 0.7rem; min-height: 85px;
                    display: flex; flex-direction: column; justify-content: center; margin-bottom: 0.5rem;">
            <div style="font-weight: 600; color: #FFFFFF; font-size: 0.85rem;">🤖 Clinical AI Assist</div>
            <div style="font-size: 0.75rem; color: #D8DEE9; margin-top: 0.2rem;">
                Human-in-the-loop verification & explainable predictions.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with feat_col2:
        st.markdown("""
        <div style="background: #16213e; border: 1px solid rgba(255,255,255,0.12);
                    border-radius: 10px; padding: 0.8rem 0.7rem; min-height: 85px;
                    display: flex; flex-direction: column; justify-content: center; margin-bottom: 0.5rem;">
            <div style="font-weight: 600; color: #FFFFFF; font-size: 0.85rem;">📋 Quality & Reporting</div>
            <div style="font-size: 0.75rem; color: #D8DEE9; margin-top: 0.2rem;">
                Pre-scan quality checks and clinician-verified PDF/CSV reports.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with feat_col3:
        st.markdown("""
        <div style="background: #16213e; border: 1px solid rgba(255,255,255,0.12);
                    border-radius: 10px; padding: 0.8rem 0.7rem; min-height: 85px;
                    display: flex; flex-direction: column; justify-content: center; margin-bottom: 0.5rem;">
            <div style="font-weight: 600; color: #FFFFFF; font-size: 0.85rem;">⚡ CPU Optimization</div>
            <div style="font-size: 0.75rem; color: #D8DEE9; margin-top: 0.2rem;">
                Batch processing support optimized for standard CPU inference.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    # Relocated disclaimers using global contrast palette
    st.warning(
        "Disclaimer: For research use and decision support only. Not intended as a "
        "standalone diagnostic tool. All findings require confirmation "
        "by a qualified microscopist."
    )
    st.markdown("<br>", unsafe_allow_html=True)

def main():
    # --- Page config ---
    st.set_page_config(
        page_title="RaphaID AI",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Force dark theme configurations programmatically
    try:
        st.config.set_option("theme.base", "dark")
        st.config.set_option("theme.primaryColor", "#64ffda")
        st.config.set_option("theme.backgroundColor", "#0d0d1a")
        st.config.set_option("theme.secondaryBackgroundColor", "#16213e")
        st.config.set_option("theme.textColor", "#D8DEE9")
    except Exception:
        pass
    # ------ SESSION STATE------
    # Track whether we've shown the splash this session
    if "splash_shown" not in st.session_state:
        st.session_state.splash_shown = False
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False
    # PART 2a
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "dashboard"
    if "weights_path" not in st.session_state:
        st.session_state["weights_path"] = "models/best.pt"
    if "conf_threshold" not in st.session_state:
        st.session_state["conf_threshold"] = 0.20
    if "iou_threshold" not in st.session_state:
        st.session_state["iou_threshold"] = 0.45
    if "show_rbc" not in st.session_state:
        st.session_state["show_rbc"] = False



    # Show splash screen only on first load
    if not st.session_state.splash_shown:
        st.session_state.splash_shown = True
        show_splash_screen()
        st.stop()




    # CHANGE 1 — Patient intake session state
    if "patient_details" not in st.session_state:
        st.session_state["patient_details"] = {
            "name": "",
            "age": None,
            "sex": "",
            "patient_id": "",
            "clinician": "",
            "facility": "",
            "notes": "",
        }

    # FEATURE 2 — Session state for verification decisions
    if "verification_decisions" not in st.session_state:
        st.session_state["verification_decisions"] = {}

    # BUGFIX — Decouple inference trigger from results persistence using session_state
    if "current_result" not in st.session_state:
        st.session_state["current_result"] = None
    if "current_image_bgr" not in st.session_state:
        st.session_state["current_image_bgr"] = None
    if "current_image_name" not in st.session_state:
        st.session_state["current_image_name"] = None


    # Auto-generate report number and study ID for this session
    if "report_meta" not in st.session_state:
        import random
        import string
        session_date = datetime.now().strftime("%Y%m%d")
        report_seq = random.randint(10000, 99999)
        st.session_state["report_meta"] = {
            "report_number": f"RPT-{session_date}-{report_seq}",
            "study_id": f"STD-{''.join(random.choices(string.ascii_uppercase, k=2))}{random.randint(100,999)}",
            "date": datetime.now().strftime("%d %B %Y"),
            "time": datetime.now().strftime("%H:%M"),
        }


    # Local variables for model settings retrieved from session state
    weights_path = st.session_state["weights_path"]
    conf_threshold = st.session_state["conf_threshold"]
    iou_threshold = st.session_state["iou_threshold"]
    show_rbc = st.session_state["show_rbc"]

    # CHANGE — CSS Polish
    st.markdown("""
    <style>
    /* Force dark theme regardless of user/browser preference */
    html, body, [data-testid="stAppViewContainer"], 
    [data-testid="stHeader"], [data-testid="stToolbar"],
    .main, .block-container {
        background-color: #0d0d1a !important;
        color: #D8DEE9 !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0d0d1a !important;
    }
    
    [data-testid="stSidebar"] > div {
        background-color: #0d0d1a !important;
    }
    
    /* Force all headings to white regardless of theme */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }
    
    /* Force body text color */
    p, span, label, div {
        color: #D8DEE9;
    }
    
    /* Force input fields to dark theme */
    .stTextInput input, .stNumberInput input, .stTextArea textarea,
    .stSelectbox > div > div {
        background-color: #16213e !important;
        color: #D8DEE9 !important;
        border-color: rgba(255,255,255,0.12) !important;
    }
    
    /* Force selectbox dropdown menu to dark theme */
    [data-baseweb="select"] > div {
        background-color: #16213e !important;
        color: #D8DEE9 !important;
    }
    
    [data-baseweb="popover"] {
        background-color: #16213e !important;
    }
    
    [role="listbox"] {
        background-color: #16213e !important;
    }
    
    [role="option"] {
        background-color: #16213e !important;
        color: #D8DEE9 !important;
    }
    
    /* Force file uploader to dark theme */
    [data-testid="stFileUploader"] {
        background-color: rgba(255,255,255,0.03) !important;
        color: #D8DEE9 !important;
    }
    
    [data-testid="stFileUploaderDropzone"] {
        background-color: rgba(255,255,255,0.03) !important;
        border-color: rgba(255,255,255,0.12) !important;
    }
    
    /* Force expander to dark theme */
    [data-testid="stExpander"] {
        background-color: rgba(255,255,255,0.02) !important;
        border-color: rgba(255,255,255,0.12) !important;
    }
    
    [data-testid="stExpander"] summary {
        color: #D8DEE9 !important;
        background-color: transparent !important;
    }
    
    /* Force dataframe/table to dark theme */
    [data-testid="stDataFrame"] {
        background-color: #16213e !important;
    }
    
    /* Force radio buttons and checkboxes to dark theme */
    [data-testid="stRadio"] label, [data-testid="stCheckbox"] label {
        color: #D8DEE9 !important;
    }
    
    /* Force metric widgets to dark theme */
    [data-testid="stMetric"] {
        background-color: transparent !important;
    }
    [data-testid="stMetricLabel"] {
        color: #9AA4B2 !important;
    }
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }

        /* ============================================================
           GLOBAL LAYOUT & SPACING
        ============================================================ */
        /* FIX 2 — Fix horizontal scrolling */
        .main .block-container {
            max-width: 100% !important;
            overflow-x: hidden !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        .main .block-container > div {
            max-width: 100% !important;
        }

        /* Consistent vertical rhythm between all elements */
        .element-container {
            margin-bottom: 0.4rem !important;
        }

        /* ============================================================
           TYPOGRAPHY
        ============================================================ */
        html, body, [class*="css"] {
            font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
            -webkit-font-smoothing: antialiased;
        }

        h1, h2, h3, h4 {
            letter-spacing: -0.02em;
            line-height: 1.2;
        }

        p, li {
            line-height: 1.65;
            color: #a8b2d1;
        }

        /* ============================================================
           MAIN HEADER
        ============================================================ */
        .main-header {
            text-align: center;
            padding: 2.5rem 2rem;
            background: linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 40%, #0f3460 100%);
            border-radius: 16px;
            margin-bottom: 2rem;
            color: white;
            border: 1px solid rgba(100, 255, 218, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        }

        .main-header h1 {
            color: #e94560;
            font-size: 2.4rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            letter-spacing: -0.03em;
        }

        .main-header p {
            color: #a8b2d1;
            font-size: 1.05rem;
            margin: 0;
            font-weight: 400;
        }

        /* ============================================================
           METRIC CARDS
        ============================================================ */
        .metric-card {
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            border: 1px solid #233554;
            border-radius: 12px;
            padding: 1.4rem 1.2rem;
            text-align: center;
            color: white;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
            height: 100%;
        }

        .metric-card:hover {
            border-color: rgba(100, 255, 218, 0.3);
            box-shadow: 0 4px 20px rgba(100, 255, 218, 0.08);
        }

        .metric-value {
            font-size: 2.2rem;
            font-weight: 800;
            color: #64ffda;
            letter-spacing: -0.02em;
            line-height: 1;
        }

        .metric-label {
            font-size: 0.8rem;
            color: #8892b0;
            margin-top: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 500;
        }

        /* ============================================================
           SEVERITY BADGE
        ============================================================ */
        .severity-badge {
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
            font-weight: 700;
            font-size: 1rem;
            text-align: center;
        }

        /* ============================================================
           BUTTONS
        ============================================================ */
        .stButton > button {
            background: linear-gradient(135deg, #e94560, #c23152) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.7rem 2rem !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            letter-spacing: 0.01em !important;
            transition: all 0.25s ease !important;
            box-shadow: 0 2px 8px rgba(233, 69, 96, 0.3) !important;
        }

        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(233, 69, 96, 0.45) !important;
        }

        .stButton > button:active {
            transform: translateY(0px) !important;
            box-shadow: 0 2px 8px rgba(233, 69, 96, 0.3) !important;
        }

        /* Primary button (Analyse Blood Smear) */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #00c2a8, #009688) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.7rem 2rem !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            letter-spacing: 0.01em !important;
            transition: all 0.25s ease !important;
            box-shadow: 0 2px 8px rgba(0, 194, 168, 0.3) !important;
        }

        /* FIX 7 — Sidebar button color */
        [data-testid="stSidebar"] div[data-testid="stButton"] > button {
            background: rgba(255,255,255,0.03) !important;
            background-image: none !important;
            color: #D8DEE9 !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
            border-radius: 8px !important;
            box-shadow: none !important;
            font-weight: 500 !important;
            padding: 0.6rem 1rem !important;
        }
        [data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] {
            background: rgba(100,255,218,0.12) !important;
            background-image: none !important;
            color: #64ffda !important;
            border: 1px solid rgba(100,255,218,0.4) !important;
            box-shadow: none !important;
        }

        /* Main content CTA — Begin New Diagnosis / Analyse Slide — 
           maximum specificity to override any sidebar bleed */
        section.main div[data-testid="stButton"] > button[kind="primary"],
        div[data-testid="stAppViewContainer"] section.main 
            div[data-testid="stButton"] > button[kind="primary"] {
            background: linear-gradient(135deg, #e94560, #c23152) !important;
            background-color: #e94560 !important;
            background-image: linear-gradient(135deg, #e94560, #c23152) !important;
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.85rem 2rem !important;
            font-weight: 700 !important;
            font-size: 1.05rem !important;
            box-shadow: 0 4px 16px rgba(233, 69, 96, 0.35) !important;
            transition: all 0.25s ease !important;
        }
        
        section.main div[data-testid="stButton"] > button[kind="primary"] p,
        section.main div[data-testid="stButton"] > button[kind="primary"] span,
        section.main div[data-testid="stButton"] > button[kind="primary"] div {
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
        }
        
        section.main div[data-testid="stButton"] > button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 24px rgba(233, 69, 96, 0.5) !important;
            color: #FFFFFF !important;
        }

        section.main div[data-testid="stButton"] > button[kind="primary"]:active {
            transform: translateY(0px) !important;
            box-shadow: 0 4px 16px rgba(233, 69, 96, 0.35) !important;
        }


        /* ============================================================
           SIDEBAR
        ============================================================ */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d0d1a 0%, #1a1a2e 100%);
            border-right: 1px solid #233554;
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 1.5rem !important;
            padding-left: 1.2rem !important;
            padding-right: 1.2rem !important;
        }

        /* Sidebar sliders */
        [data-testid="stSlider"] > div {
            padding-top: 0.3rem;
            padding-bottom: 0.8rem;
        }

        /* Sidebar metrics */
        [data-testid="metric-container"] {
            background: rgba(35, 53, 84, 0.4);
            border: 1px solid #233554;
            border-radius: 8px;
            padding: 0.6rem 0.8rem;
        }

        /* ============================================================
           STREAMLIT NATIVE ELEMENTS — DARK THEME POLISH
        ============================================================ */

        /* File uploader */
        [data-testid="stFileUploader"] {
            background: rgba(26, 26, 46, 0.6);
            border: 2px dashed #233554;
            border-radius: 12px;
            padding: 1rem;
            transition: border-color 0.2s ease;
        }

        [data-testid="stFileUploader"]:hover {
            border-color: rgba(100, 255, 218, 0.4);
        }

        /* Expanders */
        [data-testid="stExpander"] {
            background: rgba(26, 26, 46, 0.5);
            border: 1px solid #233554 !important;
            border-radius: 10px !important;
            margin-bottom: 0.8rem;
        }

        [data-testid="stExpander"] summary {
            font-weight: 600;
            color: #a8b2d1;
            padding: 0.8rem 1rem;
        }

        /* Info boxes */
        [data-testid="stInfo"] {
            background: rgba(100, 255, 218, 0.05);
            border-left: 3px solid #64ffda;
            border-radius: 0 8px 8px 0;
        }

        /* Warning boxes */
        [data-testid="stWarning"] {
            background: rgba(255, 215, 0, 0.05);
            border-left: 3px solid #ffd700;
            border-radius: 0 8px 8px 0;
        }

        /* Radio buttons */
        [data-testid="stRadio"] > div {
            gap: 1rem;
        }

        /* Dataframe / tables */
        [data-testid="stDataFrame"] {
            border: 1px solid #233554;
            border-radius: 10px;
            overflow: hidden;
        }

        /* Captions */
        [data-testid="stCaptionContainer"] p {
            color: #8892b0;
            font-size: 0.8rem;
        }

        /* Spinner */
        [data-testid="stSpinner"] {
            color: #64ffda;
        }

        /* ============================================================
           SECTION DIVIDERS
        ============================================================ */
        hr {
            border: none;
            border-top: 1px solid #233554;
            margin: 1.5rem 0;
        }

        /* ============================================================
           SCROLLBAR
        ============================================================ */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }

        ::-webkit-scrollbar-track {
            background: #1a1a2e;
        }

        ::-webkit-scrollbar-thumb {
            background: #233554;
            border-radius: 3px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #64ffda;
        }

        /* ============================================================
           DETECTION TABLE
        ============================================================ */
        .detection-table {
            width: 100%;
            border-collapse: collapse;
            border-radius: 10px;
            overflow: hidden;
        }

        .detection-table th {
            background: #16213e;
            color: #64ffda;
            padding: 0.7rem 1rem;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-weight: 600;
        }

        .detection-table td {
            padding: 0.6rem 1rem;
            border-bottom: 1px solid #233554;
            color: #a8b2d1;
            font-size: 0.9rem;
        }

        .detection-table tr:last-child td {
            border-bottom: none;
        }

        .detection-table tr:hover td {
            background: rgba(100, 255, 218, 0.03);
        }

        /* ============================================================
           FOOTER
        ============================================================ */
        footer {
            visibility: hidden;
        }

        .footer-credit {
            text-align: center;
            color: #4a5568;
            font-size: 0.78rem;
            padding: 2rem 0 1rem;
            border-top: 1px solid #1a1a2e;
            margin-top: 3rem;
            letter-spacing: 0.03em;
        }

        /* Fix 3: Main content area CTA button contrast and color override */
        section.main div[data-testid="stButton"] > button[kind="primary"] {
            background: linear-gradient(135deg, #e94560, #c23152) !important;
            background-image: linear-gradient(135deg, #e94560, #c23152) !important;
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.85rem 2rem !important;
            font-weight: 700 !important;
            font-size: 1.05rem !important;
            box-shadow: 0 4px 16px rgba(233, 69, 96, 0.35) !important;
        }

        section.main div[data-testid="stButton"] > button[kind="primary"] p,
        section.main div[data-testid="stButton"] > button[kind="primary"] div {
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
        }

        section.main div[data-testid="stButton"] > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #f05a77, #d43b60) !important;
            background-image: linear-gradient(135deg, #f05a77, #d43b60) !important;
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(233, 69, 96, 0.45) !important;
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # PART 2b — Sidebar navigation (rendered on ALL pages)
    with st.sidebar:
        # CHANGE 5 — Logo + Navigation + Status only in sidebar
        st.markdown("## 🩺 RaphaID AI")
        st.caption("AI-Powered Clinical Assistant")
        st.markdown("---")

        # CHANGE 2 — Remove reports page from navigation options
        nav_options = {
            "dashboard": "📊 Clinical Dashboard",
            "diagnosis": "⚕️ New Diagnosis",
            "settings": "⚙️ Settings",
        }

        for key, label in nav_options.items():
            is_active = st.session_state["current_page"] == key
            button_type = "primary" if is_active else "secondary"
            if st.button(label, key=f"nav_{key}",
                        use_container_width=True, type=button_type):
                st.session_state["current_page"] = key
                st.rerun()

        st.markdown("---")

        # CHANGE 3 — System Status instead of metrics
        st.markdown("### 🟢 System Status")
        status_items = [
            "Model Loaded",
            "CPU Ready",
            "WHO Rules Active",
            "Slide Quality Enabled",
            "Human Verification Enabled",
            "PDF Reporting Ready",
        ]
        for item in status_items:
            st.markdown(
                f"<div style='font-size: 0.82rem; color: #a8b2d1; "
                f"padding: 3px 0;'>🟢 {item}</div>", 
                unsafe_allow_html=True
            )
        st.markdown("---")
    # PART 2c — Page routing
    if st.session_state["current_page"] == "dashboard":
        render_dashboard()

    elif st.session_state["current_page"] == "diagnosis":
        # --- Header ---
        st.markdown("""
        <div class="main-header">
            <span style="background-color: #e94560; color: white; padding: 0.2rem 0.6rem;
                         border-radius: 20px; font-size: 0.75rem; font-weight: 600;
                         letter-spacing: 0.05em; text-transform: uppercase;
                         margin-bottom: 0.5rem; display: inline-block;">
                AI Clinical Assistant
            </span>
            <h1>🔬 RaphaID AI</h1>
            <p>Detects and classifies <em>malaria</em> parasite stages from microscopy images in seconds</p>
        </div>
        """, unsafe_allow_html=True)

        # CHANGE 2 — Value proposition strip
        vp_col1, vp_col2, vp_col3, vp_col4 = st.columns(4)

        with vp_col1:
            st.markdown("""
            <div style="text-align:center; padding:0.8rem; 
                        background:linear-gradient(135deg,#1a1a2e,#16213e); 
                        border:1px solid #233554; border-radius:10px; 
                        min-height: 110px;
                        display:flex; flex-direction:column; 
                        justify-content:center;">
                <div style="font-size:1.4rem;">🦠</div>
                <div style="font-size:0.95rem; font-weight:700; 
                            color:#64ffda; margin-top:0.3rem;">
                    Parasite Stage Detection
                </div>
                <div style="font-size:0.72rem; color:#8892b0; margin-top:0.2rem;">
                    Ring, Trophozoite, Schizont, Gametocyte
                </div>
            </div>
            """, unsafe_allow_html=True)

        with vp_col2:
            st.markdown("""
            <div style="text-align:center; padding:0.8rem; 
                        background:linear-gradient(135deg,#1a1a2e,#16213e); 
                        border:1px solid #233554; border-radius:10px; 
                        min-height: 110px;
                        display:flex; flex-direction:column; 
                        justify-content:center;">
                <div style="font-size:1.4rem;">✔</div>
                <div style="font-size:0.95rem; font-weight:700; 
                            color:#64ffda; margin-top:0.3rem;">
                    CPU Optimized
                </div>
                <div style="font-size:0.72rem; color:#8892b0; margin-top:0.2rem;">
                    No GPU required
                </div>
            </div>
            """, unsafe_allow_html=True)

        with vp_col3:
            st.markdown("""
            <div style="text-align:center; padding:0.8rem; 
                        background:linear-gradient(135deg,#1a1a2e,#16213e); 
                        border:1px solid #233554; border-radius:10px; 
                        min-height: 110px;
                        display:flex; flex-direction:column; 
                        justify-content:center;">
                <div style="font-size:1.4rem;">🏥</div>
                <div style="font-size:0.95rem; font-weight:700; 
                            color:#64ffda; margin-top:0.3rem;">
                    WHO Classification
                </div>
                <div style="font-size:0.72rem; color:#8892b0; margin-top:0.2rem;">
                    Built-in severity guidance
                </div>
            </div>
            """, unsafe_allow_html=True)


        with vp_col4:
            st.markdown("""
            <div style="text-align:center; padding:0.8rem; 
                        background:linear-gradient(135deg,#1a1a2e,#16213e); 
                        border:1px solid #233554; border-radius:10px; 
                        min-height: 110px;
                        display:flex; flex-direction:column; 
                        justify-content:center;">
                <div style="font-size:1.4rem;">📋</div>
                <div style="font-size:0.95rem; font-weight:700; 
                            color:#64ffda; margin-top:0.3rem;">
                    Clinical Reports
                </div>
                <div style="font-size:0.72rem; color:#8892b0; margin-top:0.2rem;">
                    PDF, CSV and annotated image
                </div>
            </div>
            """, unsafe_allow_html=True)

        # CHANGE 2a — Spacing after value proposition strip
        st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
        st.markdown("---")

        mode = st.radio(
            "Analysis Mode",
            ["Single Image", "Batch (Multiple Slides)"],
            horizontal=True,
        )
        # CHANGE 2b — Spacing after mode radio toggle
        st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)

        # ===================================================================
        # SINGLE IMAGE MODE
        # ===================================================================
        if mode == "Single Image":  # CHANGED: Wrap existing flow in mode check

            # CHANGE 2 — Patient intake form
            with st.expander("🏥 Patient Information", expanded=True):
                st.caption(
                    "Details entered here are stored temporarily for this session only. "
                    "They are never saved to any server or database. All fields are optional."
                )

                pt_col1, pt_col2, pt_col3 = st.columns(3)

                with pt_col1:
                    p_name = st.text_input(
                        "Patient Name",
                        value=st.session_state["patient_details"]["name"],
                        placeholder="e.g. Femi Okoro",
                        key="pt_name",
                    )
                    st.session_state["patient_details"]["name"] = p_name

                    p_age = st.number_input(
                        "Age (years)",
                        min_value=0,
                        max_value=120,
                        value=st.session_state["patient_details"]["age"]
                              if st.session_state["patient_details"]["age"] else 0,
                        step=1,
                        key="pt_age",
                        help="Enter 0 if age is unknown",
                    )
                    # Store None if age is 0 (unknown), otherwise store value
                    st.session_state["patient_details"]["age"] = (
                        p_age if p_age > 0 else None
                    )

                with pt_col2:
                    p_sex = st.selectbox(
                        "Sex (optional)",
                        options=["", "Male", "Female", "Other / Prefer not to say"],
                        index=["", "Male", "Female",
                               "Other / Prefer not to say"].index(
                            st.session_state["patient_details"]["sex"]
                        ) if st.session_state["patient_details"]["sex"] in
                           ["", "Male", "Female",
                            "Other / Prefer not to say"] else 0,
                        key="pt_sex",
                    )
                    st.session_state["patient_details"]["sex"] = p_sex

                    p_id = st.text_input(
                        "Patient / Sample ID",
                        value=st.session_state["patient_details"]["patient_id"],
                        placeholder="e.g. LAB-2026-00142",
                        key="pt_id",
                        help="Hospital or laboratory identifier used in report filename",
                    )
                    st.session_state["patient_details"]["patient_id"] = p_id

                with pt_col3:
                    p_clinician = st.text_input(
                        "Requesting Clinician",
                        value=st.session_state["patient_details"]["clinician"],
                        placeholder="e.g. Dr. O.I Olayemi",
                        key="pt_clinician",
                    )
                    st.session_state["patient_details"]["clinician"] = p_clinician

                    p_facility = st.text_input(
                        "Health Facility",
                        value=st.session_state["patient_details"]["facility"],
                        placeholder="e.g. University College Hospital Ibadan",
                        key="pt_facility",
                    )
                    st.session_state["patient_details"]["facility"] = p_facility

                p_notes = st.text_area(
                    "Clinical Notes (Optional)",
                    value=st.session_state["patient_details"]["notes"],
                    placeholder="e.g. Fever for 3 days, suspected malaria, "
                                "prior ACT treatment 6 months ago...",
                    height=80,
                    max_chars=250,
                    key="pt_notes",
                    help="Brief clinical history or presenting complaints. Max 250 characters.",
                )
                st.session_state["patient_details"]["notes"] = p_notes

                # Auto-generated metadata display
                meta = st.session_state["report_meta"]
                meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
                with meta_col1:
                    st.markdown(f"""
                    <div style="font-size:0.75rem; color:#8892b0;">Today's Date</div>
                    <div style="font-size:0.85rem; color:#64ffda;
                                font-weight:600;">{meta['date']}</div>
                    """, unsafe_allow_html=True)
                with meta_col2:
                    st.markdown(f"""
                    <div style="font-size:0.75rem; color:#8892b0;">Study ID</div>
                    <div style="font-size:0.85rem; color:#64ffda;
                                font-weight:600;">{meta['study_id']}</div>
                    """, unsafe_allow_html=True)
                with meta_col3:
                    st.markdown(f"""
                    <div style="font-size:0.75rem; color:#8892b0;">Report Number</div>
                    <div style="font-size:0.85rem; color:#64ffda;
                                font-weight:600;">{meta['report_number']}</div>
                    """, unsafe_allow_html=True)
                _, _, meta_col4 = st.columns([3, 3, 1])
                with meta_col4:
                    if st.button("🗑 Clear", key="clear_patient", 
                                help="Clear all patient details"):
                        st.session_state["patient_details"] = {
                            "name": "", "age": None, "sex": "",
                            "patient_id": "", "clinician": "",
                            "facility": "", "notes": "",
                        }
                        st.rerun()

                st.markdown("<br>", unsafe_allow_html=True)

                # Compact patient summary — shows automatically as fields are filled
                details = st.session_state["patient_details"]
                if any([details["name"], details["patient_id"],
                        details["clinician"], details["facility"]]):
                    summary_parts = []
                    if details["name"]:
                        summary_parts.append(f"**{details['name']}**")
                    if details["age"]:
                        summary_parts.append(f"Age {details['age']}")
                    if details["sex"]:
                        summary_parts.append(details["sex"])
                    if details["patient_id"]:
                        summary_parts.append(f"ID: `{details['patient_id']}`")
                    if details["clinician"]:
                        summary_parts.append(f"Ref: {details['clinician']}")

                    # Status will be filled after inference — placeholder for now
                    st.info(
                        "🏥 " + " · ".join(summary_parts) +
                        f" · Report: {meta['report_number']}"
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
                st.session_state["gallery_expanded"] = False  # CHANGED: Reset gallery toggle on new upload

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
            with st.expander("🤔 How It Works: ", expanded=False):
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
                    image_bgr = _downscale_if_needed(image_bgr)  # cap at 1600px — prevents OOM on large uploads
                    image_name = uploaded_file.name
                else:
                    sample_path = PROJECT_ROOT / st.session_state["sample_image"]
                    image_bgr = cv2.imread(str(sample_path))
                    image_name = Path(sample_path).name

                if image_bgr is None:
                    st.error("❌ Could not decode the uploaded image. Please try a different file.")
                    return

                # FEATURE 1 — Slide quality assessment
                quality = assess_slide_quality(image_bgr)
                st.session_state['last_quality'] = quality['metrics']

                if quality['has_warnings']:
                    for issue in quality['issues']:
                        if issue['severity'] == 'error':
                            err_label = {
                                'blur': 'Focus Problem',
                                'dark': 'Illumination Too Low',
                                'bright': 'Overexposure Detected'
                            }.get(issue['type'], 'Quality Issue')
                            st.markdown(f"""
                            <div style="
                                background: rgba(248,113,113,0.08);
                                border: 1px solid rgba(248,113,113,0.4);
                                border-left: 4px solid #f87171;
                                border-radius: 10px;
                                padding: 14px 16px;
                                margin-bottom: 10px;
                            ">
                                <div style="font-weight:600; color:#f87171; 
                                            margin-bottom:6px; font-size:14px;">
                                    ⚠ Slide Quality Error — {err_label}
                                </div>
                                <div style="color:#e2e8f0; font-size:13.5px; 
                                            line-height:1.6;">
                                    {issue['message']}
                                </div>
                                <div style="color:#9ca3af; font-size:12px; 
                                            margin-top:6px;">
                                    {issue['detail']}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            warn_label = {
                                'blur': 'Focus Advisory',
                                'dark': 'Illumination Advisory', 
                                'bright': 'Exposure Advisory'
                            }.get(issue['type'], 'Quality Advisory')
                            st.markdown(f"""
                            <div style="
                                background: rgba(255,215,0,0.07);
                                border: 1px solid rgba(255,215,0,0.35);
                                border-left: 4px solid #ffd700;
                                border-radius: 10px;
                                padding: 14px 16px;
                                margin-bottom: 10px;
                            ">
                                <div style="font-weight:600; color:#ffd700; 
                                            margin-bottom:6px; font-size:14px;">
                                    ⚡ Slide Quality Warning — {warn_label}
                                </div>
                                <div style="color:#e2e8f0; font-size:13.5px; 
                                            line-height:1.6;">
                                    {issue['message']}
                                </div>
                                <div style="color:#9ca3af; font-size:12px; 
                                            margin-top:6px;">
                                    {issue['detail']}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                # Hard block on errors — do not show Analyse Slide button
                if not quality['passed']:
                    st.info("""
    **Next steps:**
    1. Clean the microscope lens and slide
    2. Ensure the slide is perfectly in focus
    3. Check illumination levels
    4. Recapture the image and try again
    """)
                    st.stop()

                # Soft advisory on warnings — show button with a note
                if quality['has_warnings'] and quality['passed']:
                    st.caption(
                        "⚡ Quality warnings detected above. You may proceed, "
                        "but results should be interpreted with caution and "
                        "confirmed by microscopy."
                    )

                if quality['passed'] and not quality['has_warnings']:
                    st.success("✅ Slide quality checks passed. Image is sharp and well-illuminated.")

                # FEATURE 2 — Image hash change detection
                import hashlib
                img_hash = hashlib.md5(image_bgr.tobytes()).hexdigest()[:12]

                if st.session_state.get("last_img_hash") != img_hash:
                    st.session_state["verification_decisions"] = {}
                    st.session_state["last_img_hash"] = img_hash
                    st.session_state["verified_result"] = None
                    st.session_state["current_result"] = None  # BUGFIX: new image requires fresh Analyse Slide click

                # CHANGE 3 — Analyse Slide button
                st.markdown("<br>", unsafe_allow_html=True)
                _, btn_col, _ = st.columns([1, 2, 1])
                with btn_col:
                    run_scan = st.button(
                        "🔬 Analyse Slide",
                        use_container_width=True,
                        help="Run YOLOv8 malaria parasite detection on this image. "
                             "Adjust Detection Sensitivity in the sidebar first if needed.",
                        type="primary",
                    )

                # Trigger fresh inference only when the button is explicitly clicked
                # AND the image has changed (avoid re-running on every rerun)
                if run_scan:
                    # Load model (cached to avoid reloading on every interaction)
                    detector = load_model(weights_path)

                    if detector is None:
                        st.error(
                            f"❌ Model weights not found at `{weights_path}`. "
                            "Please train a model first or update the weights path."
                        )
                        return

                    with st.spinner("🔍 Analyzing blood smear ..."):
                        result = detector.predict(
                            image_bgr,
                            conf=conf_threshold,
                            iou=iou_threshold,
                            annotate=True,
                        )
                        if len(result.detections) == 0:
                            st.warning(
                                "⚠️ No cells or parasites detected. Please ensure you are "
                                "uploading a Giemsa-stained blood smear microscopy image."
                            )
                        elif result.total_rbc + result.total_parasites < 5:
                            st.warning(
                                "⚠️ Very few cells detected. Results may be unreliable. "
                                "For best results, use high-quality microscopy images at "
                                "100x oil immersion magnification."
                            ) 
                        result.image_path = image_name
                    st.session_state["current_result"] = result
                    st.session_state["current_image_bgr"] = image_bgr
                    st.session_state["current_image_name"] = image_name
                    # Reset verification state for the new analysis run
                    st.session_state["verification_decisions"] = {}
                    st.session_state["verified_result"] = None

                # Show results if we have a stored result for the current image.
                # This persists across reruns triggered by verification buttons,
                # solving the bug where accept/reject discarded all results.
                has_result = (
                    st.session_state.get("current_result") is not None
                    and st.session_state.get("current_image_name") == image_name
                )

                if not has_result:
                    st.markdown("""
                    <div style="text-align:center; padding:2rem; color:#8892b0;">
                        <p style="font-size:0.95rem;">
                            👆 Adjust <strong style="color:#64ffda;">Detection 
                            Sensitivity</strong> in the sidebar if needed, then 
                            click <strong style="color:#e94560;">Analyse Slide</strong> 
                            to begin.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.stop()

                # From here onward, use the STORED result, not a freshly computed one
                result = st.session_state["current_result"]
                image_bgr = st.session_state["current_image_bgr"]
                image_name = st.session_state["current_image_name"]

                # CHANGED: Count uncertainty tiers 
                uncertain_count, confident_parasite_count = _count_tiers(result)

                # --- Results Layout ---
                st.markdown("---")

                # CHANGED: Uncertainty warning banner (above results)
                if uncertain_count > 0:
                    st.warning(
                        f"⚠️ {uncertain_count} detection(s) require human verification "
                        "due to low model confidence. These are highlighted in yellow."
                    )

                # Metrics row
                # CHANGE 2c — Metric columns gap
                col1, col2, col3, col4 = st.columns(4, gap="small")
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
                # CHANGE 2d — Image comparison columns gap
                img_col1, img_col2 = st.columns(2, gap="medium")

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

                # CHANGED: Clinical Summary Card (Single Image mode only)
                st.markdown("---")
                st.markdown("### 🩺 AI Screening Summary")

                # 1. Parasites detected count
                # 2. Dominant stage determination
                p_counts = {}
                for d in result.detections:
                    if d.class_name in PARASITE_CLASSES:
                        p_counts[d.class_name] = p_counts.get(d.class_name, 0) + 1

                dominant_stage = "None"
                if p_counts:
                    dominant_stage = max(p_counts, key=p_counts.get).replace("_", " ").title()

                # 3 & 4. Estimated parasitemia & Severity
                severity_str = classify_severity(result.parasitemia_pct)

                # 5. Recommendation
                if uncertain_count > 0:
                    rec_str = f"Microscopy review strongly advised: {uncertain_count} detection(s) flagged for human verification."
                    rec_style = "background-color: #e94560; color: #1a1a2e; border-left: 5px solid #d4af37;"
                    rec_icon = "⚠️"
                elif result.parasitemia_pct == 0:
                    rec_str = "No parasites detected. Routine confirmation recommended per standard clinical protocol."
                    rec_style = "background-color: #1f3a2b; color: #64ffda; border-left: 5px solid #64ffda;"
                    rec_icon = "✅"
                elif result.parasitemia_pct < 1:
                    rec_str = "Low parasitemia detected. Microscopy review advised before final diagnosis."
                    rec_style = "background-color: #3b3a1a; color: #ffeb3b; border-left: 5px solid #ffd214;"
                    rec_icon = "🟡"
                elif result.parasitemia_pct < 5:
                    rec_str = "Moderate parasitemia detected. Microscopy review advised before final diagnosis."
                    rec_style = "background-color: #3a2510; color: #ff9800; border-left: 5px solid #ff9800;"
                    rec_icon = "🟠"
                else:
                    rec_str = "Severe parasitemia detected. Immediate microscopy confirmation and clinical correlation advised."
                    rec_style = "background-color: #421818; color: #ff4d4d; border-left: 5px solid #ff4d4d;"
                    rec_icon = "🚨"

                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1a1a2e, #16213e); border: 1px solid #233554; border-radius: 12px; padding: 1.5rem; color: white;">
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1.5rem;">
                        <div>
                            <div style="font-size: 0.85rem; color: #8892b0;">Parasites Detected</div>
                            <div style="font-size: 1.8rem; font-weight: 700; color: #ff4d4d;">{result.total_parasites}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.85rem; color: #8892b0;">Dominant Stage</div>
                            <div style="font-size: 1.8rem; font-weight: 700; color: #64ffda;">{dominant_stage}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.85rem; color: #8892b0;">Estimated Parasitemia</div>
                            <div style="font-size: 1.8rem; font-weight: 700; color: #64ffda;">{result.parasitemia_pct:.2f}%</div>
                        </div>
                        <div>
                            <div style="font-size: 0.85rem; color: #8892b0;">Severity</div>
                            <div style="font-size: 1.2rem; font-weight: 700; margin-top: 0.5rem;">{severity_str}</div>
                        </div>
                    </div>
                    <div style="{rec_style} padding: 1rem; border-radius: 8px; font-weight: 600; display: flex; align-items: center; gap: 0.8rem;">
                        <span style="font-size: 1.5rem;">{rec_icon}</span>
                        <span>{rec_str}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # FEATURE 2 — Verified panel for metrics card override if decisions have been made
                # We calculate verified metrics here so that the main Clinical Summary Card displays them.
                verified_data = st.session_state.get("verified_result")
                if verified_data and (verified_data['accepted_uncertain'] + verified_data['rejected_uncertain']) > 0:
                    # Recalculate dominant stage using clinician decisions
                    vp_counts = {}
                    for det_i, d in enumerate(result.detections):
                        if d.class_name in PARASITE_CLASSES:
                            is_uncertain = UNCERTAINTY_THRESHOLD_LOW <= d.confidence <= UNCERTAINTY_THRESHOLD_HIGH
                            if is_uncertain:
                                key = str(det_i)
                                if st.session_state["verification_decisions"].get(key, True):
                                    vp_counts[d.class_name] = vp_counts.get(d.class_name, 0) + 1
                            else:
                                vp_counts[d.class_name] = vp_counts.get(d.class_name, 0) + 1
                    v_dominant_stage = "None"
                    if vp_counts:
                        v_dominant_stage = max(vp_counts, key=vp_counts.get).replace("_", " ").title()

                    v_severity_str = classify_severity(verified_data['verified_parasitemia_pct'])

                    # Update recommendation for summary card
                    if verified_data['pending_uncertain'] > 0:
                        v_rec_str = f"Microscopy review strongly advised — {verified_data['pending_uncertain']} detection(s) pending verification."
                        v_rec_style = "background-color: #ffd214; color: #1a1a2e; border-left: 5px solid #d4af37;"
                        v_rec_icon = "⚠️"
                    elif verified_data['verified_parasitemia_pct'] == 0:
                        v_rec_str = "No parasites detected. Routine confirmation recommended per standard clinical protocol."
                        v_rec_style = "background-color: #1f3a2b; color: #64ffda; border-left: 5px solid #64ffda;"
                        v_rec_icon = "✅"
                    elif verified_data['verified_parasitemia_pct'] < 1:
                        v_rec_str = "Low parasitemia detected. Microscopy review advised before final diagnosis."
                        v_rec_style = "background-color: #3b3a1a; color: #ffeb3b; border-left: 5px solid #ffd214;"
                        v_rec_icon = "🟡"
                    elif verified_data['verified_parasitemia_pct'] < 5:
                        v_rec_str = "Moderate parasitemia detected. Microscopy review advised before final diagnosis."
                        v_rec_style = "background-color: #3a2510; color: #ff9800; border-left: 5px solid #ff9800;"
                        v_rec_icon = "🟠"
                    else:
                        v_rec_str = "Severe parasitemia detected. Immediate microscopy confirmation and clinical correlation advised."
                        v_rec_style = "background-color: #421818; color: #ff4d4d; border-left: 5px solid #ff4d4d;"
                        v_rec_icon = "🚨"

                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #0f2027, #1a2a1a); border: 1px solid #4ade80; border-radius: 12px; padding: 1.5rem; color: white; margin-top: 1rem;">
                        <div style="font-size: 11px; font-weight: 600; color: #4ade80; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 1rem;">
                            ✓ Clinician-Verified Screening Summary
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1.5rem;">
                            <div>
                                <div style="font-size: 0.85rem; color: #8892b0;">Verified Parasites</div>
                                <div style="font-size: 1.8rem; font-weight: 700; color: #ff4d4d;">{verified_data['verified_parasites']}</div>
                            </div>
                            <div>
                                <div style="font-size: 0.85rem; color: #8892b0;">Verified Dominant Stage</div>
                                <div style="font-size: 1.8rem; font-weight: 700; color: #64ffda;">{v_dominant_stage}</div>
                            </div>
                            <div>
                                <div style="font-size: 0.85rem; color: #8892b0;">Verified Parasitemia</div>
                                <div style="font-size: 1.8rem; font-weight: 700; color: #64ffda;">{verified_data['verified_parasitemia_pct']:.2f}%</div>
                            </div>
                            <div>
                                <div style="font-size: 0.85rem; color: #8892b0;">Verified Severity</div>
                                <div style="font-size: 1.2rem; font-weight: 700; margin-top: 0.5rem;">{v_severity_str}</div>
                            </div>
                        </div>
                        <div style="{v_rec_style} padding: 1rem; border-radius: 8px; font-weight: 600; display: flex; align-items: center; gap: 0.8rem;">
                            <span style="font-size: 1.5rem;">{v_rec_icon}</span>
                            <span>{v_rec_str}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                # CHANGED: Inference timing caption (Single Image mode)
                st.caption(f"⏱️ Inference completed in {result.inference_time_sec:.2f} seconds (CPU)")

                # --- INSERTION POINT: _render_detailed_analysis call ---
                _render_detailed_analysis(result, uncertain_count)

                # --- CHANGED: Detection Crop Gallery (Single Image mode only) ---
                _render_detection_gallery(image_bgr, result)

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
                        is_parasite = " Yes" if cls_name in PARASITE_CLASSES else "-"
                        table_data.append({
                            "Class": cls_name.replace("_", " ").title(),
                            "Count": count,
                            "Avg Confidence": f"{avg_conf:.3f}",
                            "Parasite?": is_parasite,
                        })

                    st.table(table_data)
                else:
                    st.info("No objects detected at the current confidence threshold.")



                # ── FEATURE 2 — Human-in-the-loop verification ───────────────────────────
                if uncertain_count > 0:
                    st.markdown("---")
                    st.markdown("### 👤 Clinician Verification Required")
                    st.caption(
                        f"{uncertain_count} detection(s) have confidence between "
                        f"35% and 45% and require your review. Accept or reject "
                        f"each detection below. Parasitemia and severity will "
                        f"recalculate immediately based on your decisions."
                    )

                    # Collect uncertain detections with their indices
                    uncertain_dets = [
                        (i, det) for i, det in enumerate(result.detections)
                        if (det.class_name in PARASITE_CLASSES and
                            UNCERTAINTY_THRESHOLD_LOW 
                            <= det.confidence 
                            <= UNCERTAINTY_THRESHOLD_HIGH)
                    ]

                    img_h, img_w = image_bgr.shape[:2]

                    # Render each uncertain detection as a verification card
                    for idx, (det_i, det) in enumerate(uncertain_dets):
                        key = str(det_i)
                        current = st.session_state["verification_decisions"].get(key)

                        # Crop the detection from the original un-annotated image
                        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
                        bw, bh = x2 - x1, y2 - y1
                        pad_x = max(8, int(bw * 0.2))
                        pad_y = max(8, int(bh * 0.2))
                        cx1 = max(0, x1 - pad_x)
                        cy1 = max(0, y1 - pad_y)
                        cx2 = min(img_w, x2 + pad_x)
                        cy2 = min(img_h, y2 + pad_y)
                        crop = image_bgr[cy1:cy2, cx1:cx2]

                        if crop.size == 0:
                            continue

                        crop_display = cv2.resize(
                            crop, (200, 200), interpolation=cv2.INTER_CUBIC
                        )
                        crop_rgb = cv2.cvtColor(crop_display, cv2.COLOR_BGR2RGB)

                        # Card border color reflects decision state
                        if current is None:
                            border_color = "#ffd700"
                            status_label = "⏳ Pending review"
                            status_color = "#ffd700"
                        elif current:
                            border_color = "#4ade80"
                            status_label = "✓ Accepted as parasite"
                            status_color = "#4ade80"
                        else:
                            border_color = "#f87171"
                            status_label = "✗ Rejected as false positive"
                            status_color = "#f87171"

                        st.markdown(f"""
                        <div style="
                            border: 1px solid {border_color};
                            border-radius: 12px;
                            padding: 16px;
                            margin-bottom: 12px;
                            background: rgba(26,26,46,0.6);
                        ">
                            <div style="font-weight:600; color:#e2e8f0; 
                                        margin-bottom:4px; font-size:14px;">
                                Detection {idx + 1} of {len(uncertain_dets)} — 
                                {det.class_name.replace('_',' ').title()}
                            </div>
                            <div style="font-size:12px; color:#8892b0; 
                                        margin-bottom:12px;">
                                Confidence: {det.confidence:.1%} · 
                                Position: ({x1}, {y1}) — ({x2}, {y2}) · 
                                <span style="color:{status_color};">
                                    {status_label}
                                </span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        v_col1, v_col2, v_col3 = st.columns([1, 1, 2])

                        with v_col1:
                            st.image(crop_rgb, caption="Zoomed crop", 
                                     use_container_width=True)

                        with v_col2:
                            accept_key = f"accept_{det_i}_{img_hash}"
                            reject_key = f"reject_{det_i}_{img_hash}"

                            if st.button(
                                "✓ Accept",
                                key=accept_key,
                                help="Confirm this is a real parasite. "
                                     "It will be included in the parasite count.",
                                use_container_width=True,
                            ):
                                st.session_state["verification_decisions"][key] = True
                                # Recalculate verified metrics immediately
                                st.session_state["verified_result"] = compute_verified_parasitemia(
                                    result,
                                    st.session_state["verification_decisions"]
                                )
                                st.rerun()

                            if st.button(
                                "✗ Reject",
                                key=reject_key,
                                help="Mark this as a false positive. "
                                     "It will be excluded from the parasite count.",
                                use_container_width=True,
                            ):
                                st.session_state["verification_decisions"][key] = False
                                # Recalculate verified metrics immediately
                                st.session_state["verified_result"] = compute_verified_parasitemia(
                                    result,
                                    st.session_state["verification_decisions"]
                                )
                                st.rerun()

                        with v_col3:
                            # Clinical context for this stage
                            stage_context = {
                                "ring": (
                                    "Ring stage (early trophozoite). "
                                    "Appears as a thin ring or disc shape inside "
                                    "the RBC. Pale centre, thin cytoplasm rim. "
                                    "Most common stage in P. vivax infection."
                                ),
                                "trophozoite": (
                                    "Trophozoite stage. "
                                    "Larger, irregular shape filling more of the "
                                    "RBC. Often shows amoeboid cytoplasm. "
                                    "Active feeding stage: indicates established "
                                    "infection."
                                ),
                                "schizont": (
                                    "Schizont stage. "
                                    "Multiple chromatin dots (merozoites) visible "
                                    "inside the RBC. Presence in peripheral blood "
                                    "may indicate higher disease severity."
                                ),
                                "gametocyte": (
                                    "Gametocyte stage. "
                                    "Oval to round shape, fills and may enlarge "
                                    "the RBC. The transmissible stage — patient "
                                    "may be infectious to mosquitoes."
                                ),
                            }.get(det.class_name, "")

                            if stage_context:
                                st.markdown(
                                    f"<div style='font-size:12.5px; "
                                    f"color:#a8b2d1; line-height:1.6;'>"
                                    f"{stage_context}</div>",
                                    unsafe_allow_html=True
                                )

                    # ── Verified results summary ──────────────────────────────
                    verified = compute_verified_parasitemia(
                        result,
                        st.session_state["verification_decisions"]
                    )

                    decisions_made = (
                        verified['accepted_uncertain'] + 
                        verified['rejected_uncertain']
                    )

                    st.markdown("---")

                    if verified['pending_uncertain'] > 0:
                        st.info(
                            f"⏳ {verified['pending_uncertain']} detection(s) not yet "
                            f"reviewed. Results below include unreviewed detections "
                            f"as accepted (conservative default). Review all "
                            f"detections above for a fully verified result."
                        )

                    if decisions_made > 0:
                        # Show verified results panel
                        v_severity = classify_severity(
                            verified['verified_parasitemia_pct']
                        )

                        st.markdown(f"""
                        <div style="
                            background: rgba(15, 32, 39, 0.6);
                            border: 1px solid #4ade80;
                            border-radius: 12px;
                            padding: 20px;
                            margin-top: 8px;
                        ">
                            <div style="font-size:13px; font-weight:600; 
                                        color:#4ade80; margin-bottom:16px; 
                                        text-transform:uppercase; letter-spacing:.06em;">
                                ✓ Clinician-Verified Result
                            </div>
                            <div style="display:grid; 
                                        grid-template-columns:repeat(3,1fr); 
                                        gap:16px; margin-bottom:16px;">
                                <div>
                                    <div style="font-size:11px; color:#6b7280; 
                                                text-transform:uppercase; 
                                                letter-spacing:.06em;">
                                        Verified Parasites
                                    </div>
                                    <div style="font-size:28px; font-weight:700; 
                                                color:#f87171; margin-top:4px;">
                                        {verified['verified_parasites']}
                                    </div>
                                </div>
                                <div>
                                    <div style="font-size:11px; color:#6b7280; 
                                                text-transform:uppercase; 
                                                letter-spacing:.06em;">
                                        Verified Parasitemia
                                    </div>
                                    <div style="font-size:28px; font-weight:700; 
                                                color:#64ffda; margin-top:4px;">
                                        {verified['verified_parasitemia_pct']:.2f}%
                                    </div>
                                </div>
                                <div>
                                    <div style="font-size:11px; color:#6b7280; 
                                                text-transform:uppercase; 
                                                letter-spacing:.06em;">
                                        Severity
                                    </div>
                                    <div style="font-size:16px; font-weight:700; 
                                                color:#e2e8f0; margin-top:8px;">
                                        {v_severity}
                                    </div>
                                </div>
                            </div>
                            <div style="font-size:12px; color:#6b7280; 
                                        border-top:1px solid #1f3a2b; 
                                        padding-top:12px;">
                                Accepted: {verified['accepted_uncertain']} · 
                                Rejected: {verified['rejected_uncertain']} · 
                                Pending: {verified['pending_uncertain']} · 
                                Verified at: 
                                {datetime.now().strftime('%H:%M:%S')}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Update PDF to use verified results when available
                        st.session_state["verified_result"] = verified

                    # Keep the educational expander below for reference
                    with st.expander("ℹ️ About uncertainty flagging:"):
                        st.markdown(
                            "Detections with confidence between **35%** and **45%** "
                            "are flagged for clinician review. This range represents "
                            "the model's zone of genuine uncertainty: confident "
                            "enough to surface but not confident enough to "
                            "auto-classify. Your accept/reject decisions directly "
                            "update the parasitemia calculation above."
                        )

                else:
                    # No uncertain detections — show passive expander
                    with st.expander("ℹ️ About uncertainty flagging"):
                        st.markdown(
                            "Detections with confidence between **35    %** and **45%** "
                            "are flagged for clinician review. All detections in "
                            "this analysis exceeded the 45% confidence threshold "
                            "and were auto-classified without requiring review."
                        )

                # CHANGE 1 — Relocated download section to results flow
                # --- Download section ---
                st.markdown("---")
                st.markdown("### 📥 Download Reports")

                # CHANGE 2e — Download buttons columns gap
                dl_col1, dl_col2, dl_col3 = st.columns(3, gap="small")

                with dl_col1:
                    # PDF Report
                    try:
                        # CHANGE 5 — Updated PDF call with patient details
                        # FEATURE 2 — Use clinician-verified result if available, otherwise use model result
                        verified_data = st.session_state.get("verified_result")

                        if verified_data and (
                            verified_data['accepted_uncertain'] +
                            verified_data['rejected_uncertain']
                        ) > 0:
                            # Clinician has reviewed — note this in the scan status
                            v_decisions = (
                                verified_data['accepted_uncertain'] +
                                verified_data['rejected_uncertain']
                            )
                            if verified_data['verified_parasites'] == 0:
                                scan_status = "Negative — Clinician-verified"
                            else:
                                scan_status = (
                                    f"Positive — {verified_data['verified_parasites']} "
                                    f"parasite(s) · Clinician-verified "
                                    f"({v_decisions} uncertain detections reviewed)"
                                )
                        else:
                            if result.total_parasites == 0:
                                scan_status = "Negative: No parasites detected"
                            elif uncertain_count > 0 and result.total_parasites == uncertain_count:
                                scan_status = "Needs Review: Uncertain detections only"
                            else:
                                scan_status = (
                                    f"Positive — {result.total_parasites} parasite(s) detected"
                                )

                        pdf_bytes = generate_pdf_report(
                            result,
                            result.annotated_image,
                            uncertain_count=uncertain_count,
                            patient_details=st.session_state.get("patient_details"),
                            report_meta=st.session_state.get("report_meta"),
                            scan_status=scan_status,
                            verified_data=st.session_state.get("verified_result"),  # FIX 5
                        )

                        # Use Patient ID for filename — more clinical, more private than name
                        patient_id_slug = (
                            st.session_state.get("patient_details", {})
                            .get("patient_id", "")
                            .replace(" ", "-")
                            .replace("/", "-")
                            .strip("-")
                        )
                        report_num = (
                            st.session_state.get("report_meta", {})
                            .get("report_number", "")
                            .replace("-", "")
                        )
                        if patient_id_slug:
                            pdf_filename = f"{patient_id_slug}_malaria_report.pdf"
                        elif report_num:
                            pdf_filename = f"{report_num}_malaria_report.pdf"
                        else:
                            pdf_filename = f"malaria_report_{Path(image_name).stem}.pdf"

                        st.download_button(
                            label="📄 Download PDF Report",
                            data=pdf_bytes,
                            file_name=pdf_filename,
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

        # ===================================================================
        # CHANGED: BATCH PROCESSING MODE
        # ===================================================================
        else:
            st.markdown("### 📦 Batch Processing")
            st.caption(
                "⏱️ Processing time scales with image count and size. "
                "Large batches may take several minutes on CPU."
            )

            batch_files = st.file_uploader(
                "Upload multiple blood smear images",
                type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
                accept_multiple_files=True,
                key="batch_uploader",  # CHANGED: unique key to avoid conflict with single uploader
            )

            if batch_files:
                # Load model (cached — same instance as single mode)
                detector = load_model(weights_path)

                if detector is None:
                    st.error(
                        f"❌ Model weights not found at `{weights_path}`. "
                        "Please train a model first or update the weights path."
                    )
                else:
                    progress_bar = st.progress(0, text="Processing images...")
                    batch_summary_rows = []
                    # Store annotated images keyed by patient_id for expanders
                    batch_annotated = {}
                    batch_results_map = {}

                    for i, uploaded in enumerate(batch_files):
                        # Decode image
                        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                        img_bgr = _downscale_if_needed(img_bgr)  # cap at 1600px — prevents OOM on large uploads
                        patient_id = Path(uploaded.name).stem

                        if img_bgr is None:
                            batch_summary_rows.append({
                                "Patient ID": patient_id,
                                "Parasites Detected": "ERROR",
                                "Uncertain Detections": "—",
                                "Parasitemia %": "—",
                                "Status": "❌ Decode Error",
                            })
                            progress_bar.progress(
                                (i + 1) / len(batch_files),
                                text=f"Processing {i + 1}/{len(batch_files)}...",
                            )
                            continue

                        # Run inference (reuses cached model)
                        res = detector.predict(
                            img_bgr,
                            conf=conf_threshold,
                            iou=iou_threshold,
                            annotate=True,
                        )
                        res.image_path = uploaded.name

                        # CHANGED: Count tiers for this image
                        unc_count, conf_count = _count_tiers(res)

                        # Determine status based on confident parasites only
                        status = "🦠 Positive" if conf_count > 0 else "✅ Negative"
                        # If no confident parasites but uncertain exist, flag as needs review
                        if conf_count == 0 and unc_count > 0:
                            status = "⚠️ Needs Review"

                        batch_summary_rows.append({
                            "Patient ID": patient_id,
                            "Parasites Detected": conf_count,
                            "Uncertain Detections": unc_count,
                            "Parasitemia %": f"{res.parasitemia_pct:.2f}",
                            "Status": status,
                        })

                        # Store annotated image for drill-down
                        if not show_rbc:
                            batch_annotated[patient_id] = _draw_filtered(img_bgr, res)
                        else:
                            batch_annotated[patient_id] = (
                                res.annotated_image if res.annotated_image is not None
                                else img_bgr
                            )
                        batch_results_map[patient_id] = res

                        progress_bar.progress(
                            (i + 1) / len(batch_files),
                            text=f"Processing {i + 1}/{len(batch_files)}...",
                        )

                    progress_bar.empty()
                    st.success(f"✅ Processed {len(batch_files)} image(s).")

                    # --- Summary table ---
                    st.markdown("### 📊 Batch Summary")
                    df = pd.DataFrame(batch_summary_rows)
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    # --- CSV download ---
                    csv_bytes = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Batch Summary (CSV)",
                        data=csv_bytes,
                        file_name="batch_summary.csv",
                        mime="text/csv",
                    )

                    # --- PDF Reports for each image ---
                    st.markdown("### 📄 Individual PDF Reports")
                    pdf_cols = st.columns(3)
                    for idx, (pid, res) in enumerate(batch_results_map.items()):
                        col_idx = idx % 3
                        with pdf_cols[col_idx]:
                            try:
                                # Generate PDF for this image
                                unc_count, _ = _count_tiers(res)
                                if res.total_parasites == 0:
                                    scan_status = "Negative: No parasites detected"
                                elif unc_count > 0 and res.total_parasites == unc_count:
                                    scan_status = "Needs Review: Uncertain detections only"
                                else:
                                    scan_status = f"Positive — {res.total_parasites} parasite(s) detected"

                                pdf_bytes = generate_pdf_report(
                                    res,
                                    batch_annotated.get(pid),
                                    uncertain_count=unc_count,
                                    patient_details=st.session_state.get("patient_details"),
                                    report_meta=st.session_state.get("report_meta"),
                                    scan_status=scan_status,
                                    verified_data=None,
                                )
                                pdf_filename = f"{pid}_malaria_report.pdf"
                                st.download_button(
                                    label=f"📄 {pid}",
                                    data=pdf_bytes,
                                    file_name=pdf_filename,
                                    mime="application/pdf",
                                    key=f"pdf_{pid}",
                                )
                            except ImportError:
                                st.warning("Install `fpdf2` for PDF generation: `pip install fpdf2`")
                                break
                            except Exception as e:
                                st.error(f"PDF error for {pid}: {e}")

                    # --- Annotated Images Download ---
                    st.markdown("### 🖼️ Annotated Images")
                    img_cols = st.columns(3)
                    for idx, (pid, ann_img) in enumerate(batch_annotated.items()):
                        col_idx = idx % 3
                        with img_cols[col_idx]:
                            if ann_img is not None:
                                img_rgb = cv2.cvtColor(ann_img, cv2.COLOR_BGR2RGB)
                                pil_img = Image.fromarray(img_rgb)
                                buf = io.BytesIO()
                                pil_img.save(buf, format="PNG")
                                st.download_button(
                                    label=f"🖼️ {pid}",
                                    data=buf.getvalue(),
                                    file_name=f"annotated_{pid}.png",
                                    mime="image/png",
                                    key=f"img_{pid}",
                                )

                    # CHANGED: Uncertainty warning if any image has uncertain detections
                    total_uncertain = sum(
                        r.get("Uncertain Detections", 0)
                        for r in batch_summary_rows
                        if isinstance(r.get("Uncertain Detections"), int)
                    )
                    if total_uncertain > 0:
                        st.warning(
                            f"⚠️ {total_uncertain} total detection(s) across all slides "
                            "require human verification. Expand individual slides below "
                            "to review flagged regions."
                        )

                    # --- Per-image expanders ---
                    st.markdown("### 🔬 Individual Slide Details")
                    for row in batch_summary_rows:
                        pid = row["Patient ID"]
                        with st.expander(f"🔬 {pid} — {row['Status']}"):
                            ann_img = batch_annotated.get(pid)
                            if ann_img is not None:
                                ann_rgb = cv2.cvtColor(ann_img, cv2.COLOR_BGR2RGB)
                                st.image(ann_rgb, use_container_width=True)
                            else:
                                st.info("No annotated image available.")

                            res = batch_results_map.get(pid)
                            if res:
                                uc, cc = _count_tiers(res)
                                if uc > 0:
                                    st.warning(
                                        f"⚠️ {uc} detection(s) flagged as uncertain "
                                        "on this slide."
                                    )
                                st.caption(
                                    f"Parasitemia: {res.parasitemia_pct:.2f}% · "
                                    f"Total cells: {res.total_rbc + res.total_parasites} · "
                                    f"Parasites: {res.total_parasites}"
                                )
                                # CHANGED: Inference timing caption per-image in Batch mode
                                st.caption(f"⏱️ Inference completed in {res.inference_time_sec:.2f} seconds (CPU)")
                            # NOTE: Detection crop gallery could be added here per-expander
                            # in a future iteration, but is omitted to avoid performance
                            # issues with large batches.

                    # CHANGED: Uncertainty flagging expander (shared with batch mode)
                    with st.expander("ℹ️ About Uncertainty Flagging"):
                        st.markdown(
                            "Detections with confidence between **35%** and **45%** are "
                            "flagged for human review rather than auto-classified. This "
                            "respects clinical safety protocols — when the model is unsure, "
                            "a trained professional should verify the result rather than "
                            "relying on an automated label."
                        )


    elif st.session_state["current_page"] == "settings":
        st.markdown("### ⚙️ Settings")
        st.caption("Adjust AI model thresholds and configuration")
        st.markdown("---")
        
        st.session_state["conf_threshold"] = st.slider(
            "Detection Sensitivity",
            min_value=0.05,
            max_value=0.95,
            value=st.session_state["conf_threshold"],
            step=0.05,
            help="Minimum detection confidence. Lower = more detections (more false positives).",
        )

        st.session_state["iou_threshold"] = st.slider(
            "NMS Overlap Tolerance",
            min_value=0.1,
            max_value=0.9,
            value=st.session_state["iou_threshold"],
            step=0.05,
            help="Non-Maximum Suppression threshold. Lower = fewer overlapping boxes.",
        )

        st.session_state["show_rbc"] = st.checkbox(
            "Show Red Blood Cells",
            value=st.session_state["show_rbc"],
            help="Toggle visibility of healthy RBC detections (reduces visual clutter).",
        )
        
        st.markdown("---")
        with st.expander("⚙️ Advanced / Developer Settings", expanded=True):
            st.session_state["weights_path"] = st.text_input(
                "Model Weights Path",
                value=st.session_state["weights_path"],
                help="Path to trained YOLOv8 .pt file",
            )
            if 'last_quality' in st.session_state:
                q = st.session_state['last_quality']
                st.markdown("**Last slide quality metrics**")
                st.caption(f"Sharpness: {q['sharpness']} · "
                           f"Brightness: {q['brightness']}/255 · "
                           f"Saturation: {q['saturation_pct']}%")

    # CHANGE 2f — Footer team credit (only on dashboard)
    if st.session_state["current_page"] == "dashboard":
        st.markdown(
            "<div class='footer-credit'>"
            "Built by Team Devions &nbsp;·&nbsp; "
            "NACOS UI × DATICAN Competition 2026"
            "</div>",
            unsafe_allow_html=True
        )


# CHANGE 6
def _render_detailed_analysis(result: PredictionResult, uncertain_count: int) -> None:
    """Render Plotly charts for detailed confidence, stage breakdown, and parasitemia estimate."""
    import plotly.graph_objects as go

    with st.expander("📊 Detailed Analysis", expanded=False):
        # SECTION A — Two side-by-side charts
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Detection Confidence Distribution")
            high_conf = sum(
                1 for d in result.detections
                if d.class_name in PARASITE_CLASSES and d.confidence > UNCERTAINTY_THRESHOLD_HIGH
            )
            uncertain = sum(
                1 for d in result.detections
                if d.class_name in PARASITE_CLASSES and UNCERTAINTY_THRESHOLD_LOW <= d.confidence <= UNCERTAINTY_THRESHOLD_HIGH
            )
            
            x_vals = ["High Confidence Parasites", "Uncertain Parasites", "Red Blood Cells"]
            y_vals = [high_conf, uncertain, result.total_rbc]
            colors = ["#e94560", "#ffd700", "#64ffda"]

            fig1 = go.Figure(data=[
                go.Bar(
                    x=x_vals,
                    y=y_vals,
                    text=y_vals,
                    textposition="outside",
                    marker_color=colors
                )
            ])
            fig1.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(26,26,46,0.8)",
                font=dict(color="#8892b0"),
                xaxis=dict(gridcolor="#233554"),
                yaxis=dict(gridcolor="#233554"),
                height=300,
                margin=dict(t=30, b=20, l=10, r=10),
                showlegend=False
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.markdown("#### Parasite Stage Breakdown")
            stage_colors = {
                "ring": "#ff4d4d",
                "trophozoite": "#ff9800",
                "schizont": "#ffeb3b",
                "gametocyte": "#ff00ff"
            }
            # Compute counts from result.detections filtered to PARASITE_CLASSES
            stage_counts = {}
            for d in result.detections:
                if d.class_name in PARASITE_CLASSES:
                    stage_counts[d.class_name] = stage_counts.get(d.class_name, 0) + 1

            # Only include classes with count > 0
            stages_to_show = {k: v for k, v in stage_counts.items() if v > 0}

            if not stages_to_show:
                st.info("No parasites detected in this image.")
            else:
                y_labels = [cls.replace("_", " ").title() for cls in stages_to_show.keys()]
                x_vals2 = list(stages_to_show.values())
                colors2 = [stage_colors.get(cls, "#ffffff") for cls in stages_to_show.keys()]

                fig2 = go.Figure(data=[
                    go.Bar(
                        x=x_vals2,
                        y=y_labels,
                        orientation="h",
                        text=x_vals2,
                        textposition="outside",
                        marker_color=colors2
                    )
                ])
                fig2.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(26,26,46,0.8)",
                    font=dict(color="#8892b0"),
                    xaxis=dict(gridcolor="#233554"),
                    yaxis=dict(gridcolor="#233554"),
                    height=300,
                    margin=dict(t=30, b=20, l=10, r=10),
                    showlegend=False
                )
                st.plotly_chart(fig2, use_container_width=True)

        # SECTION B — Full-width parasitemia gauge below the two charts
        st.markdown("#### Parasitemia Estimate")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=result.parasitemia_pct,
            number={"suffix": "%", "font": {"color": "#64ffda", "size": 32}},
            delta={
                "reference": 5.0,
                "suffix": "% vs severe threshold",
                "increasing": {"color": "#e94560"},
                "decreasing": {"color": "#64ffda"}
            },
            gauge={
                "axis": {"range": [0, 20], "tickcolor": "#8892b0"},
                "bar": {"color": "#e94560"},
                "bgcolor": "rgba(26,26,46,0.8)",
                "bordercolor": "#233554",
                "steps": [
                    {"range": [0, 1], "color": "rgba(100,255,218,0.15)"},
                    {"range": [1, 5], "color": "rgba(255,152,0,0.15)"},
                    {"range": [5, 20], "color": "rgba(233,69,96,0.15)"}
                ],
                "threshold": {
                    "line": {"color": "#ffd700", "width": 3},
                    "thickness": 0.75,
                    "value": 5.0
                }
            },
            title={
                "text": "Parasitemia %\nWHO: <1% Low · 1–5% Moderate · >5% Severe",
                "font": {"color": "#64ffda"}
            }
        ))

        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            height=320,
            margin=dict(t=100, b=20, l=60, r=60),
            font=dict(color="#8892b0")
        )
        st.plotly_chart(fig_gauge, use_container_width=True)


# ---------------------------------------------------------------------------
# CHANGED: Detection Crop Gallery — zoomed close-ups for interpretability
# ---------------------------------------------------------------------------
def _render_detection_gallery(
    image_bgr: np.ndarray,
    result: PredictionResult,
) -> None:
    """Render a gallery of zoomed detection crops (Single Image mode only).

    Shows each non-RBC detection as a 150×150 padded crop from the ORIGINAL
    (un-annotated) image, sorted with uncertain detections first.
    """
    non_rbc_dets = [
        d for d in result.detections if d.class_name in PARASITE_CLASSES
    ]

    if not non_rbc_dets:
        st.info(
            "No parasites detected in this image — "
            "showing healthy blood cells only."
        )
        return

    st.markdown("---")
    st.markdown("### 🔬 Detection Close-ups")
    st.caption("Zoomed crops of each detected parasite for visual verification")

    img_h, img_w = image_bgr.shape[:2]
    gallery_items = []

    for det in non_rbc_dets:
        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
        box_w = x2 - x1
        box_h = y2 - y1

        # Skip degenerate boxes
        if box_w <= 0 or box_h <= 0:
            continue

        # 10% padding on each side, clamped to image bounds
        pad_x = max(1, int(box_w * 0.1))
        pad_y = max(1, int(box_h * 0.1))
        crop_x1 = max(0, x1 - pad_x)
        crop_y1 = max(0, y1 - pad_y)
        crop_x2 = min(img_w, x2 + pad_x)
        crop_y2 = min(img_h, y2 + pad_y)

        crop = image_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            continue  # Edge case: completely out-of-bounds

        crop_resized = cv2.resize(crop, (150, 150), interpolation=cv2.INTER_CUBIC)
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)

        uncertain = _is_uncertain(det.confidence)

        gallery_items.append({
            "image": crop_rgb,
            "class_name": det.class_name,
            "confidence": det.confidence,
            "is_uncertain": uncertain,
        })

    if not gallery_items:
        return  # All boxes were degenerate

    # Sort: uncertain first (most clinically important), then confident descending
    gallery_items.sort(key=lambda x: (not x["is_uncertain"], -x["confidence"]))

    # CHANGED: Split into initial (first 12) and remaining for Show More toggle
    max_initial = 12
    initial_items = gallery_items[:max_initial]
    remaining_items = gallery_items[max_initial:]

    # --- Helper: render a list of gallery items in a 4-column grid ---
    def _render_grid(items):
        cols_per_row = 4
        for i in range(0, len(items), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx >= len(items):
                    break
                item = items[idx]
                with col:
                    st.image(item["image"], use_container_width=True)
                    display_name = item["class_name"].replace("_", " ").title()
                    st.markdown(f"**{display_name}**")
                    st.caption(f"Confidence: {item['confidence']:.1%}")
                    if item["is_uncertain"]:
                        st.caption("⚠️ Needs review")
                    else:
                        st.caption("✓ High confidence")

    # Always render the first 12 (or fewer) items
    _render_grid(initial_items)

    # CHANGED: Show More / Show Less toggle when gallery exceeds 12 items
    if remaining_items:
        total = len(gallery_items)
        is_expanded = st.session_state.get("gallery_expanded", False)

        if is_expanded:
            # Render the remaining items
            _render_grid(remaining_items)

        # Centered toggle button
        _, btn_col, _ = st.columns([2, 1, 2])
        with btn_col:
            if is_expanded:
                if st.button("🔼 Show less", key="gallery_collapse_btn", use_container_width=True):
                    st.session_state["gallery_expanded"] = False
                    st.rerun()
            else:
                if st.button(f"🔽 Show all {total} detections", key="gallery_expand_btn", use_container_width=True):
                    st.session_state["gallery_expanded"] = True
                    st.rerun()

        st.caption("Top 12 shown by default, sorted by clinical priority")


# FEATURE 1 — Slide quality assessment pure CV function
def assess_slide_quality(image_bgr: np.ndarray) -> dict:
    """Assess blood smear slide quality before running inference.
    
    Performs three independent quality checks using classical computer 
    vision — no ML model involved. Fast, deterministic, interpretable.
    
    Checks:
        1. Blur — Laplacian variance measures frequency content.
           A focused microscopy image has sharp cell edges producing 
           high-frequency gradients. Blur suppresses these gradients,
           collapsing the variance. Threshold 80 is calibrated for 
           640x640 Giemsa-stained slides.
           
        2. Darkness — Mean pixel intensity of the greyscale image.
           Giemsa-stained slides under correct illumination typically
           produce mean brightness between 60 and 210. Below 40 
           indicates insufficient condenser illumination or a thick 
           smear that will absorb too much light for reliable detection.
           
        3. Overexposure — Saturated pixels (value 255 in any channel)
           as a percentage of total pixels. Above 15% saturation means
           the illumination is too intense, washing out morphological 
           detail the model depends on for classification.
    
    Returns:
        dict with keys:
            'issues': list of dicts, each with:
                'type': str ('blur' | 'dark' | 'bright')
                'severity': str ('warning' | 'error')
                'message': str — operator-facing message
                'detail': str — technical detail for the sidebar
            'metrics': dict with raw values:
                'sharpness': float — Laplacian variance
                'brightness': float — mean pixel intensity  
                'saturation_pct': float — % saturated pixels
            'passed': bool — True if no errors (warnings allowed)
            'has_warnings': bool — True if any issues exist
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    issues = []
    
    # ── Check 1: Blur via Laplacian variance ──────────────────────
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 2.0:
        issues.append({
            'type': 'blur',
            'severity': 'error',
            'message': (
                'This image is severely blurred. Inference has been '
                'paused. Please refocus the microscope objective and '
                'upload a new image before proceeding.'
            ),
            'detail': f'Sharpness score: {laplacian_var:.2f} (minimum: 2.00)',
        })
    elif laplacian_var < 2.7:
        issues.append({
            'type': 'blur',
            'severity': 'warning',
            'message': (
                'This image appears slightly out of focus. Detection '
                'results may be unreliable for faint ring-stage parasites. '
                'Consider refocusing and re-uploading for best accuracy.'
            ),
            'detail': f'Sharpness score: {laplacian_var:.2f} (recommended: >2.70)',
        })
    
    # ── Check 2: Darkness ─────────────────────────────────────────
    mean_brightness = float(gray.mean())
    if mean_brightness < 100:
        issues.append({
            'type': 'dark',
            'severity': 'error',
            'message': (
                'This image is too dark for reliable analysis. '
                'Check microscope condenser illumination and ensure '
                'the slide is correctly positioned on the stage.'
            ),
            'detail': f'Mean brightness: {mean_brightness:.1f}/255 (minimum: 100)',
        })
    elif mean_brightness < 130:
        issues.append({
            'type': 'dark',
            'severity': 'warning',
            'message': (
                'This image is darker than optimal. Parasite detection '
                'sensitivity may be reduced. Consider increasing '
                'condenser illumination.'
            ),
            'detail': f'Mean brightness: {mean_brightness:.1f}/255 (recommended: >130)',
        })
    
    # ── Check 3: Overexposure ─────────────────────────────────────
    saturated = np.sum(image_bgr >= 255) / image_bgr.size * 100
    if saturated > 25:
        issues.append({
            'type': 'bright',
            'severity': 'error',
            'message': (
                'This image is severely overexposed. Morphological '
                'detail required for parasite classification has been '
                'lost. Reduce microscope illumination intensity and '
                're-capture the image.'
            ),
            'detail': f'Saturated pixels: {saturated:.1f}% (maximum: 25%)',
        })
    elif saturated > 15:
        issues.append({
            'type': 'bright',
            'severity': 'warning',
            'message': (
                'This image may be slightly overexposed. Consider '
                'reducing illumination intensity for optimal detection '
                'of pale ring-stage parasites.'
            ),
            'detail': f'Saturated pixels: {saturated:.1f}% (recommended: <15%)',
        })
    
    has_errors = any(i['severity'] == 'error' for i in issues)
    
    return {
        'issues': issues,
        'metrics': {
            'sharpness': round(laplacian_var, 2),
            'brightness': round(mean_brightness, 1),
            'saturation_pct': round(float(saturated), 1),
        },
        'passed': not has_errors,
        'has_warnings': len(issues) > 0,
    }


# FEATURE 2 — Recompute verified parasitemia pure function
def compute_verified_parasitemia(
    result: PredictionResult,
    decisions: dict,
) -> dict:
    """Recompute parasitemia after clinician verification decisions.
    
    The clinician reviews each uncertain detection and marks it as 
    accepted (treat as confirmed parasite) or rejected (treat as 
    false positive). This function recalculates the core clinical 
    metrics using only accepted detections plus all confident 
    detections.
    
    Args:
        result: The original PredictionResult from model inference.
        decisions: dict mapping detection index (str) to bool.
                   True = accepted, False = rejected.
                   Detections not in decisions default to accepted
                   (conservative — err toward sensitivity).
    
    Returns:
        dict with keys:
            'verified_parasites': int
            'verified_rbc': int  
            'verified_parasitemia_pct': float
            'accepted_uncertain': int
            'rejected_uncertain': int
            'pending_uncertain': int — not yet reviewed
    """
    verified_parasites = 0
    verified_rbc = 0
    accepted_uncertain = 0
    rejected_uncertain = 0
    pending_uncertain = 0
    verified_class_counts = {}  # FIX 1 — per-class breakdown for PDF table
    
    for i, det in enumerate(result.detections):
        key = str(i)
        is_uncertain = (
            UNCERTAINTY_THRESHOLD_LOW 
            <= det.confidence 
            <= UNCERTAINTY_THRESHOLD_HIGH
        )
        
        if det.class_name == "red_blood_cell":
            verified_rbc += 1
            verified_class_counts[det.class_name] = (
                verified_class_counts.get(det.class_name, 0) + 1
            )
            continue
        
        if det.class_name not in PARASITE_CLASSES:
            continue
            
        if is_uncertain:
            if key not in decisions:
                # Not yet reviewed — include by default (conservative)
                verified_parasites += 1
                pending_uncertain += 1
                verified_class_counts[det.class_name] = (
                    verified_class_counts.get(det.class_name, 0) + 1
                )
            elif decisions[key]:
                # Clinician accepted
                verified_parasites += 1
                accepted_uncertain += 1
                verified_class_counts[det.class_name] = (
                    verified_class_counts.get(det.class_name, 0) + 1
                )
            else:
                # Clinician rejected — exclude from count entirely
                rejected_uncertain += 1
        else:
            # Confident detection — always included
            verified_parasites += 1
            verified_class_counts[det.class_name] = (
                verified_class_counts.get(det.class_name, 0) + 1
            )
    
    total_cells = verified_rbc + verified_parasites
    parasitemia = (
        (verified_parasites / total_cells * 100) 
        if total_cells > 0 else 0.0
    )
    
    return {
        'verified_parasites': verified_parasites,
        'verified_rbc': verified_rbc,
        'verified_parasitemia_pct': round(parasitemia, 2),
        'accepted_uncertain': accepted_uncertain,
        'rejected_uncertain': rejected_uncertain,
        'pending_uncertain': pending_uncertain,
        'verified_class_counts': verified_class_counts,  # FIX 1
    }


def _draw_filtered(
    image_bgr: np.ndarray,
    result: PredictionResult,
) -> np.ndarray:
    """Redraw detections showing only parasites (no healthy RBCs).

    CHANGED: Also applies uncertainty visual treatment — uncertain detections
    are drawn with yellow boxes and INCONCLUSIVE labels, matching the
    behaviour in predict.py's _draw_detections().
    """
    from src.inference.predict import CLASS_COLORS

    img = image_bgr.copy()
    for det in result.detections:
        if det.class_name == "red_blood_cell":
            continue  # Skip healthy cells

        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]

        # CHANGED: Apply uncertainty visual treatment
        uncertain = _is_uncertain(det.confidence)

        if uncertain:
            color = UNCERTAIN_COLOR
            thickness = 3
            label = f"INCONCLUSIVE {det.confidence:.2f}"
        else:
            color = CLASS_COLORS.get(det.class_name, (255, 255, 255))
            thickness = 2
            label = f"{det.class_name} {det.confidence:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
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
    resolved_path = PROJECT_ROOT / weights_path
    if not resolved_path.exists():
        return None
    return MalariaDetector(str(resolved_path), device="cpu")

if __name__ == "__main__":
    main()

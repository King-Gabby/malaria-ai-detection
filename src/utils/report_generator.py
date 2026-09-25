"""
Report Generator — Clinical PDF/CSV Reports for Radiology
"""

import io
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
from fpdf import FPDF
from PIL import Image


def strip_emoji_for_pdf(text: str) -> str:
    """Remove emoji and non-Latin-1 characters for PDF compatibility."""
    return text.encode("latin-1", errors="ignore").decode("latin-1").strip()


def generate_mri_report(
    result: Any,
    patient_details: Dict = None,
    report_meta: Dict = None,
) -> bytes:
    """Generate PDF report for MRI brain analysis."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    colors = {
        "header": (15, 52, 96),
        "text": (0, 0, 0),
        "accent": (100, 255, 218),
        "warning": (248, 113, 113),
    }

    # Title
    pdf.set_fill_color(*colors["header"])
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 14, "MRI Brain Analysis Report", ln=True, align="C", fill=True)
    pdf.set_font("Helvetica", "", 8)
    pdf.cell(0, 6, "YOLOv8n Detection + UNet Segmentation · BraTS/fastMRI", ln=True, align="C", fill=True)
    pdf.set_text_color(*colors["text"])
    pdf.ln(5)

    # Metadata
    if report_meta:
        pdf.set_font("Helvetica", "", 8)
        pdf.set_fill_color(245, 247, 250)
        meta_line = f"Report: {report_meta.get('report_number', 'N/A')} | Study: {report_meta.get('study_id', 'N/A')} | Date: {report_meta.get('date', '')} {report_meta.get('time', '')}"
        pdf.cell(0, 7, meta_line, ln=True, fill=True, align="C")
        pdf.ln(4)

    # Patient info
    if patient_details and any(v for v in patient_details.values() if v):
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Patient Information", ln=True)
        pdf.ln(1)

        for label, key in [("Name", "name"), ("Patient ID", "patient_id"), ("Age/Sex", "age_sex"),
                           ("Clinician", "clinician"), ("Facility", "facility")]:
            val = patient_details.get(key)
            if val:
                pdf.set_font("Helvetica", "B", 9)
                pdf.cell(50, 7, f"  {label}", border=1, fill=True)
                pdf.set_font("Helvetica", "", 9)
                pdf.cell(120, 7, f"  {strip_emoji_for_pdf(str(val))}", border=1, ln=True, fill=True)
        pdf.ln(6)

    # Results Summary
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Findings", ln=True)
    pdf.ln(2)

    # Tumor
    tumor_vol = getattr(result, 'total_tumor_volume_mm3', 0)
    edema_vol = getattr(result, 'total_edema_volume_mm3', 0)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Tumor:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    if tumor_vol > 0:
        pdf.multi_cell(0, 6, f"Tumor volume: {tumor_vol:.1f} mm³ ({tumor_vol/1000:.2f} mL)")
        pdf.multi_cell(0, 6, "Recommendation: Neurosurgical consultation advised. Consider contrast-enhanced MRI for characterization.")
    else:
        pdf.multi_cell(0, 6, "No tumor detected. Routine follow-up per clinical indication.")
    pdf.ln(3)

    # Edema
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Edema:", ln=True)
    pdf.set_font("Helvetica", "", 10)
    if edema_vol > 0:
        pdf.multi_cell(0, 6, f"Edema volume: {edema_vol:.1f} mm³ ({edema_vol/1000:.2f} mL)")
        pdf.multi_cell(0, 6, "Clinical correlation recommended. Edema may indicate mass effect or inflammation.")
    else:
        pdf.multi_cell(0, 6, "No significant peritumoral edema detected.")
    pdf.ln(3)

    # Detection details
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Detection Details", ln=True)
    pdf.set_font("Helvetica", "", 10)

    counts = result._per_class_counts() if hasattr(result, '_per_class_counts') else {}
    pdf.cell(90, 7, "Class", border=1, align="C")
    pdf.cell(40, 7, "Count", border=1, align="C", ln=True)

    for cls_name, count in counts.items():
        pdf.cell(90, 7, cls_name.replace("_", " ").title(), border=1)
        pdf.cell(40, 7, str(count), border=1, align="C", ln=True)

    pdf.ln(4)

    # Annotated image
    if hasattr(result, 'annotated_image') and result.annotated_image is not None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img_rgb = cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB)
            Image.fromarray(img_rgb).save(tmp.name)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Annotated MRI Slice", ln=True)
            pdf.image(tmp.name, w=180)

    # Disclaimer
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 5, strip_emoji_for_pdf(
        "DISCLAIMER: This is an AI-assisted screening tool for research and educational purposes only. "
        "It is NOT a certified medical diagnostic device. All findings must be confirmed by a qualified radiologist."
    ))

    return bytes(pdf.output())


def generate_ct_report(
    result: Any,
    patient_details: Dict = None,
    report_meta: Dict = None,
) -> bytes:
    """Generate PDF report for CT chest analysis."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_fill_color(15, 52, 96)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 14, "CT Chest Analysis Report", ln=True, align="C", fill=True)
    pdf.set_font("Helvetica", "", 8)
    pdf.cell(0, 6, "YOLOv8n Detection · LIDC/COVID-CT", ln=True, align="C", fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # Metadata
    if report_meta:
        pdf.set_font("Helvetica", "", 8)
        pdf.set_fill_color(245, 247, 250)
        meta_line = f"Report: {report_meta.get('report_number', 'N/A')} | Study: {report_meta.get('study_id', 'N/A')} | Date: {report_meta.get('date', '')} {report_meta.get('time', '')}"
        pdf.cell(0, 7, meta_line, ln=True, fill=True, align="C")
        pdf.ln(4)

    # Patient info
    if patient_details and any(v for v in patient_details.values() if v):
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Patient Information", ln=True)
        pdf.ln(1)
        for label, key in [("Name", "name"), ("Patient ID", "patient_id"), ("Age/Sex", "age_sex"),
                           ("Clinician", "clinician"), ("Facility", "facility")]:
            val = patient_details.get(key)
            if val:
                pdf.set_font("Helvetica", "B", 9)
                pdf.cell(50, 7, f"  {label}", border=1, fill=True)
                pdf.set_font("Helvetica", "", 9)
                pdf.cell(120, 7, f"  {strip_emoji_for_pdf(str(val))}", border=1, ln=True, fill=True)
        pdf.ln(6)

    # Nodule findings
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Pulmonary Nodule Assessment", ln=True)
    pdf.ln(2)

    total_vol = getattr(result, 'total_nodule_volume_mm3', 0)
    nodule_count = sum(1 for d in result.detections if d.class_name == "nodule") if hasattr(result, 'detections') else 0

    pdf.set_font("Helvetica", "", 10)
    if nodule_count > 0:
        pdf.multi_cell(0, 6, f"Nodules detected: {nodule_count}")
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 6, f"Total estimated nodule volume: {total_vol:.1f} mm3 ({total_vol/1000:.3f} mL)")

        # Fleischner Society guidelines reference
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, "Fleischner Society Guidelines (2017):", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 5, (
            "- Solid nodule <6mm: No routine follow-up (low risk); optional CT at 12 months (high risk)\n"
            "- Solid nodule 6-8mm: CT at 6-12 months, then 18-24 months\n"
            "- Solid nodule >8mm: Consider PET/CT, tissue sampling, or CT at 3 months\n"
            "- Subsolid/ground-glass: Different follow-up algorithm applies"
        ))
    else:
        pdf.multi_cell(0, 6, "No pulmonary nodules detected.")

    pdf.ln(3)

    # Other findings
    consolidation_count = sum(1 for d in result.detections if d.class_name == "consolidation") if hasattr(result, 'detections') else 0
    effusion_count = sum(1 for d in result.detections if d.class_name == "effusion") if hasattr(result, 'detections') else 0

    if consolidation_count > 0:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Consolidation:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, f"Consolidations detected: {consolidation_count}. Suggests pneumonia, organizing pneumonia, or malignancy. Clinical correlation required.")
        pdf.ln(3)

    if effusion_count > 0:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Pleural Effusion:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, f"Pleural effusions detected: {effusion_count}. Consider thoracentesis if clinically indicated.")
        pdf.ln(3)

    # Detection table
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Detection Summary", ln=True)
    pdf.set_font("Helvetica", "", 10)

    counts = result._per_class_counts() if hasattr(result, '_per_class_counts') else {}
    pdf.cell(90, 7, "Class", border=1, align="C")
    pdf.cell(40, 7, "Count", border=1, align="C", ln=True)

    for cls_name, count in counts.items():
        pdf.cell(90, 7, cls_name.replace("_", " ").title(), border=1)
        pdf.cell(40, 7, str(count), border=1, align="C", ln=True)

    pdf.ln(4)

    # Annotated image
    if hasattr(result, 'annotated_image') and result.annotated_image is not None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img_rgb = cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB)
            Image.fromarray(img_rgb).save(tmp.name)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Annotated CT Slice", ln=True)
            pdf.image(tmp.name, w=180)

    # Disclaimer
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 5, strip_emoji_for_pdf(
        "DISCLAIMER: This is an AI-assisted screening tool for research and educational purposes only. "
        "It is NOT a certified medical diagnostic device. All findings must be confirmed by a qualified radiologist."
    ))

    return bytes(pdf.output())


def generate_xray_report(
    result: Any,
    patient_details: Dict = None,
    report_meta: Dict = None,
) -> bytes:
    """Generate PDF report for Chest X-ray analysis."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_fill_color(15, 52, 96)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 14, "Chest X-ray Analysis Report", ln=True, align="C", fill=True)
    pdf.set_font("Helvetica", "", 8)
    pdf.cell(0, 6, "YOLOv8n Detection · CheXpert/NIH ChestX-ray", ln=True, align="C", fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # Metadata
    if report_meta:
        pdf.set_font("Helvetica", "", 8)
        pdf.set_fill_color(245, 247, 250)
        meta_line = f"Report: {report_meta.get('report_number', 'N/A')} | Study: {report_meta.get('study_id', 'N/A')} | Date: {report_meta.get('date', '')} {report_meta.get('time', '')}"
        pdf.cell(0, 7, meta_line, ln=True, fill=True, align="C")
        pdf.ln(4)

    # Patient info
    if patient_details and any(v for v in patient_details.values() if v):
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Patient Information", ln=True)
        pdf.ln(1)
        for label, key in [("Name", "name"), ("Patient ID", "patient_id"), ("Age/Sex", "age_sex"),
                           ("Clinician", "clinician"), ("Facility", "facility")]:
            val = patient_details.get(key)
            if val:
                pdf.set_font("Helvetica", "B", 9)
                pdf.cell(50, 7, f"  {label}", border=1, fill=True)
                pdf.set_font("Helvetica", "", 9)
                pdf.cell(120, 7, f"  {strip_emoji_for_pdf(str(val))}", border=1, ln=True, fill=True)
        pdf.ln(6)

    # Findings
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Findings", ln=True)
    pdf.ln(2)

    counts = result._per_class_counts() if hasattr(result, '_per_class_counts') else {}
    pneumonia = counts.get("pneumonia", 0)
    tb = counts.get("tb", 0)
    cardiomegaly = counts.get("cardiomegaly", 0)

    if pneumonia > 0:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(200, 50, 50)
        pdf.cell(0, 8, f"Pneumonia: {pneumonia} focus/foci detected", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, "Airspace opacity consistent with pneumonia. Recommend clinical correlation, inflammatory markers, and sputum culture. Consider follow-up CXR in 4-6 weeks.")
        pdf.ln(3)

    if tb > 0:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(200, 50, 50)
        pdf.cell(0, 8, f"Tuberculosis: {tb} suspicious lesion(s)", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, "Findings suggestive of pulmonary TB (upper lobe predilection, cavitation). Urgent sputum AFB x3, GeneXpert, and infectious disease referral recommended.")
        pdf.ln(3)

    if cardiomegaly > 0:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(255, 165, 0)
        pdf.cell(0, 8, f"Cardiomegaly: Detected", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, "Cardiothoracic ratio appears increased. Recommend echocardiogram for ejection fraction assessment and heart failure evaluation.")
        pdf.ln(3)

    if pneumonia == 0 and tb == 0 and cardiomegaly == 0:
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, "No significant pathology detected. Heart size and lung fields appear within normal limits.")

    pdf.ln(3)

    # Detection table
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Detection Summary", ln=True)
    pdf.set_font("Helvetica", "", 10)

    pdf.cell(90, 7, "Class", border=1, align="C")
    pdf.cell(40, 7, "Count", border=1, align="C", ln=True)

    for cls_name, count in counts.items():
        pdf.cell(90, 7, cls_name.replace("_", " ").title(), border=1)
        pdf.cell(40, 7, str(count), border=1, align="C", ln=True)

    pdf.ln(4)

    # Annotated image
    if hasattr(result, 'annotated_image') and result.annotated_image is not None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img_rgb = cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB)
            Image.fromarray(img_rgb).save(tmp.name)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Annotated Chest X-ray", ln=True)
            pdf.image(tmp.name, w=180)

    # Disclaimer
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 5, strip_emoji_for_pdf(
        "DISCLAIMER: This is an AI-assisted screening tool for research and educational purposes only. "
        "It is NOT a certified medical diagnostic device. All findings must be confirmed by a qualified radiologist."
    ))

    return bytes(pdf.output())


def generate_csv_report_radiology(result: Any, modality: str) -> str:
    """Generate CSV report for radiology results."""
    import csv
    output = io.StringIO()
    writer = csv.writer(output)

    if modality == "mri":
        writer.writerow(["class", "confidence", "slice_idx", "x1", "y1", "x2", "y2", "volume_mm3"])
        for d in result.detections:
            writer.writerow([d.class_name, f"{d.confidence:.4f}", d.slice_idx,
                           *[f"{v:.1f}" for v in d.bbox_xyxy], f"{d.volume_mm3:.2f}"])
    elif modality == "ct":
        writer.writerow(["class", "confidence", "slice_idx", "x1", "y1", "x2", "y2", "diameter_mm", "volume_mm3"])
        for d in result.detections:
            writer.writerow([d.class_name, f"{d.confidence:.4f}", d.slice_idx,
                           *[f"{v:.1f}" for v in d.bbox_xyxy],
                           f"{getattr(d, 'diameter_mm', 0):.1f}", f"{getattr(d, 'volume_mm3', 0):.2f}"])
    elif modality == "xray":
        writer.writerow(["class", "confidence", "laterality", "x1", "y1", "x2", "y2"])
        for d in result.detections:
            writer.writerow([d.class_name, f"{d.confidence:.4f}", d.laterality,
                           *[f"{v:.1f}" for v in d.bbox_xyxy]])

    return output.getvalue()
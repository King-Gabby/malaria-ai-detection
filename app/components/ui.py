"""
Shared UI Components
"""

import streamlit as st
from typing import Dict, List, Any


def metric_card(label: str, value: str, icon: str = "", color: str = "#64ffda") -> None:
    """Render a styled metric card."""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #233554;
        border-radius: 12px;
        padding: 1.4rem 1.2rem;
        text-align: center;
        color: white;
        height: 100%;
    ">
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
        <div style="font-size: 2.2rem; font-weight: 800; color: {color};">{value}</div>
        <div style="font-size: 0.8rem; color: #8892b0; margin-top: 0.5rem; text-transform: uppercase;">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def info_card(title: str, content: str, icon: str = "") -> None:
    """Render an info card."""
    st.markdown(f"""
    <div style="
        background: #16213e;
        border: 1px solid #233554;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    ">
        <div style="font-weight: 600; color: #FFFFFF; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
            <span>{icon}</span>
            <span>{title}</span>
        </div>
        <div style="color: #D8DEE9; font-size: 0.9rem;">{content}</div>
    </div>
    """, unsafe_allow_html=True)


def status_badge(text: str, status: str = "info") -> None:
    """Render a status badge."""
    colors = {
        "success": ("#1f3a2b", "#64ffda"),
        "warning": ("#3b3a1a", "#ffd700"),
        "error": ("#421818", "#ff4d4d"),
        "info": ("#1a2a3a", "#64ffda"),
    }
    bg, fg = colors.get(status, colors["info"])
    st.markdown(f"""
    <span style="
        background: {bg};
        color: {fg};
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    ">{text}</span>
    """, unsafe_allow_html=True)


def detection_table(detections: List[Dict], class_colors: Dict = None) -> None:
    """Render a styled detection table."""
    if not detections:
        st.info("No detections to display.")
        return

    import pandas as pd
    df = pd.DataFrame(detections)
    st.dataframe(df, use_container_width=True, hide_index=True)


def patient_summary(details: Dict) -> None:
    """Render a compact patient summary."""
    parts = []
    if details.get("name"):
        parts.append(f"**{details['name']}**")
    if details.get("age"):
        parts.append(f"Age {details['age']}")
    if details.get("sex"):
        parts.append(details["sex"])
    if details.get("patient_id"):
        parts.append(f"ID: `{details['patient_id']}`")

    if parts:
        st.info("🏥 " + " · ".join(parts))
    else:
        st.caption("No patient details entered")


def download_buttons(pdf_bytes: bytes = None, csv_data: str = None, img_bytes: bytes = None,
                     prefix: str = "report") -> None:
    """Render download buttons for reports."""
    c1, c2, c3 = st.columns(3)
    if pdf_bytes:
        with c1:
            st.download_button("📄 PDF", pdf_bytes, f"{prefix}.pdf", "application/pdf")
    if csv_data:
        with c2:
            st.download_button("📊 CSV", csv_data, f"{prefix}.csv", "text/csv")
    if img_bytes:
        with c3:
            st.download_button("🖼️ Image", img_bytes, f"{prefix}.png", "image/png")
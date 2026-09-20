"""
Footer Component — Session/Metrics/Analytics Panels
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Any


def render_footer(
    session_data: Dict[str, Any] = None,
    show_analytics: bool = True,
) -> None:
    """
    Render the footer with session info, metrics, and analytics.

    Args:
        session_data: Current session data dict
        show_analytics: Whether to show analytics panel
    """
    if session_data is None:
        session_data = st.session_state.get("session_data", {})

    st.markdown("---")

    # Footer panels
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 📊 Session Info")
        st.markdown(f"""
        <div style="font-size: 0.85rem; color: #8892b0; line-height: 1.8;">
            <div><strong>Session ID:</strong> {session_data.get('session_id', 'N/A')[:8]}</div>
            <div><strong>Started:</strong> {session_data.get('start_time', datetime.now().strftime('%H:%M:%S'))}</div>
            <div><strong>Module:</strong> {session_data.get('current_module', 'dashboard').title()}</div>
            <div><strong>Analyses:</strong> {session_data.get('analysis_count', 0)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### 📈 Quick Metrics")
        metrics = session_data.get("metrics", {})
        st.markdown(f"""
        <div style="font-size: 0.85rem; color: #8892b0; line-height: 1.8;">
            <div><strong>Total Detections:</strong> {metrics.get('total_detections', 0)}</div>
            <div><strong>Positive Cases:</strong> {metrics.get('positive_cases', 0)}</div>
            <div><strong>Verifications:</strong> {metrics.get('verifications', 0)}</div>
            <div><strong>Reports:</strong> {metrics.get('reports_generated', 0)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        if show_analytics:
            st.markdown("#### 📉 Analytics")
            analytics = session_data.get("analytics", {})
            st.markdown(f"""
            <div style="font-size: 0.85rem; color: #8892b0; line-height: 1.8;">
                <div><strong>Malaria:</strong> {analytics.get('malaria', 0)} scans</div>
                <div><strong>Sickle Cell:</strong> {analytics.get('sickle_cell', 0)} scans</div>
                <div><strong>ALL:</strong> {analytics.get('all', 0)} scans</div>
                <div><strong>Iron Def:</strong> {analytics.get('iron_deficiency', 0)} scans</div>
            </div>
            """, unsafe_allow_html=True)

    # Copyright
    st.markdown(
        "<div class='footer-credit'>"
        "Built by Team Devions &nbsp;·&nbsp; "
        "NACOS UI × DATICAN Competition 2026"
        "</div>",
        unsafe_allow_html=True
    )


def update_session_metrics(module: str, result: Dict = None) -> None:
    """Update session metrics after an analysis."""
    if "session_data" not in st.session_state:
        st.session_state.session_data = {
            "session_id": datetime.now().strftime("%Y%m%d%H%M%S"),
            "start_time": datetime.now().strftime("%H:%M:%S"),
            "current_module": "dashboard",
            "analysis_count": 0,
            "metrics": {
                "total_detections": 0,
                "positive_cases": 0,
                "verifications": 0,
                "reports_generated": 0,
            },
            "analytics": {
                "malaria": 0,
                "sickle_cell": 0,
                "all": 0,
                "iron_deficiency": 0,
            },
        }

    session = st.session_state.session_data
    session["analysis_count"] += 1
    session["current_module"] = module

    if result:
        session["metrics"]["total_detections"] += result.get("total_detections", 0)
        if result.get("positive", False):
            session["metrics"]["positive_cases"] += 1

    # Update module-specific analytics
    module_map = {
        "malaria": "malaria",
        "sickle_cell": "sickle_cell",
        "all": "all",
        "iron_deficiency": "iron_deficiency",
        "mri": "malaria",  # Radiology - map to existing for now
        "ct": "malaria",
        "xray": "malaria",
    }
    if module in module_map:
        session["analytics"][module_map[module]] += 1
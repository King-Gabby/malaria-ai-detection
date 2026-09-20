"""
Clinical Theme — Teal/Emerald Dark Theme for Medical UI
"""

import streamlit as st


def get_theme_colors() -> dict:
    """Return the clinical color palette."""
    return {
        # Background
        "bg_primary": "#0d0d1a",
        "bg_secondary": "#16213e",
        "bg_card": "#1a1a2e",
        "bg_hover": "#1f2a48",

        # Text
        "text_primary": "#D8DEE9",
        "text_secondary": "#8892b0",
        "text_muted": "#6b7280",
        "text_accent": "#64ffda",

        # Accents
        "accent_teal": "#64ffda",
        "accent_emerald": "#4ade80",
        "accent_red": "#e94560",
        "accent_orange": "#ff9800",
        "accent_yellow": "#ffd700",
        "accent_purple": "#ff00ff",

        # Borders
        "border_subtle": "#233554",
        "border_accent": "#64ffda",

        # Status
        "success": "#4ade80",
        "warning": "#ffd700",
        "error": "#f87171",
        "info": "#64ffda",
    }


def apply_theme() -> None:
    """Apply the clinical dark theme via CSS injection."""
    colors = get_theme_colors()

    css = f"""
    <style>
    /* ============================================================
       GLOBAL DARK THEME OVERRIDE
    ============================================================ */
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stHeader"], [data-testid="stToolbar"],
    .main, .block-container {{
        background-color: {colors["bg_primary"]} !important;
        color: {colors["text_primary"]} !important;
    }}

    [data-testid="stSidebar"] {{
        background-color: {colors["bg_primary"]} !important;
    }}

    [data-testid="stSidebar"] > div {{
        background-color: {colors["bg_primary"]} !important;
    }}

    /* Force all headings to white */
    h1, h2, h3, h4, h5, h6 {{
        color: #FFFFFF !important;
    }}

    /* Force body text color */
    p, span, label, div {{
        color: {colors["text_primary"]};
    }}

    /* Input fields */
    .stTextInput input, .stNumberInput input, .stTextArea textarea,
    .stSelectbox > div > div {{
        background-color: {colors["bg_card"]} !important;
        color: {colors["text_primary"]} !important;
        border-color: {colors["border_subtle"]} !important;
    }}

    .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {{
        border-color: {colors["accent_teal"]} !important;
        box-shadow: 0 0 0 2px {colors["accent_teal"]}40 !important;
    }}

    /* Selectbox dropdown */
    [data-baseweb="select"] > div {{
        background-color: {colors["bg_card"]} !important;
        color: {colors["text_primary"]} !important;
    }}

    [data-baseweb="popover"] {{
        background-color: {colors["bg_card"]} !important;
    }}

    [role="listbox"] {{
        background-color: {colors["bg_card"]} !important;
    }}

    [role="option"] {{
        background-color: {colors["bg_card"]} !important;
        color: {colors["text_primary"]} !important;
    }}

    [role="option"]:hover {{
        background-color: {colors["bg_hover"]} !important;
    }}

    /* File uploader */
    [data-testid="stFileUploader"] {{
        background-color: rgba(255,255,255,0.03) !important;
        color: {colors["text_primary"]} !important;
    }}

    [data-testid="stFileUploaderDropzone"] {{
        background-color: rgba(255,255,255,0.03) !important;
        border-color: {colors["border_subtle"]} !important;
        border-style: dashed !important;
    }}

    [data-testid="stFileUploaderDropzone"]:hover {{
        border-color: {colors["accent_teal"]} !important;
    }}

    /* Expanders */
    [data-testid="stExpander"] {{
        background-color: {colors["bg_card"]} !important;
        border: 1px solid {colors["border_subtle"]} !important;
        border-radius: 10px !important;
        margin-bottom: 0.8rem;
    }}

    [data-testid="stExpander"] summary {{
        color: {colors["text_secondary"]} !important;
        background-color: transparent !important;
        font-weight: 600;
    }}

    /* Info/Warning/Error boxes */
    [data-testid="stInfo"] {{
        background-color: rgba(100, 255, 218, 0.05) !important;
        border-left: 3px solid {colors["accent_teal"]} !important;
        border-radius: 0 8px 8px 0 !important;
    }}

    [data-testid="stWarning"] {{
        background-color: rgba(255, 215, 0, 0.05) !important;
        border-left: 3px solid {colors["accent_yellow"]} !important;
        border-radius: 0 8px 8px 0 !important;
    }}

    [data-testid="stError"] {{
        background-color: rgba(248, 113, 113, 0.05) !important;
        border-left: 3px solid {colors["error"]} !important;
        border-radius: 0 8px 8px 0 !important;
    }}

    [data-testid="stSuccess"] {{
        background-color: rgba(74, 222, 128, 0.05) !important;
        border-left: 3px solid {colors["success"]} !important;
        border-radius: 0 8px 8px 0 !important;
    }}

    /* Radio buttons */
    [data-testid="stRadio"] > div {{
        gap: 1rem;
    }}

    [data-testid="stRadio"] label {{
        color: {colors["text_primary"]} !important;
    }}

    /* Dataframe / tables */
    [data-testid="stDataFrame"] {{
        background-color: {colors["bg_card"]} !important;
        border: 1px solid {colors["border_subtle"]} !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }}

    /* Checkboxes */
    [data-testid="stCheckbox"] label {{
        color: {colors["text_primary"]} !important;
    }}

    /* Metrics */
    [data-testid="stMetric"] {{
        background-color: transparent !important;
    }}

    [data-testid="stMetricLabel"] {{
        color: {colors["text_muted"]} !important;
    }}

    [data-testid="stMetricValue"] {{
        color: #FFFFFF !important;
    }}

    /* Slider */
    [data-testid="stSlider"] [data-baseweb="slider"] {{
        color: {colors["accent_teal"]} !important;
    }}

    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, {colors["accent_red"]}, #c23152) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.7rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.01em !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 2px 8px rgba(233, 69, 96, 0.3) !important;
    }}

    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(233, 69, 96, 0.45) !important;
    }}

    .stButton > button:active {{
        transform: translateY(0px) !important;
        box-shadow: 0 2px 8px rgba(233, 69, 96, 0.3) !important;
    }}

    /* Primary button */
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, {colors["accent_teal"]}, #00b894) !important;
        color: {colors["bg_primary"]} !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.7rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.01em !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 2px 8px rgba(100, 255, 218, 0.3) !important;
    }}

    .stButton > button[kind="primary"]:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(100, 255, 218, 0.45) !important;
    }}

    /* Sidebar buttons */
    [data-testid="stSidebar"] div[data-testid="stButton"] > button {{
        background: rgba(255,255,255,0.03) !important;
        background-image: none !important;
        color: {colors["text_primary"]} !important;
        border: 1px solid {colors["border_subtle"]} !important;
        border-radius: 8px !important;
        box-shadow: none !important;
        font-weight: 500 !important;
        padding: 0.6rem 1rem !important;
    }}

    [data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] {{
        background: rgba(100,255,218,0.12) !important;
        background-image: none !important;
        color: {colors["accent_teal"]} !important;
        border: 1px solid rgba(100,255,218,0.4) !important;
        box-shadow: none !important;
    }}

    /* Main content CTA buttons */
    section.main div[data-testid="stButton"] > button[kind="primary"],
    div[data-testid="stAppViewContainer"] section.main
        div[data-testid="stButton"] > button[kind="primary"] {{
        background: linear-gradient(135deg, {colors["accent_red"]}, #c23152) !important;
        background-color: {colors["accent_red"]} !important;
        background-image: linear-gradient(135deg, {colors["accent_red"]}, #c23152) !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.85rem 2rem !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        box-shadow: 0 4px 16px rgba(233, 69, 96, 0.35) !important;
        transition: all 0.25s ease !important;
    }}

    section.main div[data-testid="stButton"] > button[kind="primary"]:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(233, 69, 96, 0.5) !important;
    }}

    /* Tabs */
    [data-testid="stTabs"] [role="tablist"] {{
        gap: 0.5rem;
    }}

    [data-testid="stTabs"] [role="tab"] {{
        background-color: {colors["bg_card"]} !important;
        color: {colors["text_secondary"]} !important;
        border: 1px solid {colors["border_subtle"]} !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 0.75rem 1.5rem !important;
    }}

    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
        background-color: {colors["bg_primary"]} !important;
        color: {colors["accent_teal"]} !important;
        border-bottom-color: {colors["bg_primary"]} !important;
    }}

    /* Progress bar */
    [data-testid="stProgress"] > div > div {{
        background-color: {colors["accent_teal"]} !important;
    }}

    /* Spinner */
    [data-testid="stSpinner"] {{
        color: {colors["accent_teal"]} !important;
    }}

    /* Horizontal rule */
    hr {{
        border: none !important;
        border-top: 1px solid {colors["border_subtle"]} !important;
        margin: 1.5rem 0 !important;
    }}

    /* Scrollbar */
    ::-webkit-scrollbar {{
        width: 6px;
        height: 6px;
    }}

    ::-webkit-scrollbar-track {{
        background: {colors["bg_secondary"]};
    }}

    ::-webkit-scrollbar-thumb {{
        background: {colors["border_subtle"]};
        border-radius: 3px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: {colors["accent_teal"]};
    }}

    /* Custom component classes */
    .metric-card {{
        background: linear-gradient(135deg, {colors["bg_card"]}, {colors["bg_secondary"]});
        border: 1px solid {colors["border_subtle"]};
        border-radius: 12px;
        padding: 1.4rem 1.2rem;
        text-align: center;
        color: white;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
    }}

    .metric-card:hover {{
        border-color: rgba(100, 255, 218, 0.3);
        box-shadow: 0 4px 20px rgba(100, 255, 218, 0.08);
    }}

    .metric-value {{
        font-size: 2.2rem;
        font-weight: 800;
        color: {colors["accent_teal"]};
        letter-spacing: -0.02em;
        line-height: 1;
    }}

    .metric-label {{
        font-size: 0.8rem;
        color: {colors["text_muted"]};
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 500;
    }}

    .severity-badge {{
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1rem;
        text-align: center;
    }}

    .detection-table {{
        width: 100%;
        border-collapse: collapse;
        border-radius: 10px;
        overflow: hidden;
    }}

    .detection-table th {{
        background: {colors["bg_card"]};
        color: {colors["accent_teal"]};
        padding: 0.7rem 1rem;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
    }}

    .detection-table td {{
        padding: 0.6rem 1rem;
        border-bottom: 1px solid {colors["border_subtle"]};
        color: {colors["text_secondary"]};
        font-size: 0.9rem;
    }}

    .detection-table tr:last-child td {{
        border-bottom: none;
    }}

    .detection-table tr:hover td {{
        background: rgba(100, 255, 218, 0.03);
    }}

    .footer-credit {{
        text-align: center;
        color: {colors["text_muted"]};
        font-size: 0.78rem;
        padding: 2rem 0 1rem;
        border-top: 1px solid {colors["bg_secondary"]};
        margin-top: 3rem;
        letter-spacing: 0.03em;
    }}

    /* Card grid */
    .card-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
    }}

    .info-card {{
        background: {colors["bg_card"]};
        border: 1px solid {colors["border_subtle"]};
        border-radius: 10px;
        padding: 1rem;
        transition: border-color 0.2s ease;
    }}

    .info-card:hover {{
        border-color: rgba(100, 255, 218, 0.3);
    }}
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)

    # Also set Streamlit config options
    try:
        st.config.set_option("theme.base", "dark")
        st.config.set_option("theme.primaryColor", colors["accent_teal"])
        st.config.set_option("theme.backgroundColor", colors["bg_primary"])
        st.config.set_option("theme.secondaryBackgroundColor", colors["bg_card"])
        st.config.set_option("theme.textColor", colors["text_primary"])
    except Exception:
        pass
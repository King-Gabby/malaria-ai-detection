"""RaphaID AI - Reusable UI Components"""

import streamlit as st
from typing import Optional, List, Dict, Any
from .theme import COLORS


def metric_card(
    label: str,
    value: str,
    icon: str = "",
    color: str = "primary",
    help_text: Optional[str] = None,
    delta: Optional[str] = None,
    delta_color: str = "normal",
) -> None:
    """Render a metric card with consistent styling."""
    color_map = {
        "primary": COLORS["brand"]["primary"],
        "secondary": COLORS["brand"]["secondary"],
        "success": COLORS["semantic"]["success"],
        "warning": COLORS["semantic"]["warning"],
        "error": COLORS["semantic"]["error"],
        "info": COLORS["semantic"]["info"],
    }
    accent_color = color_map.get(color, COLORS["brand"]["primary"])
    
    delta_html = ""
    if delta:
        delta_color_map = {
            "normal": COLORS["text"]["muted"],
            "positive": COLORS["semantic"]["success"],
            "negative": COLORS["semantic"]["error"],
        }
        dc = delta_color_map.get(delta_color, COLORS["text"]["muted"])
        delta_html = f'<div style="font-size: var(--font-size-xs); color: {dc}; margin-top: var(--spacing-xs); font-weight: 500;">{delta}</div>'
    
    help_html = ""
    if help_text:
        help_html = f'<div style="font-size: var(--font-size-xs); color: var(--color-text-dim); margin-top: var(--spacing-xs);">{help_text}</div>'
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, var(--color-background-card), var(--color-background-tertiary));
        border: 1px solid var(--color-border-light);
        border-radius: var(--radius-lg);
        padding: var(--spacing-lg);
        text-align: center;
        transition: all var(--transition-normal);
        height: 100%;
    " onmouseover="this.style.borderColor='var(--color-border-medium)'; this.style.boxShadow='var(--shadow-md)'" 
       onmouseout="this.style.borderColor='var(--color-border-light)'; this.style.boxShadow='none'">
        {f'<div style="font-size: 1.5rem; margin-bottom: var(--spacing-sm);">{icon}</div>' if icon else ''}
        <div style="font-size: var(--font-size-xs); color: var(--color-text-muted); 
                    text-transform: uppercase; letter-spacing: var(--letter-spacing-wider); 
                    font-weight: 600; margin-bottom: var(--spacing-xs);">{label}</div>
        <div style="font-size: var(--font-size-3xl); font-weight: 800; 
                    color: {accent_color}; line-height: 1; letter-spacing: var(--letter-spacing-tight);">
            {value}
        </div>
        {delta_html}
        {help_html}
    </div>
    """, unsafe_allow_html=True)


def info_card(
    title: str,
    content: str,
    icon: str = "",
    variant: str = "default",
    action_label: Optional[str] = None,
    action_key: Optional[str] = None,
) -> bool:
    """Render an info card with optional action button. Returns True if action clicked."""
    variant_styles = {
        "default": {
            "bg": "linear-gradient(135deg, var(--color-background-card), var(--color-background-tertiary))",
            "border": "var(--color-border-light)",
            "icon_bg": "var(--color-overlay-medium)",
        },
        "primary": {
            "bg": "linear-gradient(135deg, var(--color-brand-primary), var(--color-brand-primary_light))",
            "border": "transparent",
            "icon_bg": "rgba(255, 255, 255, 0.2)",
        },
        "success": {
            "bg": "linear-gradient(135deg, var(--color-semantic-success_bg), var(--color-background-tertiary))",
            "border": "var(--color-semantic-success_border)",
            "icon_bg": "var(--color-semantic-success_bg)",
        },
        "warning": {
            "bg": "linear-gradient(135deg, var(--color-semantic-warning_bg), var(--color-background-tertiary))",
            "border": "var(--color-semantic-warning_border)",
            "icon_bg": "var(--color-semantic-warning_bg)",
        },
        "error": {
            "bg": "linear-gradient(135deg, var(--color-semantic-error_bg), var(--color-background-tertiary))",
            "border": "var(--color-semantic-error_border)",
            "icon_bg": "var(--color-semantic-error_bg)",
        },
        "info": {
            "bg": "linear-gradient(135deg, var(--color-semantic-info_bg), var(--color-background-tertiary))",
            "border": "var(--color-semantic-info_border)",
            "icon_bg": "var(--color-semantic-info_bg)",
        },
    }
    
    style = variant_styles.get(variant, variant_styles["default"])
    text_color = "var(--color-text-inverse)" if variant == "primary" else "var(--color-text-primary)"
    muted_color = "rgba(255, 255, 255, 0.7)" if variant == "primary" else "var(--color-text-muted)"
    
    clicked = False
    if action_label and action_key:
        clicked = st.button(action_label, key=action_key, use_container_width=True)
    
    action_html = ""
    if action_label:
        action_html = f"""
        <div style="margin-top: var(--spacing-md); padding-top: var(--spacing-md); border-top: 1px solid var(--color-border-light);">
            <div id="{action_key}-placeholder" style="height: 2.5rem;"></div>
        </div>
        """
    
    st.markdown(f"""
    <div style="
        background: {style['bg']};
        border: 1px solid {style['border']};
        border-radius: var(--radius-lg);
        padding: var(--spacing-lg);
        transition: all var(--transition-normal);
        height: 100%;
    ">
        <div style="display: flex; align-items: flex-start; gap: var(--spacing-md);">
            {f'<div style="width: 48px; height: 48px; background: {style["icon_bg"]}; border-radius: var(--radius-md); display: flex; align-items: center; justify-content: center; font-size: 1.5rem; flex-shrink: 0;">{icon}</div>' if icon else ''}
            <div style="flex: 1; min-width: 0;">
                <div style="font-size: var(--font-size-lg); font-weight: 700; color: {text_color}; margin-bottom: var(--spacing-xs);">{title}</div>
                <div style="font-size: var(--font-size-base); color: {muted_color}; line-height: var(--line-height-relaxed);">{content}</div>
            </div>
        </div>
        {action_html}
    </div>
    """, unsafe_allow_html=True)
    
    return clicked


def status_badge(
    text: str,
    variant: str = "default",
    icon: str = "",
    size: str = "md",
) -> None:
    """Render a status badge."""
    variant_styles = {
        "default": {"bg": "var(--color-overlay-medium)", "color": "var(--color-text-secondary)", "border": "var(--color-border-light)"},
        "success": {"bg": "var(--color-semantic-success_bg)", "color": "var(--color-semantic-success)", "border": "var(--color-semantic-success_border)"},
        "warning": {"bg": "var(--color-semantic-warning_bg)", "color": "var(--color-semantic-warning)", "border": "var(--color-semantic-warning_border)"},
        "error": {"bg": "var(--color-semantic-error_bg)", "color": "var(--color-semantic-error)", "border": "var(--color-semantic-error_border)"},
        "info": {"bg": "var(--color-semantic-info_bg)", "color": "var(--color-semantic-info)", "border": "var(--color-semantic-info_border)"},
        "pending": {"bg": "rgba(255, 215, 0, 0.15)", "color": "#ffd700", "border": "rgba(255, 215, 0, 0.3)"},
    }
    
    size_styles = {
        "sm": {"padding": "0.2rem 0.5rem", "font_size": "var(--font-size-xs)", "gap": "0.25rem"},
        "md": {"padding": "0.35rem 0.75rem", "font_size": "var(--font-size-sm)", "gap": "0.35rem"},
        "lg": {"padding": "0.5rem 1rem", "font_size": "var(--font-size-base)", "gap": "0.5rem"},
    }
    
    style = variant_styles.get(variant, variant_styles["default"])
    sz = size_styles.get(size, size_styles["md"])
    
    st.markdown(f"""
    <span style="
        display: inline-flex;
        align-items: center;
        gap: {sz['gap']};
        padding: {sz['padding']};
        background: {style['bg']};
        border: 1px solid {style['border']};
        border-radius: var(--radius-full);
        font-size: {sz['font_size']};
        font-weight: 600;
        color: {style['color']};
        letter-spacing: var(--letter-spacing-wide);
    ">
        {icon if icon else ''}{text}
    </span>
    """, unsafe_allow_html=True)


def section_header(
    title: str,
    subtitle: Optional[str] = None,
    icon: str = "",
    action: Optional[Dict[str, Any]] = None,
) -> None:
    """Render a section header with optional action."""
    action_html = ""
    if action:
        action_html = f"""
        <div>
            <button style="
                background: var(--color-brand-primary);
                color: white;
                border: none;
                border-radius: var(--radius-md);
                padding: var(--spacing-sm) var(--spacing-md);
                font-weight: 600;
                font-size: var(--font-size-sm);
                cursor: pointer;
                transition: all var(--transition-fast);
            " onmouseover="this.style.background='var(--color-brand-primary_hover)'" 
               onmouseout="this.style.background='var(--color-brand-primary)'">
                {action['label']}
            </button>
        </div>
        """
    
    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: var(--spacing-md); flex-wrap: wrap; gap: var(--spacing-md);">
        <div style="display: flex; align-items: center; gap: var(--spacing-sm);">
            {f'<span style="font-size: 1.5rem;">{icon}</span>' if icon else ''}
            <div>
                <h2 style="margin: 0; font-size: var(--font-size-2xl); font-weight: 800; color: var(--color-text-primary);">{title}</h2>
                {f'<p style="margin: var(--spacing-xs) 0 0 0; font-size: var(--font-size-base); color: var(--color-text-muted);">{subtitle}</p>' if subtitle else ''}
            </div>
        </div>
        {action_html}
    </div>
    """, unsafe_allow_html=True)


def divider(label: Optional[str] = None) -> None:
    """Render a styled divider with optional label."""
    if label:
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin: var(--spacing-xl) 0; color: var(--color-text-muted);">
            <div style="flex: 1; border-top: 1px solid var(--color-border-light);"></div>
            <span style="padding: 0 var(--spacing-md); font-size: var(--font-size-xs); font-weight: 600; text-transform: uppercase; letter-spacing: var(--letter-spacing-wider);">{label}</span>
            <div style="flex: 1; border-top: 1px solid var(--color-border-light);"></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <hr style="border: none; border-top: 1px solid var(--color-border-light); margin: var(--spacing-xl) 0;">
        """, unsafe_allow_html=True)


def empty_state(
    title: str,
    message: str,
    icon: str = "🔍",
    action_label: Optional[str] = None,
    action_key: Optional[str] = None,
) -> bool:
    """Render an empty state with optional action."""
    clicked = False
    if action_label and action_key:
        clicked = st.button(action_label, key=action_key, use_container_width=True, type="primary")
    
    st.markdown(f"""
    <div style="text-align: center; padding: var(--spacing-3xl) var(--spacing-xl); color: var(--color-text-muted);">
        <div style="font-size: 4rem; margin-bottom: var(--spacing-lg); opacity: 0.5;">{icon}</div>
        <h3 style="margin: 0 0 var(--spacing-sm) 0; font-size: var(--font-size-xl); font-weight: 700; color: var(--color-text-primary);">{title}</h3>
        <p style="margin: 0; font-size: var(--font-size-base); max-width: 400px; margin-left: auto; margin-right: auto;">{message}</p>
    </div>
    """, unsafe_allow_html=True)
    
    return clicked


def button_primary(label: str, key: str, use_container_width: bool = True, disabled: bool = False) -> bool:
    """Render a primary action button."""
    return st.button(label, key=key, use_container_width=use_container_width, disabled=disabled, type="primary")


def button_secondary(label: str, key: str, use_container_width: bool = True, disabled: bool = False) -> bool:
    """Render a secondary action button."""
    return st.button(label, key=key, use_container_width=use_container_width, disabled=disabled, type="secondary")


def button_ghost(label: str, key: str, use_container_width: bool = True, disabled: bool = False) -> bool:
    """Render a ghost/tertiary action button."""
    return st.button(label, key=key, use_container_width=use_container_width, disabled=disabled)


def card_grid(items: List[Dict[str, Any]], cols: int = 4) -> None:
    """Render a responsive grid of cards."""
    for i in range(0, len(items), cols):
        row_items = items[i:i + cols]
        columns = st.columns(len(row_items))
        for col, item in zip(columns, row_items):
            with col:
                metric_card(
                    label=item.get("label", ""),
                    value=item.get("value", ""),
                    icon=item.get("icon", ""),
                    color=item.get("color", "primary"),
                    help_text=item.get("help"),
                    delta=item.get("delta"),
                    delta_color=item.get("delta_color", "normal"),
                )


def stat_row(stats: List[Dict[str, Any]]) -> None:
    """Render a horizontal row of statistics."""
    cols = st.columns(len(stats))
    for col, stat in zip(cols, stats):
        with col:
            st.markdown(f"""
            <div style="text-align: center; padding: var(--spacing-md);">
                <div style="font-size: var(--font-size-2xl); font-weight: 800; color: var(--color-brand-secondary);">{stat['value']}</div>
                <div style="font-size: var(--font-size-xs); color: var(--color-text-muted); text-transform: uppercase; letter-spacing: var(--letter-spacing-wider); margin-top: var(--spacing-xs);">{stat['label']}</div>
            </div>
            """, unsafe_allow_html=True)
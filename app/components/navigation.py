"""
Navigation Component — 3-Module Dropdown Navigation
"""

import streamlit as st
from typing import Dict, Callable


def render_navigation(
    current_module: str,
    on_module_change: Callable[[str], None],
    modules: Dict[str, Dict] = None,
) -> None:
    """
    Render the 3-module navigation with dropdowns.

    Args:
        current_module: Currently active module ('detection', 'radiology', 'chatbot')
        on_module_change: Callback when module changes
        modules: Module configuration dict
    """
    if modules is None:
        modules = {
            "detection": {
                "label": "🔬 Detection",
                "icon": "🔬",
                "submodules": {
                    "malaria": "🦠 Malaria",
                    "sickle_cell": "🩸 Sickle Cell",
                    "all": "🧬 ALL Leukemia",
                    "iron_deficiency": "🩸 Iron Deficiency",
                },
            },
            "radiology": {
                "label": "🏥 Radiology",
                "icon": "🏥",
                "submodules": {
                    "mri": "🧠 MRI Brain",
                    "ct": "🫁 CT Chest",
                    "xray": "🩻 X-ray Chest",
                },
            },
            "chatbot": {
                "label": "🤖 Chatbot",
                "icon": "🤖",
                "submodules": {
                    "assistant": "💬 Medical Assistant",
                },
            },
        }

    with st.sidebar:
        # Logo/Brand
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; margin-bottom: 1rem;
                    background: linear-gradient(135deg, #0d0d1a, #1a1a2e);
                    border-radius: 12px; border: 1px solid #233554;">
            <div style="font-size: 2rem;">🩺</div>
            <div style="font-size: 1.3rem; font-weight: 800; color: #64ffda; margin-top: 0.5rem;">
                RaphaID AI
            </div>
            <div style="font-size: 0.75rem; color: #8892b0; margin-top: 0.25rem;">
                Offline Multi-Disease Diagnostic
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Main module selection
        st.markdown("### 📋 Modules")

        for module_key, module_info in modules.items():
            is_active = module_key == current_module

            # Main module button
            btn_label = f"{module_info['icon']} {module_info['label']}"
            if st.button(
                btn_label,
                key=f"nav_{module_key}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                on_module_change(module_key)
                st.rerun()

            # Show submodules if active
            if is_active and "submodules" in module_info:
                st.markdown("<div style='margin-left: 0.5rem; margin-bottom: 0.5rem;'>", unsafe_allow_html=True)
                for sub_key, sub_label in module_info["submodules"].items():
                    if st.button(
                        f"  {sub_label}",
                        key=f"nav_{module_key}_{sub_key}",
                        use_container_width=True,
                        type="secondary",
                    ):
                        st.session_state[f"{module_key}_submodule"] = sub_key
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")

        # Quick status
        st.markdown("### 🟢 System Status")
        status_items = [
            ("Models", "Lazy Loaded"),
            ("Device", "CPU"),
            ("RAM", "< 6GB Target"),
            ("Offline", "Ready"),
        ]
        for label, value in status_items:
            st.markdown(
                f"<div style='font-size: 0.8rem; color: #8892b0; "
                f"padding: 2px 0; display: flex; justify-content: space-between;'>"
                f"<span>🟢 {label}</span><span>{value}</span></div>",
                unsafe_allow_html=True
            )


def render_module_tabs(module_key: str, submodules: Dict[str, str]) -> str:
    """Render tabs for submodules within a module."""
    tabs = st.tabs(list(submodules.values()))
    for i, (sub_key, sub_label) in enumerate(submodules.items()):
        with tabs[i]:
            # This will be filled by the module's render function
            pass
    # Return the selected submodule (first one for now)
    return list(submodules.keys())[0]
"""
Shared UI Components
"""

from .theme import apply_theme, get_theme_colors
from .navigation import render_navigation
from .footer import render_footer

__all__ = ["apply_theme", "get_theme_colors", "render_navigation", "render_footer"]
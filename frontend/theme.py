from __future__ import annotations

# We use Streamlit's native CSS variables as much as possible.
# These tokens are for custom elements that need specific accent colors.
COLORS: dict[str, str] = {
    "accent": "var(--primary-color)",
    "accent_hover": "var(--primary-color)",
    "accent_light": "rgba(108, 99, 255, 0.1)",
    "text_primary": "var(--text-color)",
    "text_secondary": "rgba(128, 128, 128, 0.7)",
    "text_muted": "rgba(128, 128, 128, 0.5)",
    "success": "#00C9A7",
    "error": "#FF6B6B",
    "warning": "#FFB347",
    
    # Restoring Dashboard chart palette to prevent KeyErrors
    "chart1": "#6C63FF",
    "chart2": "#00C9A7",
    "chart3": "#FFB347",
    "chart4": "#FF6B6B",
    "chart5": "#38BDF8",
    "bg_card": "var(--secondary-bg-color)",
    "bg_secondary": "var(--secondary-bg-color)",
    "border": "rgba(128, 128, 128, 0.2)",
}

# CSS Variables mapping for easier injection
CSS_VARS = """
:root {
    --invenio-accent: var(--primary-color);
    --invenio-bg-card: var(--secondary-bg-color);
    --invenio-border: rgba(128, 128, 128, 0.2);
}
"""

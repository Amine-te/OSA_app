# ─────────────────────────────────────────────────────────────
# styles.py — Dual-theme stylesheet system for OSA Desktop
# Themes: "light" (Docker-inspired) | "dark" (OSA purple)
# ─────────────────────────────────────────────────────────────
from __future__ import annotations
from typing import Optional

FONT_FAMILY = "'Inter', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"

# ── Docker Desktop–Inspired Light Theme ──────────────────────
LIGHT_COLORS: dict = {
    "bg_primary":         "#F2F4F7",
    "bg_secondary":       "#FFFFFF",
    "bg_card":            "#FFFFFF",
    "bg_sidebar":         "#1A2540",   # Docker-style dark navy sidebar
    "bg_input":           "#F9FAFB",
    "bg_hover":           "#EEF2FF",
    "border":             "#E2E5EA",
    "border_accent":      "#1D63ED",
    "text_primary":       "#111827",
    "text_secondary":     "#4B5563",
    "text_muted":         "#9CA3AF",
    "accent_start":       "#1D63ED",
    "accent_end":         "#1448C0",
    "accent_mid":         "#1A5ADA",
    "success":            "#16A34A",
    "warning":            "#D97706",
    "danger":             "#DC2626",
    "info":               "#2563EB",
    # Sidebar-specific (dark panel, light text)
    "sidebar_text":       "#E2E8F0",
    "sidebar_text_muted": "#94A3B8",
    "sidebar_active_bg":  "#2D4A8A",
    "sidebar_hover_bg":   "#243154",
    "plot_bg":            "#FFFFFF",
}

# ── OSA Original Dark Theme (purple gradient) ─────────────────
DARK_COLORS: dict = {
    "bg_primary":         "#0e1117",
    "bg_secondary":       "#161b22",
    "bg_card":            "#1c2128",
    "bg_sidebar":         "#131720",
    "bg_input":           "#21262d",
    "bg_hover":           "#292e36",
    "border":             "#30363d",
    "border_accent":      "#667eea",
    "text_primary":       "#e6edf3",
    "text_secondary":     "#8b949e",
    "text_muted":         "#6e7681",
    "accent_start":       "#667eea",
    "accent_end":         "#764ba2",
    "accent_mid":         "#7161c0",
    "success":            "#3fb950",
    "warning":            "#d29922",
    "danger":             "#f85149",
    "info":               "#58a6ff",
    "sidebar_text":       "#e6edf3",
    "sidebar_text_muted": "#6e7681",
    "sidebar_active_bg":  "#30363d",
    "sidebar_hover_bg":   "#292e36",
    "plot_bg":            "#1c2128",
}

# Mutable working palette — mutated in-place so all imports see updates
COLORS: dict = dict(LIGHT_COLORS)
_THEME: list = ["light"]


def current_theme() -> str:
    return _THEME[0]


def set_theme(name: str, app=None) -> None:
    """Switch to 'light' or 'dark'. Pass QApplication to auto-reapply stylesheet."""
    src = LIGHT_COLORS if name == "light" else DARK_COLORS
    COLORS.update(src)
    _THEME[0] = name
    if app is not None:
        app.setStyleSheet(generate_qss())


def toggle_theme(app=None) -> str:
    """Toggle between light and dark; returns the new theme name."""
    new = "dark" if _THEME[0] == "light" else "light"
    set_theme(new, app)
    return new


def generate_qss() -> str:
    """Build QSS from the current COLORS dict."""
    c = COLORS
    ff = FONT_FAMILY

    # Button gradient — blue in light, purple in dark
    btn_grad  = f"qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 {c['accent_start']},stop:1 {c['accent_end']})"
    btn_hover = (
        "qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #3B7FF0,stop:1 #2860D8)"
        if _THEME[0] == "light" else
        "qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #7b8ff0,stop:1 #8a5eb5)"
    )
    btn_press = (
        "qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #1040C0,stop:1 #0D35A0)"
        if _THEME[0] == "light" else
        "qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #5568d4,stop:1 #663f95)"
    )

    return f"""
/* ── Global ─────────────────────────────────────────────── */
QMainWindow, QDialog {{
    background-color: {c["bg_primary"]};
    color: {c["text_primary"]};
    font-family: {ff};
    font-size: 13px;
}}
QWidget {{
    color: {c["text_primary"]};
    font-size: 13px;
    font-family: {ff};
}}

/* ── Menu Bar ────────────────────────────────────────────── */
QMenuBar {{
    background: {c["bg_secondary"]};
    color: {c["text_primary"]};
    border-bottom: 1px solid {c["border"]};
    padding: 2px 0;
    font-size: 13px;
}}
QMenuBar::item:selected {{
    background: {c["bg_hover"]};
    border-radius: 4px;
}}
QMenu {{
    background: {c["bg_secondary"]};
    color: {c["text_primary"]};
    border: 1px solid {c["border"]};
    border-radius: 6px;
    padding: 4px;
}}
QMenu::item {{
    padding: 6px 20px 6px 12px;
    border-radius: 4px;
}}
QMenu::item:selected {{
    background: {c["bg_hover"]};
    color: {c["accent_start"]};
}}

/* ── ToolBar ─────────────────────────────────────────────── */
QToolBar {{
    background: {c["bg_secondary"]};
    border-bottom: 1px solid {c["border"]};
    spacing: 4px;
    padding: 4px 8px;
}}
QToolBar::separator {{
    background: {c["border"]};
    width: 1px;
    margin: 4px 6px;
}}

/* ── Scrollbars ─────────────────────────────────────────── */
QScrollBar:vertical {{
    background: {c["bg_primary"]};
    width: 8px;
    border-radius: 4px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {c["border"]};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {c["text_muted"]}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: {c["bg_primary"]};
    height: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal {{
    background: {c["border"]};
    border-radius: 4px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{ background: {c["text_muted"]}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* ── Tab Widget ─────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {c["border"]};
    border-radius: 8px;
    background: {c["bg_secondary"]};
    margin-top: -1px;
}}
QTabBar::tab {{
    background: {c["bg_primary"]};
    color: {c["text_secondary"]};
    padding: 10px 24px;
    margin-right: 4px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    border: 1px solid {c["border"]};
    border-bottom: none;
    font-weight: 500;
    font-size: 13px;
}}
QTabBar::tab:selected {{
    background: {c["bg_secondary"]};
    color: {c["accent_start"]};
    border-color: {c["border_accent"]};
    border-bottom: 2px solid {c["accent_start"]};
    font-weight: 600;
}}
QTabBar::tab:hover:!selected {{
    background: {c["bg_hover"]};
    color: {c["text_primary"]};
}}

/* ── Push Buttons ───────────────────────────────────────── */
QPushButton {{
    background: {btn_grad};
    color: white;
    border: none;
    padding: 9px 20px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 13px;
    min-height: 18px;
}}
QPushButton:hover {{ background: {btn_hover}; }}
QPushButton:pressed {{ background: {btn_press}; }}
QPushButton:disabled {{
    background: {c["bg_hover"]};
    color: {c["text_muted"]};
}}
QPushButton[class="secondary"] {{
    background: {c["bg_secondary"]};
    border: 1px solid {c["border"]};
    color: {c["text_primary"]};
}}
QPushButton[class="secondary"]:hover {{
    background: {c["bg_hover"]};
    border-color: {c["accent_start"]};
    color: {c["accent_start"]};
}}
QPushButton[class="danger"] {{
    background: {c["danger"]};
    color: white;
    border: none;
}}
QPushButton[class="danger"]:hover {{ background: #b91c1c; }}

/* ── Line Edits & Text Inputs ───────────────────────────── */
QLineEdit, QTextEdit, QPlainTextEdit {{
    background: {c["bg_input"]};
    border: 1px solid {c["border"]};
    border-radius: 6px;
    padding: 8px 12px;
    color: {c["text_primary"]};
    font-size: 13px;
    selection-background-color: {c["accent_start"]};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border-color: {c["accent_start"]};
    outline: none;
}}

/* ── Sliders ────────────────────────────────────────────── */
QSlider::groove:horizontal {{
    border: none;
    height: 6px;
    background: {c["bg_hover"]};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {c["accent_start"]};
    border: none;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}}
QSlider::handle:horizontal:hover {{ background: {c["accent_mid"]}; }}
QSlider::sub-page:horizontal {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {c["accent_start"]}, stop:1 {c["accent_end"]});
    border-radius: 3px;
}}

/* ── Spin Boxes ─────────────────────────────────────────── */
QSpinBox, QDoubleSpinBox {{
    background: {c["bg_input"]};
    border: 1px solid {c["border"]};
    border-radius: 6px;
    padding: 6px 10px;
    color: {c["text_primary"]};
    font-size: 13px;
}}
QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {c["accent_start"]}; }}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background: {c["bg_hover"]};
    border: none;
    width: 20px;
}}

/* ── Combo Boxes ────────────────────────────────────────── */
QComboBox {{
    background: {c["bg_input"]};
    border: 1px solid {c["border"]};
    border-radius: 6px;
    padding: 8px 12px;
    color: {c["text_primary"]};
    font-size: 13px;
    min-width: 120px;
}}
QComboBox:focus {{ border-color: {c["accent_start"]}; }}
QComboBox QAbstractItemView {{
    background: {c["bg_secondary"]};
    border: 1px solid {c["border"]};
    color: {c["text_primary"]};
    selection-background-color: {c["accent_start"]};
    selection-color: white;
    border-radius: 4px;
}}
QComboBox::drop-down {{ border: none; width: 24px; }}

/* ── Labels ─────────────────────────────────────────────── */
QLabel {{
    color: {c["text_primary"]};
    background: transparent;
    border: none;
}}
QLabel[class="heading"] {{
    font-size: 20px;
    font-weight: 700;
    color: {c["text_primary"]};
}}
QLabel[class="subheading"] {{
    font-size: 15px;
    font-weight: 600;
    color: {c["text_secondary"]};
}}
QLabel[class="muted"] {{
    color: {c["text_muted"]};
    font-size: 12px;
}}

/* ── Table Widget ───────────────────────────────────────── */
QTableWidget {{
    background: {c["bg_secondary"]};
    color: {c["text_primary"]};
    gridline-color: {c["border"]};
    border: 1px solid {c["border"]};
    border-radius: 8px;
    font-size: 13px;
    alternate-background-color: {c["bg_primary"]};
}}
QTableWidget::item {{
    padding: 8px 12px;
    border-bottom: 1px solid {c["border"]};
}}
QTableWidget::item:selected {{
    background: rgba(29, 99, 237, 0.12);
    color: {c["text_primary"]};
}}
QHeaderView::section {{
    background: {c["bg_primary"]};
    color: {c["text_secondary"]};
    padding: 10px 12px;
    border: none;
    border-bottom: 2px solid {c["border_accent"]};
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
}}

/* ── Progress Bar ───────────────────────────────────────── */
QProgressBar {{
    background: {c["bg_hover"]};
    border: none;
    border-radius: 4px;
    height: 8px;
    font-size: 0px;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 {c["accent_start"]}, stop:1 {c["accent_end"]});
    border-radius: 4px;
}}

/* ── Dock Widget ────────────────────────────────────────── */
QDockWidget {{
    color: {c["text_primary"]};
    font-weight: 600;
    font-size: 14px;
}}
QDockWidget::title {{
    background: {c["bg_secondary"]};
    padding: 10px 16px;
    border-bottom: 1px solid {c["border"]};
    text-align: left;
}}

/* ── Status Bar ─────────────────────────────────────────── */
QStatusBar {{
    background: {c["bg_secondary"]};
    color: {c["text_secondary"]};
    border-top: 1px solid {c["border"]};
    font-size: 12px;
    padding: 4px 12px;
}}

/* ── Group Box ──────────────────────────────────────────── */
QGroupBox {{
    border: 1px solid {c["border"]};
    border-radius: 8px;
    margin-top: 16px;
    padding: 16px 12px 12px 12px;
    font-weight: 600;
    font-size: 13px;
    color: {c["text_secondary"]};
    background: {c["bg_secondary"]};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 4px 12px;
    background: {c["bg_primary"]};
    border-radius: 4px;
    color: {c["accent_start"]};
}}

/* ── Tool Tips ──────────────────────────────────────────── */
QToolTip {{
    background: {c["bg_secondary"]};
    color: {c["text_primary"]};
    border: 1px solid {c["border"]};
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
}}

/* ── Splitter ───────────────────────────────────────────── */
QSplitter::handle {{
    background: {c["border"]};
}}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}

/* ── List Widget ────────────────────────────────────────── */
QListWidget {{
    background: {c["bg_secondary"]};
    border: 1px solid {c["border"]};
    border-radius: 6px;
    color: {c["text_primary"]};
    font-size: 12px;
}}
QListWidget::item {{
    padding: 6px 10px;
    border-bottom: 1px solid {c["border"]};
}}
QListWidget::item:selected {{
    background: rgba(29, 99, 237, 0.12);
    color: {c["accent_start"]};
}}

/* ── Text Browser ───────────────────────────────────────── */
QTextBrowser {{
    background: {c["bg_input"]};
    border: 1px solid {c["border"]};
    border-radius: 6px;
    color: {c["text_primary"]};
    padding: 8px;
    font-size: 13px;
}}
"""


# Initial stylesheet (light theme)
GLOBAL_QSS: str = generate_qss()

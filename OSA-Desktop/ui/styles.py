# ─────────────────────────────────────────────────────────────
# styles.py — Centralized QSS stylesheet for the OSA Desktop
# Dark theme with gradient purple accents (#667eea → #764ba2)
# ─────────────────────────────────────────────────────────────

# Color tokens
COLORS = {
    "bg_primary":       "#0e1117",
    "bg_secondary":     "#161b22",
    "bg_card":          "#1c2128",
    "bg_sidebar":       "#131720",
    "bg_input":         "#21262d",
    "bg_hover":         "#292e36",
    "border":           "#30363d",
    "border_accent":    "#667eea",
    "text_primary":     "#e6edf3",
    "text_secondary":   "#8b949e",
    "text_muted":       "#6e7681",
    "accent_start":     "#667eea",
    "accent_end":       "#764ba2",
    "accent_mid":       "#7161c0",
    "success":          "#3fb950",
    "warning":          "#d29922",
    "danger":           "#f85149",
    "info":             "#58a6ff",
}

FONT_FAMILY = "'Helvetica Neue', '.AppleSystemUIFont', 'Segoe UI', Arial, sans-serif"

GLOBAL_QSS = f"""
/* ── Global ─────────────────────────────────────────────── */
QMainWindow, QWidget {{
    background-color: {COLORS["bg_primary"]};
    color: {COLORS["text_primary"]};
    font-family: {FONT_FAMILY};
    font-size: 13px;
}}

/* ── Scrollbars ─────────────────────────────────────────── */
QScrollBar:vertical {{
    background: {COLORS["bg_secondary"]};
    width: 8px;
    border-radius: 4px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {COLORS["border"]};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {COLORS["text_muted"]};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: {COLORS["bg_secondary"]};
    height: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal {{
    background: {COLORS["border"]};
    border-radius: 4px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {COLORS["text_muted"]};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ── Tab Widget ─────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
    background: {COLORS["bg_secondary"]};
    margin-top: -1px;
}}
QTabBar::tab {{
    background: {COLORS["bg_secondary"]};
    color: {COLORS["text_secondary"]};
    padding: 10px 24px;
    margin-right: 4px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    border: 1px solid {COLORS["border"]};
    border-bottom: none;
    font-weight: 500;
    font-size: 13px;
}}
QTabBar::tab:selected {{
    background: {COLORS["bg_card"]};
    color: {COLORS["accent_start"]};
    border-color: {COLORS["border_accent"]};
    border-bottom: 2px solid {COLORS["accent_start"]};
    font-weight: 600;
}}
QTabBar::tab:hover:!selected {{
    background: {COLORS["bg_hover"]};
    color: {COLORS["text_primary"]};
}}

/* ── Push Buttons ───────────────────────────────────────── */
QPushButton {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {COLORS["accent_start"]}, stop:1 {COLORS["accent_end"]});
    color: white;
    border: none;
    padding: 10px 22px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 13px;
    min-height: 18px;
}}
QPushButton:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #7b8ff0, stop:1 #8a5eb5);
}}
QPushButton:pressed {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #5568d4, stop:1 #663f95);
}}
QPushButton:disabled {{
    background: {COLORS["bg_hover"]};
    color: {COLORS["text_muted"]};
}}
QPushButton[class="secondary"] {{
    background: {COLORS["bg_input"]};
    border: 1px solid {COLORS["border"]};
    color: {COLORS["text_primary"]};
}}
QPushButton[class="secondary"]:hover {{
    background: {COLORS["bg_hover"]};
    border-color: {COLORS["accent_start"]};
}}
QPushButton[class="danger"] {{
    background: {COLORS["danger"]};
}}
QPushButton[class="danger"]:hover {{
    background: #e5443c;
}}

/* ── Line Edits & Text Inputs ───────────────────────────── */
QLineEdit, QTextEdit, QPlainTextEdit {{
    background: {COLORS["bg_input"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 6px;
    padding: 8px 12px;
    color: {COLORS["text_primary"]};
    font-size: 13px;
    selection-background-color: {COLORS["accent_start"]};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border-color: {COLORS["accent_start"]};
}}

/* ── Sliders ────────────────────────────────────────────── */
QSlider::groove:horizontal {{
    border: none;
    height: 6px;
    background: {COLORS["bg_input"]};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {COLORS["accent_start"]};
    border: none;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}}
QSlider::handle:horizontal:hover {{
    background: {COLORS["accent_mid"]};
}}
QSlider::sub-page:horizontal {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {COLORS["accent_start"]}, stop:1 {COLORS["accent_end"]});
    border-radius: 3px;
}}

/* ── Spin Boxes ─────────────────────────────────────────── */
QSpinBox, QDoubleSpinBox {{
    background: {COLORS["bg_input"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 6px;
    padding: 6px 10px;
    color: {COLORS["text_primary"]};
    font-size: 13px;
}}
QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {COLORS["accent_start"]};
}}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background: {COLORS["bg_hover"]};
    border: none;
    width: 20px;
}}

/* ── Combo Boxes ────────────────────────────────────────── */
QComboBox {{
    background: {COLORS["bg_input"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 6px;
    padding: 8px 12px;
    color: {COLORS["text_primary"]};
    font-size: 13px;
    min-width: 120px;
}}
QComboBox:focus {{
    border-color: {COLORS["accent_start"]};
}}
QComboBox QAbstractItemView {{
    background: {COLORS["bg_card"]};
    border: 1px solid {COLORS["border"]};
    color: {COLORS["text_primary"]};
    selection-background-color: {COLORS["accent_start"]};
    border-radius: 4px;
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}

/* ── Labels / Headings ──────────────────────────────────── */
QLabel {{
    color: {COLORS["text_primary"]};
    background: transparent;
    border: none;
}}
QLabel[class="heading"] {{
    font-size: 20px;
    font-weight: 700;
    color: {COLORS["text_primary"]};
}}
QLabel[class="subheading"] {{
    font-size: 15px;
    font-weight: 600;
    color: {COLORS["text_secondary"]};
}}
QLabel[class="muted"] {{
    color: {COLORS["text_muted"]};
    font-size: 12px;
}}

/* ── Table Widget ───────────────────────────────────────── */
QTableWidget {{
    background: {COLORS["bg_card"]};
    color: {COLORS["text_primary"]};
    gridline-color: {COLORS["border"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
    font-size: 13px;
}}
QTableWidget::item {{
    padding: 8px 12px;
    border-bottom: 1px solid {COLORS["border"]};
}}
QTableWidget::item:selected {{
    background: rgba(102, 126, 234, 0.2);
    color: {COLORS["text_primary"]};
}}
QHeaderView::section {{
    background: {COLORS["bg_secondary"]};
    color: {COLORS["text_secondary"]};
    padding: 10px 12px;
    border: none;
    border-bottom: 2px solid {COLORS["border_accent"]};
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
}}

/* ── Progress Bar ───────────────────────────────────────── */
QProgressBar {{
    background: {COLORS["bg_input"]};
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
    font-size: 0px;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {COLORS["accent_start"]}, stop:1 {COLORS["accent_end"]});
    border-radius: 4px;
}}

/* ── Dock Widget (Sidebar) ──────────────────────────────── */
QDockWidget {{
    titlebar-close-icon: none;
    color: {COLORS["text_primary"]};
    font-weight: 600;
    font-size: 14px;
}}
QDockWidget::title {{
    background: {COLORS["bg_sidebar"]};
    padding: 12px 16px;
    border-bottom: 1px solid {COLORS["border"]};
    text-align: left;
}}

/* ── Status Bar ─────────────────────────────────────────── */
QStatusBar {{
    background: {COLORS["bg_secondary"]};
    color: {COLORS["text_secondary"]};
    border-top: 1px solid {COLORS["border"]};
    font-size: 12px;
    padding: 4px 12px;
}}

/* ── Group Box ──────────────────────────────────────────── */
QGroupBox {{
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
    margin-top: 16px;
    padding: 16px 12px 12px 12px;
    font-weight: 600;
    font-size: 13px;
    color: {COLORS["text_secondary"]};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 4px 12px;
    background: {COLORS["bg_primary"]};
    border-radius: 4px;
    color: {COLORS["accent_start"]};
}}

/* ── Tool Tips ──────────────────────────────────────────── */
QToolTip {{
    background: {COLORS["bg_card"]};
    color: {COLORS["text_primary"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 12px;
}}

/* ── File Dialog ────────────────────────────────────────── */
QFileDialog {{
    background: {COLORS["bg_primary"]};
}}
"""

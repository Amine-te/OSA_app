# ─────────────────────────────────────────────────────────────
# sidebar.py — Configuration sidebar panel for OSA Desktop
# Collapsible sidebar matching Streamlit's ⚙️ Configuration
# ─────────────────────────────────────────────────────────────

import json
import platform
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QSlider, QGroupBox,
    QFileDialog, QScrollArea, QSizePolicy, QSpacerItem
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

try:
    from PyQt6.QtWidgets import QGraphicsBlurEffect
except ImportError:
    QGraphicsBlurEffect = None

from ui.styles import COLORS


class SidebarPanel(QWidget):
    """Configuration sidebar matching Streamlit's sidebar layout."""

    # Signal emitted when user clicks Initialize Pipeline
    initialize_requested = pyqtSignal(dict)

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self.setFixedWidth(320)
        self.setObjectName("sidebar_panel")

        # Glassmorphism styling
        if platform.system() == 'Darwin':
            self.setStyleSheet(f"""
                QWidget#sidebar_panel {{
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 rgba(19, 23, 32, 242),
                        stop:0.3 rgba(22, 27, 34, 238),
                        stop:0.7 rgba(19, 23, 32, 240),
                        stop:1 rgba(16, 20, 28, 245));
                    border-right: 1px solid rgba(102, 126, 234, 0.12);
                }}
                QWidget {{
                    background: transparent;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QWidget {{
                    background: {COLORS['bg_sidebar']};
                }}
            """)

        # Scroll area for long content
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background: {COLORS["bg_sidebar"]};
            }}
        """)

        container = QWidget()
        self.layout_main = QVBoxLayout(container)
        self.layout_main.setContentsMargins(16, 16, 16, 16)
        self.layout_main.setSpacing(12)

        # ── Header ──
        hdr = QLabel("⚙️  Configuration")
        hdr.setStyleSheet(
            f"font-size: 18px; font-weight: 700; color: {COLORS['text_primary']}; "
            f"padding-bottom: 8px; border-bottom: 1px solid {COLORS['border']}; background: transparent;"
        )
        self.layout_main.addWidget(hdr)

        # ── Model Paths Section ──
        self._add_section("Model Paths")

        self.input_yolo = self._add_path_field(
            "YOLO Product Model",
            config.get("models", {}).get("yolo_product", ""),
            "Model Files (*.pt *.pth)"
        )
        self.input_cnn = self._add_path_field(
            "CNN Classifier Model",
            config.get("models", {}).get("cnn_class", ""),
            "Model Files (*.pt *.pth)"
        )
        self.input_void = self._add_path_field(
            "Void Detection Model",
            config.get("models", {}).get("yolo_void", ""),
            "Model Files (*.pt *.pth)"
        )

        # ── Class Names Section ──
        self._add_section("Product Classes")

        class_names = config.get("class_names", [])
        class_str = ", ".join(class_names) if class_names else ""

        info_lbl = QLabel(f"📋 {len(class_names)} classes loaded from config")
        info_lbl.setStyleSheet(
            f"font-size: 11px; color: {COLORS['info']}; padding: 4px 0; background: transparent;"
        )
        self.layout_main.addWidget(info_lbl)

        self.input_classes = QLineEdit(class_str)
        self.input_classes.setPlaceholderText("e.g. cocacola, oil, water")
        self.input_classes.setToolTip("Comma-separated product class names")
        self.layout_main.addWidget(self.input_classes)

        # ── Thresholds Section ──
        self._add_section("Detection Thresholds")

        conf = config.get("thresholds", {}).get("confidence", 0.5)
        self.slider_conf, self.lbl_conf_val = self._add_slider(
            "Product Detection Confidence", int(conf * 100), 10, 100
        )

        void_conf = config.get("thresholds", {}).get("void_confidence", 0.5)
        self.slider_void, self.lbl_void_val = self._add_slider(
            "Void Detection Confidence", int(void_conf * 100), 10, 100
        )

        # ── Initialize Button ──
        self.layout_main.addSpacerItem(QSpacerItem(1, 20))

        self.btn_init = QPushButton("🚀  Initialize Pipeline")
        self.btn_init.setMinimumHeight(44)
        self.btn_init.setStyleSheet(f"""
            QPushButton {{
                font-size: 14px;
                font-weight: 700;
            }}
        """)
        self.btn_init.clicked.connect(self._on_init_clicked)
        self.layout_main.addWidget(self.btn_init)

        # Status label
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet(
            f"font-size: 12px; color: {COLORS['text_muted']}; padding: 4px 0; background: transparent;"
        )
        self.lbl_status.setWordWrap(True)
        self.layout_main.addWidget(self.lbl_status)

        # push everything up
        self.layout_main.addStretch()

        # ── Footer ──
        footer = QLabel("🛒 OSA Desktop v2.0")
        footer.setStyleSheet(
            f"font-size: 11px; color: {COLORS['text_muted']}; padding: 8px 0; background: transparent;"
        )
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout_main.addWidget(footer)

        scroll.setWidget(container)
        outer.addWidget(scroll)

    # ── Helpers ──────────────────────────────────────────────

    def _add_section(self, title):
        lbl = QLabel(title)
        lbl.setStyleSheet(
            f"font-size: 13px; font-weight: 600; color: {COLORS['accent_start']}; "
            f"padding-top: 12px; padding-bottom: 4px; background: transparent;"
        )
        self.layout_main.addWidget(lbl)

    def _add_path_field(self, label, default_value, filter_str):
        lbl = QLabel(label)
        lbl.setStyleSheet(
            f"font-size: 12px; color: {COLORS['text_secondary']}; padding: 2px 0; background: transparent;"
        )
        self.layout_main.addWidget(lbl)

        row = QHBoxLayout()
        row.setSpacing(6)
        inp = QLineEdit(default_value)
        inp.setToolTip(label)
        
        btn = QPushButton("📂")
        btn.setFixedSize(36, 34)
        btn.setProperty("class", "secondary")
        btn.setToolTip(f"Browse for {label}")
        btn.clicked.connect(lambda: self._browse_file(inp, filter_str))

        row.addWidget(inp)
        row.addWidget(btn)
        self.layout_main.addLayout(row)
        return inp

    def _add_slider(self, label, default, min_val, max_val):
        lbl = QLabel(label)
        lbl.setStyleSheet(
            f"font-size: 12px; color: {COLORS['text_secondary']}; padding: 2px 0; background: transparent;"
        )
        self.layout_main.addWidget(lbl)

        row = QHBoxLayout()
        row.setSpacing(10)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        slider.setTickInterval(10)

        val_lbl = QLabel(f"{default / 100:.2f}")
        val_lbl.setFixedWidth(40)
        val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        val_lbl.setStyleSheet(
            f"font-size: 13px; font-weight: 600; color: {COLORS['accent_start']}; background: transparent;"
        )

        slider.valueChanged.connect(lambda v: val_lbl.setText(f"{v / 100:.2f}"))

        row.addWidget(slider)
        row.addWidget(val_lbl)
        self.layout_main.addLayout(row)
        return slider, val_lbl

    def _browse_file(self, line_edit, filter_str):
        path, _ = QFileDialog.getOpenFileName(self, "Select File", "", filter_str)
        if path:
            line_edit.setText(path)

    def _on_init_clicked(self):
        """Gather current config and emit signal."""
        class_text = self.input_classes.text()
        class_names = [n.strip() for n in class_text.split(",") if n.strip()]

        config = {
            "models": {
                "yolo_product": self.input_yolo.text(),
                "cnn_class": self.input_cnn.text(),
                "yolo_void": self.input_void.text(),
            },
            "class_names": class_names,
            "thresholds": {
                "confidence": self.slider_conf.value() / 100.0,
                "void_confidence": self.slider_void.value() / 100.0,
            },
            "ui": self.config.get("ui", {}),
        }
        self.initialize_requested.emit(config)

    def set_status(self, text, is_error=False):
        color = COLORS["danger"] if is_error else COLORS["success"]
        self.lbl_status.setStyleSheet(
            f"font-size: 12px; color: {color}; padding: 4px 0; background: transparent;"
        )
        self.lbl_status.setText(text)

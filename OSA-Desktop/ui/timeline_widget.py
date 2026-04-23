"""Video frame scrubber with anomaly markers (missing / low stock)."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSlider, QWidget

from ui.styles import COLORS


class VideoTimeline(QWidget):
    """
    Horizontal timeline: slider + optional markers for frames with issues.
    marker_types: list parallel to frame indices — 'missing', 'anomaly', or ''.
    """

    frame_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._markers = []  # per-frame tag
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider)
        self._lbl = QLabel("Frame —")
        self._lbl.setMinimumWidth(120)
        self._lbl.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(self._slider, stretch=1)
        row.addWidget(self._lbl)

    def _on_slider(self, v):
        self._lbl.setText(f"Frame {v + 1}")
        self.frame_changed.emit(v)

    def set_frame_count(self, n: int):
        self._slider.blockSignals(True)
        self._slider.setMaximum(max(0, n - 1))
        self._slider.setValue(0)
        self._slider.blockSignals(False)
        self._lbl.setText(f"Frame {1}" if n else "Frame —")

    def set_value(self, idx: int):
        self._slider.blockSignals(True)
        self._slider.setValue(max(0, min(idx, self._slider.maximum())))
        self._slider.blockSignals(False)
        self._lbl.setText(f"Frame {idx + 1}")

    def set_markers(self, markers: list):
        """markers[i] in ('missing','anomaly','')"""
        self._markers = list(markers)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._markers or self._slider.maximum() <= 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        groove = self._slider.geometry()
        if groove.width() <= 0:
            painter.end()
            return
        n = len(self._markers)
        max_idx = self._slider.maximum()
        for i, tag in enumerate(self._markers):
            if not tag or i > max_idx:
                continue
            x = groove.left() + (i / max(max_idx, 1)) * groove.width()
            color = QColor(COLORS["danger"]) if tag == "missing" else QColor(COLORS["warning"])
            painter.setPen(QPen(color, 2))
            painter.drawLine(int(x), groove.top() - 4, int(x), groove.bottom() + 4)
        painter.end()

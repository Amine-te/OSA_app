"""Live FPS and latency trend for the performance dock."""

import time
from collections import deque

import pyqtgraph as pg
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ui.styles import COLORS


class PerformancePanel(QWidget):
    """Compact pyqtgraph strip + device line (multi-source ready)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fps_buf = deque(maxlen=120)
        self._lat_buf = deque(maxlen=120)
        self._t0 = time.time()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self.device_lbl = QLabel("Device: —")
        self.device_lbl.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        lay.addWidget(self.device_lbl)

        self.plot = pg.PlotWidget()
        self.plot.setBackground(COLORS.get("plot_bg", COLORS["bg_card"]))
        self.plot.setMinimumHeight(140)
        self.plot.showGrid(x=True, y=True, alpha=0.12)
        self.plot.setLabel("left", "FPS / latency (ms)", color=COLORS["text_muted"])
        self.plot.setLabel("bottom", "Time (s)", color=COLORS["text_muted"])
        self.fps_curve = self.plot.plot(
            pen=pg.mkPen(color=COLORS["success"], width=2), name="FPS"
        )
        self.lat_curve = self.plot.plot(
            pen=pg.mkPen(color=COLORS["accent_start"], width=2), name="Latency"
        )
        lay.addWidget(self.plot)

        self.sources_note = QLabel("Sources: 1 (expandable to multi-camera grid)")
        self.sources_note.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 10px;")
        self.sources_note.setWordWrap(True)
        lay.addWidget(self.sources_note)

    def set_device(self, device: str):
        self.device_lbl.setText(f"Device: {device}")

    def push_sample(self, fps: float, latency_ms: float):
        t = time.time() - self._t0
        self._fps_buf.append((t, fps))
        self._lat_buf.append((t, latency_ms))
        xs = [a[0] for a in self._fps_buf]
        self.fps_curve.setData(xs, [a[1] for a in self._fps_buf])
        self.lat_curve.setData(xs, [a[1] for a in self._lat_buf])

    def reset(self):
        self._fps_buf.clear()
        self._lat_buf.clear()
        self._t0 = time.time()
        self.fps_curve.setData([], [])
        self.lat_curve.setData([], [])

    def apply_theme(self) -> None:
        """Refresh plot background after a theme switch."""
        self.plot.setBackground(COLORS.get("plot_bg", COLORS["bg_card"]))

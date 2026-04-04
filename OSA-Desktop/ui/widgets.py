# ─────────────────────────────────────────────────────────────
# widgets.py — Professional custom widgets for OSA Desktop
# Gauge, MetricCard, StockTable, GradientHeader, ExportBar
# ─────────────────────────────────────────────────────────────

import math
import json
import os
from datetime import datetime

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QPushButton,
    QHeaderView, QSizePolicy, QFileDialog, QGraphicsDropShadowEffect
)
from PyQt6.QtCore import Qt, QRectF, QTimer, pyqtProperty, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import (
    QPainter, QPen, QColor, QConicalGradient,
    QRadialGradient, QLinearGradient, QFont, QBrush, QPainterPath
)

import pyqtgraph as pg
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ui.styles import COLORS


# ── Gradient Header Banner ──────────────────────────────────

class GradientHeader(QWidget):
    """Painted gradient banner matching Streamlit's #667eea → #764ba2."""

    def __init__(self, title="🛒 Intelligent Retail Shelf Analysis",
                 subtitle="AI-powered product detection · inventory tracking · restocking intelligence",
                 parent=None):
        super().__init__(parent)
        self.title = title
        self.subtitle = subtitle
        self.setFixedHeight(100)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()

        # Gradient background
        grad = QLinearGradient(0, 0, rect.width(), 0)
        grad.setColorAt(0.0, QColor("#667eea"))
        grad.setColorAt(0.5, QColor("#7161c0"))
        grad.setColorAt(1.0, QColor("#764ba2"))
        painter.setBrush(QBrush(grad))
        painter.setPen(Qt.PenStyle.NoPen)

        path = QPainterPath()
        path.addRoundedRect(QRectF(rect), 12, 12)
        painter.drawPath(path)

        # Subtle overlay dots pattern
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 8))
        for x in range(0, rect.width(), 20):
            for y in range(0, rect.height(), 20):
                painter.drawEllipse(x, y, 2, 2)

        # Title text
        painter.setPen(QColor("white"))
        title_font = QFont("Helvetica Neue", 22, QFont.Weight.Bold)
        painter.setFont(title_font)
        title_rect = QRectF(rect.x(), rect.y() + 16, rect.width(), 40)
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignCenter, self.title)

        # Subtitle text
        painter.setPen(QColor(255, 255, 255, 180))
        sub_font = QFont("Helvetica Neue", 11, QFont.Weight.Normal)
        painter.setFont(sub_font)
        sub_rect = QRectF(rect.x(), rect.y() + 56, rect.width(), 24)
        painter.drawText(sub_rect, Qt.AlignmentFlag.AlignCenter, self.subtitle)

        painter.end()


# ── Metric Card ─────────────────────────────────────────────

class MetricCard(QWidget):
    """Card displaying an icon, title, value, and color-coded left border."""

    def __init__(self, icon="📦", title="Metric", value="0",
                 accent_color=None, parent=None):
        super().__init__(parent)
        self._accent = accent_color or COLORS["accent_start"]
        self.setFixedHeight(110)
        self.setMinimumWidth(180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 14, 16, 14)
        layout.setSpacing(4)

        # Icon + title row
        top = QHBoxLayout()
        top.setSpacing(8)
        icon_lbl = QLabel(icon)
        icon_lbl.setStyleSheet("font-size: 22px; background: transparent;")
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(
            f"font-size: 12px; font-weight: 600; color: {COLORS['text_secondary']}; "
            f"text-transform: uppercase; letter-spacing: 1px; background: transparent;"
        )
        top.addWidget(icon_lbl)
        top.addWidget(title_lbl)
        top.addStretch()

        # Value
        self.value_lbl = QLabel(str(value))
        self.value_lbl.setStyleSheet(
            f"font-size: 28px; font-weight: 700; color: {COLORS['text_primary']}; "
            f"background: transparent;"
        )

        layout.addLayout(top)
        layout.addWidget(self.value_lbl)

    def set_value(self, value):
        self.value_lbl.setText(str(value))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect())

        # Card background
        bg_path = QPainterPath()
        bg_path.addRoundedRect(rect.adjusted(1, 1, -1, -1), 10, 10)
        painter.setPen(QPen(QColor(COLORS["border"]), 1))
        painter.setBrush(QColor(COLORS["bg_card"]))
        painter.drawPath(bg_path)

        # Left accent bar
        accent_path = QPainterPath()
        accent_rect = QRectF(0, 8, 4, rect.height() - 16)
        accent_path.addRoundedRect(accent_rect, 2, 2)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(self._accent))
        painter.drawPath(accent_path)

        painter.end()
        super().paintEvent(event)


# ── Gauge Widget ────────────────────────────────────────────

class GaugeWidget(QWidget):
    """Circular gauge with gradient arc, threshold markers."""

    def __init__(self, title="Stock", value=0, parent=None):
        super().__init__(parent)
        self._title = title
        self._value = 0.0
        self._target_value = float(value)
        self.setMinimumSize(180, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Animate value
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._animate_step)

        if value > 0:
            self._timer.start()

    def set_value(self, value):
        self._target_value = float(value)
        self._timer.start()

    def _animate_step(self):
        diff = self._target_value - self._value
        if abs(diff) < 0.5:
            self._value = self._target_value
            self._timer.stop()
        else:
            self._value += diff * 0.08
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        side = min(w, h - 36)
        radius = side * 0.42
        cx = w / 2
        cy = h / 2 - 8

        pen_width = max(10, side * 0.08)

        # Background arc (track)
        pen = QPen(QColor(COLORS["bg_input"]), pen_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        arc_rect = QRectF(cx - radius, cy - radius, radius * 2, radius * 2)
        start_angle = 225 * 16
        span_angle = -270 * 16
        painter.drawArc(arc_rect, start_angle, span_angle)

        # Value arc with gradient color
        value_span = int(-270 * (self._value / 100.0)) * 16
        if self._value >= 90:
            arc_color = QColor(COLORS["success"])
        elif self._value >= 70:
            arc_color = QColor(COLORS["warning"])
        else:
            arc_color = QColor(COLORS["danger"])

        pen.setColor(arc_color)
        painter.setPen(pen)
        painter.drawArc(arc_rect, start_angle, value_span)

        # Threshold markers (small ticks at 70% and 90%)
        painter.setPen(QPen(QColor(COLORS["text_muted"]), 1))
        for threshold in [70, 90]:
            angle_deg = 225 - (threshold / 100.0) * 270
            angle_rad = math.radians(angle_deg)
            inner_r = radius - pen_width / 2 - 4
            outer_r = radius + pen_width / 2 + 4
            x1 = cx + inner_r * math.cos(angle_rad)
            y1 = cy - inner_r * math.sin(angle_rad)
            x2 = cx + outer_r * math.cos(angle_rad)
            y2 = cy - outer_r * math.sin(angle_rad)
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # Center value text
        painter.setPen(QColor(COLORS["text_primary"]))
        val_font = QFont("Helvetica Neue", max(1, max(18, int(side * 0.14))), QFont.Weight.Bold)
        painter.setFont(val_font)
        painter.drawText(QRectF(cx - radius, cy - 16, radius * 2, 36),
                         Qt.AlignmentFlag.AlignCenter,
                         f"{self._value:.0f}%")

        # Status label
        if self._value >= 90:
            status = "GOOD"
            status_color = COLORS["success"]
        elif self._value >= 70:
            status = "MODERATE"
            status_color = COLORS["warning"]
        else:
            status = "LOW"
            status_color = COLORS["danger"]

        painter.setPen(QColor(status_color))
        status_font = QFont("Helvetica Neue", max(1, max(8, int(side * 0.06))), QFont.Weight.DemiBold)
        painter.setFont(status_font)
        painter.drawText(QRectF(cx - radius, cy + 18, radius * 2, 20),
                         Qt.AlignmentFlag.AlignCenter, status)

        # Title below gauge
        painter.setPen(QColor(COLORS["text_secondary"]))
        title_font = QFont("Helvetica Neue", max(1, max(10, int(side * 0.08))), QFont.Weight.DemiBold)
        painter.setFont(title_font)
        painter.drawText(QRectF(0, h - 30, w, 24),
                         Qt.AlignmentFlag.AlignCenter, self._title)

        painter.end()


# ── Stock Table ─────────────────────────────────────────────

class StockTable(QTableWidget):
    """Styled product summary table with color-coded status."""

    HEADERS = ["Product", "Current", "Missing", "Capacity", "Stock %", "Status"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(len(self.HEADERS))
        self.setHorizontalHeaderLabels(self.HEADERS)
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.setStyleSheet(f"""
            QTableWidget {{
                alternate-background-color: {COLORS["bg_secondary"]};
            }}
        """)
        self.setMinimumHeight(300)

    def load_data(self, stock_levels: dict):
        """Populate table from pipeline summary stock_levels dict."""
        self.setRowCount(len(stock_levels))
        for row, (product, data) in enumerate(stock_levels.items()):
            current = data.get("current_count", 0)
            missing = data.get("missing_count", 0)
            capacity = data.get("full_capacity", current + missing)
            pct = data.get("stock_percentage", 0)

            if pct >= 90:
                status = "🟢 GOOD"
                color = QColor(COLORS["success"])
            elif pct >= 70:
                status = "🟡 MODERATE"
                color = QColor(COLORS["warning"])
            else:
                status = "🔴 LOW"
                color = QColor(COLORS["danger"])

            items = [
                product.title(),
                str(current),
                str(missing),
                str(capacity),
                f"{pct:.1f}%",
                status,
            ]
            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if col == 5:
                    item.setForeground(QBrush(color))
                self.setItem(row, col, item)


# ── Export Bar ──────────────────────────────────────────────

class ExportBar(QWidget):
    """Row of export buttons: JSON, CSV, Text Report."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(12)

        lbl = QLabel("📤 Export Results")
        lbl.setStyleSheet(f"font-weight: 600; color: {COLORS['text_secondary']}; font-size: 13px;")
        layout.addWidget(lbl)
        layout.addStretch()

        self.btn_json = QPushButton("💾 JSON")
        self.btn_json.setProperty("class", "secondary")
        self.btn_json.clicked.connect(self._export_json)

        self.btn_csv = QPushButton("📊 CSV")
        self.btn_csv.setProperty("class", "secondary")
        self.btn_csv.clicked.connect(self._export_csv)

        self.btn_pdf = QPushButton("📑 PDF")
        self.btn_pdf.setProperty("class", "secondary")
        self.btn_pdf.clicked.connect(self._export_pdf)

        self.btn_report = QPushButton("📄 Report")
        self.btn_report.setProperty("class", "secondary")
        self.btn_report.clicked.connect(self._export_report)

        for btn in [self.btn_json, self.btn_csv, self.btn_pdf, self.btn_report]:
            btn.setFixedHeight(36)
            btn.setMinimumWidth(80)
            layout.addWidget(btn)
        
        self.setEnabled(False)

    def set_results(self, results):
        self._results = results
        self.setEnabled(results is not None)

    def _export_json(self):
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save JSON",
            f"shelf_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json)")
        if path:
            with open(path, 'w') as f:
                json.dump(self._results, f, indent=2, default=str)

    def _export_csv(self):
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV",
            f"shelf_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV Files (*.csv)")
        if path:
            summary = self._results.get('summary', {})
            import csv
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Product", "Current", "Missing", "Capacity", "Stock%"])
                for product, data in summary.get('stock_levels', {}).items():
                    current = data.get('current_count', 0)
                    missing = data.get('missing_count', 0)
                    cap = data.get('full_capacity', current + missing)
                    pct = data.get('stock_percentage', 0)
                    writer.writerow([product.title(), current, missing, cap, f"{pct:.1f}"])

    def _export_report(self):
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Report",
            f"shelf_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt)")
        if path:
            summary = self._results.get('summary', {})
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            lines = [
                "INTELLIGENT SHELF ANALYSIS REPORT",
                "=" * 50,
                f"Generated: {ts}",
                "",
                "OVERVIEW:",
                f"  Total Products: {summary.get('total_products_detected', 0)}",
                f"  Overall Stock:  {summary.get('overall_stock_percentage', 0):.1f}%",
                "",
                "PRODUCT INVENTORY:",
            ]
            for product, data in summary.get('stock_levels', {}).items():
                pct = data.get('stock_percentage', 0)
                status = "GOOD" if pct >= 90 else "MODERATE" if pct >= 70 else "LOW"
                current = data.get('current_count', 0)
                lines.append(f"  • {product.title()}: {current} items ({pct:.1f}% — {status})")
            with open(path, 'w') as f:
                f.write("\n".join(lines))

    def _export_pdf(self):
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF",
            f"shelf_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            "PDF Files (*.pdf)")
        if path:
            from PyQt6.QtPrintSupport import QPrinter
            from PyQt6.QtGui import QTextDocument
            summary = self._results.get('summary', {})
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            html = f"""
            <html><body style="font-family: Helvetica, sans-serif; color: #333;">
            <h2 style="color: #2c3e50;">INTELLIGENT SHELF ANALYSIS REPORT</h2>
            <hr style="border: 1px solid #ddd;">
            <p><strong>Generated:</strong> {ts}</p>
            
            <h3 style="color: #2c3e50; margin-top: 20px;">OVERVIEW</h3>
            <ul>
                <li><strong>Total Products:</strong> {summary.get('total_products_detected', 0)}</li>
                <li><strong>Overall Stock:</strong> {summary.get('overall_stock_percentage', 0):.1f}%</li>
            </ul>
            
            <h3 style="color: #2c3e50; margin-top: 20px;">PRODUCT INVENTORY</h3>
            <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
            <tr style="background-color: #f8f9fa;">
                <th style="padding: 10px; border: 1px solid #ddd; text-align: left;">Product</th>
                <th style="padding: 10px; border: 1px solid #ddd; text-align: center;">Current</th>
                <th style="padding: 10px; border: 1px solid #ddd; text-align: center;">Missing</th>
                <th style="padding: 10px; border: 1px solid #ddd; text-align: center;">Capacity</th>
                <th style="padding: 10px; border: 1px solid #ddd; text-align: center;">Stock %</th>
                <th style="padding: 10px; border: 1px solid #ddd; text-align: center;">Status</th>
            </tr>
            """
            
            for product, data in summary.get('stock_levels', {}).items():
                pct = data.get('stock_percentage', 0)
                status = "GOOD" if pct >= 90 else "MODERATE" if pct >= 70 else "LOW"
                current = data.get('current_count', 0)
                missing = data.get('missing_count', 0)
                cap = data.get('full_capacity', current + missing)
                
                status_color = "#28a745" if pct >= 90 else "#ffc107" if pct >= 70 else "#dc3545"
                
                html += f"""
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;">{product.title()}</td>
                    <td style="padding: 8px; border: 1px solid #ddd; text-align: center;">{current}</td>
                    <td style="padding: 8px; border: 1px solid #ddd; text-align: center;">{missing}</td>
                    <td style="padding: 8px; border: 1px solid #ddd; text-align: center;">{cap}</td>
                    <td style="padding: 8px; border: 1px solid #ddd; text-align: center;">{pct:.1f}%</td>
                    <td style="padding: 8px; border: 1px solid #ddd; text-align: center; color: {status_color}; font-weight: bold;">{status}</td>
                </tr>
                """
            
            html += """
            </table>
            <p style="margin-top: 40px; font-size: 10px; color: #777; text-align: center;">
                Generated by OSA Desktop Industrial Control Center
            </p>
            </body></html>
            """
            
            doc = QTextDocument()
            doc.setHtml(html)
            
            printer = QPrinter(QPrinter.PrinterMode.HighResolution)
            printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
            printer.setOutputFileName(path)
            
            doc.print(printer)



# ── Real-Time Plot Widget (dark theme) ──────────────────────

class RealTimePlotWidget(QWidget):
    """PyQtGraph widget for high-performance live meter updates."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        self.plot_widget.setBackground(COLORS["bg_card"])
        self.plot_widget.setTitle("Live Stock Level (%)",
                                  color=COLORS["text_secondary"], size="11pt")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self.plot_widget.setYRange(0, 100)
        self.plot_widget.getAxis("left").setPen(pg.mkPen(color=COLORS["text_muted"]))
        self.plot_widget.getAxis("bottom").setPen(pg.mkPen(color=COLORS["text_muted"]))
        self.plot_widget.getAxis("left").setTextPen(pg.mkPen(color=COLORS["text_secondary"]))
        self.plot_widget.getAxis("bottom").setTextPen(pg.mkPen(color=COLORS["text_secondary"]))

        # Threshold lines
        self.plot_widget.addLine(y=90, pen=pg.mkPen(color=COLORS["success"], style=Qt.PenStyle.DashLine, width=1))
        self.plot_widget.addLine(y=70, pen=pg.mkPen(color=COLORS["warning"], style=Qt.PenStyle.DashLine, width=1))

        self.timestamps = []
        self.stock_levels = []
        self.data_line = self.plot_widget.plot(
            self.timestamps, self.stock_levels,
            pen=pg.mkPen(color=COLORS["accent_start"], width=2),
            symbolBrush=QColor(COLORS["accent_start"]),
            symbolPen=None,
            symbolSize=6,
        )

    def update_plot(self, timestamp_sec, stock_level):
        self.timestamps.append(timestamp_sec)
        self.stock_levels.append(stock_level)
        if len(self.timestamps) > 200:
            self.timestamps = self.timestamps[-200:]
            self.stock_levels = self.stock_levels[-200:]
        self.data_line.setData(self.timestamps, self.stock_levels)

    def clear_plot(self):
        self.timestamps.clear()
        self.stock_levels.clear()
        self.data_line.setData([], [])


# ── Static Report Canvas (dark theme) ──────────────────────

class StaticReportCanvas(QWidget):
    """Matplotlib canvas for heavy, static chart generation."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.set_facecolor(COLORS["bg_card"])
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(COLORS["bg_card"])
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

    def plot_summary(self, summary_data):
        self.ax.clear()
        self.ax.set_facecolor(COLORS["bg_card"])

        products = list(summary_data.get('stock_levels', {}).keys())
        stock_pcts = [d.get('stock_percentage', 0)
                      for d in summary_data.get('stock_levels', {}).values()]

        if products:
            colors = [COLORS["success"] if p >= 90 else COLORS["warning"] if p >= 70 else COLORS["danger"]
                      for p in stock_pcts]
            bars = self.ax.bar([p.title() for p in products], stock_pcts, color=colors, width=0.6,
                               edgecolor=COLORS["border"], linewidth=0.5)
            self.ax.set_ylim(0, 105)
            self.ax.set_ylabel('Stock Level (%)', color=COLORS["text_secondary"], fontsize=10)
            self.ax.set_title('Product Stock Summary', color=COLORS["text_primary"],
                              fontsize=13, fontweight='bold', pad=12)
            self.ax.axhline(y=90, color=COLORS["success"], linestyle='--', linewidth=0.8, alpha=0.6)
            self.ax.axhline(y=70, color=COLORS["warning"], linestyle='--', linewidth=0.8, alpha=0.6)
            self.ax.tick_params(colors=COLORS["text_secondary"], labelsize=9)
            self.ax.spines['top'].set_visible(False)
            self.ax.spines['right'].set_visible(False)
            self.ax.spines['left'].set_color(COLORS["border"])
            self.ax.spines['bottom'].set_color(COLORS["border"])

            # Value labels on bars
            for bar, pct in zip(bars, stock_pcts):
                self.ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                             f'{pct:.0f}%', ha='center', va='bottom',
                             color=COLORS["text_primary"], fontsize=9, fontweight='bold')

        self.fig.tight_layout()
        self.canvas.draw()


# ── Toast Notifications ────────────────────────────────────

class ToastNotification(QWidget):
    """Non-blocking toast message — slides in bottom-right, fades out."""

    def __init__(self, message, icon="✅", duration=3500, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFixedSize(340, 56)
        self._message = message
        self._icon = icon

        # Fade in
        self.setWindowOpacity(0.0)
        self._fade_in_anim = QPropertyAnimation(self, b"windowOpacity")
        self._fade_in_anim.setDuration(250)
        self._fade_in_anim.setStartValue(0.0)
        self._fade_in_anim.setEndValue(1.0)
        self._fade_in_anim.start()

        QTimer.singleShot(duration, self._fade_out)

    def _fade_out(self):
        anim = QPropertyAnimation(self, b"windowOpacity")
        anim.setDuration(400)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        anim.finished.connect(self.close)
        anim.finished.connect(self.deleteLater)
        anim.start()
        self._close_anim = anim

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()).adjusted(2, 2, -2, -2), 10, 10)
        painter.setPen(QPen(QColor(COLORS["border_accent"]), 1))
        painter.setBrush(QColor(COLORS["bg_card"]))
        painter.drawPath(path)

        # Accent bar
        bar = QPainterPath()
        bar.addRoundedRect(QRectF(4, 6, 4, self.height() - 12), 2, 2)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(COLORS["accent_start"]))
        painter.drawPath(bar)

        # Icon
        painter.setFont(QFont("Helvetica Neue", 18))
        painter.setPen(QColor(COLORS["text_primary"]))
        painter.drawText(18, 37, self._icon)

        # Message
        painter.setFont(QFont("Helvetica Neue", 12))
        painter.setPen(QColor(COLORS["text_primary"]))
        painter.drawText(48, 34, self._message)
        painter.end()


class ToastManager:
    """Manages toast positioning — stacks from bottom-right."""

    def __init__(self, parent_window):
        self._parent = parent_window
        self._active = []

    def show_toast(self, message, icon="✅", duration=3500):
        toast = ToastNotification(message, icon, duration)
        toast.destroyed.connect(lambda: self._remove(toast))
        self._active.append(toast)
        self._position_toasts()
        toast.show()

    def _remove(self, toast):
        if toast in self._active:
            self._active.remove(toast)
        self._position_toasts()

    def _position_toasts(self):
        from PyQt6.QtWidgets import QApplication
        screen = QApplication.primaryScreen()
        if not screen:
            return
        geo = screen.availableGeometry()
        bottom_y = geo.bottom() - 20
        for t in reversed(self._active):
            x = geo.right() - t.width() - 24
            y = bottom_y - t.height()
            t.move(x, y)
            bottom_y = y - 8


# ── Collapsible Log Console ────────────────────────────────

class LogConsole(QWidget):
    """Retractable developer console for streaming pipeline logs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        from PyQt6.QtWidgets import QTextEdit
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toggle header
        self.header_btn = QPushButton("  ▶  Developer Console")
        self.header_btn.setFixedHeight(32)
        self.header_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS['bg_secondary']};
                color: {COLORS['text_muted']};
                border: 1px solid {COLORS['border']};
                border-radius: 0;
                text-align: left;
                padding-left: 12px;
                font-family: 'Menlo', 'Consolas', monospace;
                font-size: 11px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                color: {COLORS['text_primary']};
                background: {COLORS['bg_hover']};
            }}
        """)
        self.header_btn.clicked.connect(self.toggle)
        layout.addWidget(self.header_btn)

        # Log text area
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setVisible(False)
        self.log_area.setFixedHeight(180)
        self.log_area.setStyleSheet(f"""
            QTextEdit {{
                background: #0d1117;
                color: {COLORS['success']};
                font-family: 'Menlo', 'Consolas', monospace;
                font-size: 11px;
                border: 1px solid {COLORS['border']};
                border-top: none;
                padding: 8px;
            }}
        """)
        layout.addWidget(self.log_area)
        self._expanded = False

    def toggle(self):
        self._expanded = not self._expanded
        self.log_area.setVisible(self._expanded)
        arrow = "▼" if self._expanded else "▶"
        self.header_btn.setText(f"  {arrow}  Developer Console")

    def log(self, message, level="info"):
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        color_map = {
            "info": COLORS["text_secondary"],
            "success": COLORS["success"],
            "warning": COLORS["warning"],
            "error": COLORS["danger"],
            "debug": COLORS["text_muted"],
        }
        c = color_map.get(level, COLORS["text_secondary"])
        level_tag = level.upper().ljust(7)
        self.log_area.append(
            f'<span style="color:{COLORS["text_muted"]}">[{ts}]</span> '
            f'<span style="color:{c}">{level_tag}</span> '
            f'<span style="color:{COLORS["text_primary"]}">{message}</span>'
        )
        sb = self.log_area.verticalScrollBar()
        sb.setValue(sb.maximum())

    def clear(self):
        self.log_area.clear()

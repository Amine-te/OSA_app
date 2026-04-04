# ─────────────────────────────────────────────────────────────
# viewer.py — Interactive Split-View Image Viewer with HUD
# Zoom/Pan, Clickable BBoxes, ROI Drawing, Heatmap Overlay
# ─────────────────────────────────────────────────────────────

import math
import numpy as np
import cv2

from PyQt6.QtWidgets import QWidget, QSizePolicy
from PyQt6.QtCore import Qt, QRectF, QPointF, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QPainter, QPen, QColor, QFont, QBrush, QImage, QPixmap, QPainterPath
)

from ui.styles import COLORS


class SplitCompareViewer(QWidget):
    """
    Interactive image viewer with:
    - Slide-to-Compare split view (raw vs annotated)
    - Mouse-wheel zoom + right-click pan
    - Clickable bounding boxes (emits detection_clicked)
    - ROI drawing mode
    - Heatmap overlay toggle
    - Detection highlighting & flashing
    """

    detection_clicked = pyqtSignal(int)   # index of clicked bbox
    roi_drawn = pyqtSignal(list)          # [x1, y1, x2, y2] image coords

    # Product color palette
    PALETTE = [
        "#667eea", "#764ba2", "#3fb950", "#d29922", "#f85149",
        "#58a6ff", "#d2a8ff", "#7ee787", "#ffa657", "#ff7b72",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Images
        self._raw_pixmap = None
        self._ann_pixmap = None

        # Split
        self._split_enabled = True
        self._split_pos = 0.5

        # Zoom / Pan
        self._zoom = 1.0
        self._pan = QPointF(0, 0)

        # Detections
        self._detections = []
        self._void_detections = []
        self._highlighted_idx = -1
        self._flash_idx = -1
        self._flash_visible = False

        # Flash timer
        self._flash_timer = QTimer(self)
        self._flash_timer.setInterval(150)
        self._flash_count = 0
        self._flash_timer.timeout.connect(self._flash_tick)

        # Heatmap
        self._heatmap_enabled = False
        self._heatmap_pixmap = None

        # ROI
        self._roi_mode = False
        self._roi_rect = None
        self._drawing_roi = False
        self._roi_start = QPointF()

        # Mouse state
        self._dragging_split = False
        self._panning = False
        self._pan_start = QPointF()

    # ── Public API ──────────────────────────────────────────

    def set_images(self, raw_cv, annotated_cv):
        """Set raw and annotated images (numpy BGR arrays)."""
        self._raw_pixmap = self._cv_to_pixmap(raw_cv) if raw_cv is not None else None
        self._ann_pixmap = self._cv_to_pixmap(annotated_cv) if annotated_cv is not None else None
        self._fit_image()
        self.update()

    def set_raw_image(self, raw_cv):
        self._raw_pixmap = self._cv_to_pixmap(raw_cv) if raw_cv is not None else None
        self._fit_image()
        self.update()

    def set_annotated_image(self, ann_cv):
        self._ann_pixmap = self._cv_to_pixmap(ann_cv) if ann_cv is not None else None
        self.update()

    def set_detections(self, product_dets, void_dets=None):
        self._detections = product_dets or []
        self._void_detections = void_dets or []
        self._generate_heatmap()
        self.update()

    def highlight_detection(self, idx):
        self._highlighted_idx = idx
        self.update()

    def flash_detection(self, idx):
        self._flash_idx = idx
        self._flash_count = 0
        self._flash_visible = True
        self._flash_timer.start()

    def enable_split(self, enabled):
        self._split_enabled = enabled
        self.update()

    def enable_roi_mode(self, enabled):
        self._roi_mode = enabled
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

    def clear_roi(self):
        self._roi_rect = None
        self._roi_mode = False
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def toggle_heatmap(self, enabled):
        self._heatmap_enabled = enabled
        self.update()

    def reset_view(self):
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self._fit_image()
        self.update()

    # ── Transform helpers ───────────────────────────────────

    def _fit_image(self):
        self._zoom = 1.0
        self._pan = QPointF(0, 0)

    def _get_base_scale_offset(self):
        pm = self._raw_pixmap or self._ann_pixmap
        if not pm:
            return 1.0, QPointF(0, 0)
        iw, ih = pm.width(), pm.height()
        if iw == 0 or ih == 0:
            return 1.0, QPointF(0, 0)
        s = min(self.width() / iw, self.height() / ih)
        ox = (self.width() - iw * s) / 2
        oy = (self.height() - ih * s) / 2
        return s, QPointF(ox, oy)

    def _total_scale_offset(self):
        bs, bo = self._get_base_scale_offset()
        ts = bs * self._zoom
        to = QPointF(bo.x() * self._zoom + self._pan.x(),
                     bo.y() * self._zoom + self._pan.y())
        return ts, to

    def _to_image(self, widget_pt):
        s, o = self._total_scale_offset()
        if s < 1e-6:
            return QPointF(0, 0)
        return QPointF((widget_pt.x() - o.x()) / s, (widget_pt.y() - o.y()) / s)

    def _to_widget(self, img_pt):
        s, o = self._total_scale_offset()
        return QPointF(img_pt.x() * s + o.x(), img_pt.y() * s + o.y())

    # ── Paint ───────────────────────────────────────────────

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(COLORS["bg_card"]))

        s, o = self._total_scale_offset()
        divider_x = int(self.width() * self._split_pos)
        has_both = self._raw_pixmap and self._ann_pixmap and self._split_enabled

        # ── Draw images ──
        if has_both:
            for side in ("left", "right"):
                pm = self._raw_pixmap if side == "left" else self._ann_pixmap
                clip_x = 0 if side == "left" else divider_x
                clip_w = divider_x if side == "left" else self.width() - divider_x
                painter.save()
                painter.setClipRect(clip_x, 0, clip_w, self.height())
                painter.translate(o)
                painter.scale(s, s)
                painter.drawPixmap(0, 0, pm)
                painter.restore()
        else:
            pm = self._ann_pixmap or self._raw_pixmap
            if pm:
                painter.save()
                painter.translate(o)
                painter.scale(s, s)
                painter.drawPixmap(0, 0, pm)
                painter.restore()

        # ── Heatmap overlay ──
        if self._heatmap_enabled and self._heatmap_pixmap:
            painter.save()
            painter.translate(o)
            painter.scale(s, s)
            painter.setOpacity(0.55)
            painter.drawPixmap(0, 0, self._heatmap_pixmap)
            painter.setOpacity(1.0)
            painter.restore()

        # ── Bounding boxes (widget coords for crisp lines) ──
        if self._detections and not self._heatmap_enabled:
            for i, det in enumerate(self._detections):
                self._draw_detection_widget(painter, i, det, s)

        # ── ROI ──
        if self._roi_rect:
            tl = self._to_widget(QPointF(self._roi_rect.x(), self._roi_rect.y()))
            br = self._to_widget(QPointF(self._roi_rect.right(), self._roi_rect.bottom()))
            pen = QPen(QColor(COLORS["info"]), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(QColor(88, 166, 255, 30))
            painter.drawRect(QRectF(tl, br))

        # ── Divider ──
        if has_both:
            self._draw_divider(painter, divider_x)

        painter.end()

    def _draw_detection_widget(self, painter, idx, det, scale):
        bbox = det['bbox']
        tl = self._to_widget(QPointF(bbox[0], bbox[1]))
        br = self._to_widget(QPointF(bbox[2], bbox[3]))
        rect = QRectF(tl, br)

        subclass = det.get('subclass', 'unknown')
        conf = det.get('combined_confidence', 0)
        cidx = hash(subclass) % len(self.PALETTE)
        color = QColor(self.PALETTE[cidx])

        is_hl = (idx == self._highlighted_idx)
        is_fl = (idx == self._flash_idx and self._flash_visible)

        if is_hl or is_fl:
            color = QColor(COLORS["info"])
            pw = 3
            fill = QColor(88, 166, 255, 50)
        else:
            pw = 2
            fill = QColor(color.red(), color.green(), color.blue(), 20)

        painter.setPen(QPen(color, pw))
        painter.setBrush(QBrush(fill))
        painter.drawRect(rect)

        # Label
        label = f"{subclass} {conf:.2f}"
        font = QFont("Helvetica Neue", 10, QFont.Weight.DemiBold)
        painter.setFont(font)
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(label) + 8
        th = fm.height() + 4
        lx, ly = tl.x(), tl.y() - th - 2
        if ly < 0:
            ly = br.y() + 2

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawRoundedRect(QRectF(lx, ly, tw, th), 4, 4)
        painter.setPen(QColor("white"))
        painter.drawText(QPointF(lx + 4, ly + fm.ascent() + 2), label)

    def _draw_divider(self, painter, dx):
        painter.setPen(QPen(QColor(255, 255, 255, 220), 2))
        painter.drawLine(dx, 0, dx, self.height())

        # Handle
        hy = self.height() // 2
        handle = QRectF(dx - 16, hy - 22, 32, 44)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 230))
        painter.drawRoundedRect(handle, 8, 8)
        painter.setPen(QPen(QColor(COLORS["bg_primary"]), 2))
        painter.setFont(QFont("Helvetica Neue", 14, QFont.Weight.Bold))
        painter.drawText(handle, Qt.AlignmentFlag.AlignCenter, "⟨⟩")

        # Labels
        painter.setPen(QColor(255, 255, 255, 190))
        painter.setFont(QFont("Helvetica Neue", 11, QFont.Weight.DemiBold))
        painter.drawText(12, 24, "Original")
        painter.drawText(self.width() - 90, 24, "Detection")

    # ── Heatmap generation ──────────────────────────────────

    def _generate_heatmap(self):
        pm = self._raw_pixmap or self._ann_pixmap
        if not pm or not self._void_detections:
            self._heatmap_pixmap = None
            return
        w, h = pm.width(), pm.height()
        heatmap = np.zeros((h, w), dtype=np.float32)
        for void in self._void_detections:
            x1, y1, x2, y2 = [max(0, int(v)) for v in void['bbox']]
            x2, y2 = min(x2, w), min(y2, h)
            heatmap[y1:y2, x1:x2] += 1.0
        if heatmap.max() > 0:
            ks = max(31, int(min(w, h) * 0.08)) | 1
            heatmap = cv2.GaussianBlur(heatmap, (ks, ks), 0)
            heatmap /= heatmap.max()
        colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGBA)
        colored[:, :, 3] = (heatmap * 180).astype(np.uint8)
        qimg = QImage(colored.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
        self._heatmap_pixmap = QPixmap.fromImage(qimg)

    # ── Mouse events ────────────────────────────────────────

    def wheelEvent(self, event):
        factor = 1.12 if event.angleDelta().y() > 0 else 1 / 1.12
        old_img = self._to_image(event.position())
        self._zoom = max(0.1, min(15.0, self._zoom * factor))
        new_img = self._to_image(event.position())
        s, _ = self._total_scale_offset()
        self._pan += QPointF((new_img.x() - old_img.x()) * s,
                             (new_img.y() - old_img.y()) * s)
        self.update()

    def mousePressEvent(self, event):
        pos = event.position()
        dx = self.width() * self._split_pos

        if self._split_enabled and self._raw_pixmap and self._ann_pixmap and abs(pos.x() - dx) < 16:
            self._dragging_split = True
            return

        if self._roi_mode and event.button() == Qt.MouseButton.LeftButton:
            self._drawing_roi = True
            self._roi_start = self._to_image(pos)
            self._roi_rect = QRectF(self._roi_start, self._roi_start)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            img_pt = self._to_image(pos)
            for i, det in enumerate(self._detections):
                b = det['bbox']
                if b[0] <= img_pt.x() <= b[2] and b[1] <= img_pt.y() <= b[3]:
                    self._highlighted_idx = i
                    self.detection_clicked.emit(i)
                    self.update()
                    return
            self._highlighted_idx = -1
            self.update()

        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._panning = True
            self._pan_start = pos

    def mouseMoveEvent(self, event):
        pos = event.position()
        if self._dragging_split:
            self._split_pos = max(0.05, min(0.95, pos.x() / self.width()))
            self.update()
            return
        if self._drawing_roi:
            self._roi_rect = QRectF(self._roi_start, self._to_image(pos)).normalized()
            self.update()
            return
        if self._panning:
            delta = pos - self._pan_start
            self._pan += QPointF(delta.x(), delta.y())
            self._pan_start = pos
            self.update()
            return
        # Cursor
        dx = self.width() * self._split_pos
        if self._split_enabled and self._raw_pixmap and self._ann_pixmap and abs(pos.x() - dx) < 16:
            self.setCursor(Qt.CursorShape.SplitHCursor)
        elif self._roi_mode:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if self._dragging_split:
            self._dragging_split = False
        if self._drawing_roi:
            self._drawing_roi = False
            self._roi_mode = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            if self._roi_rect and self._roi_rect.width() > 5 and self._roi_rect.height() > 5:
                self.roi_drawn.emit([int(self._roi_rect.x()), int(self._roi_rect.y()),
                                     int(self._roi_rect.right()), int(self._roi_rect.bottom())])
        if self._panning:
            self._panning = False

    def mouseDoubleClickEvent(self, event):
        self.reset_view()

    # ── Flash animation ─────────────────────────────────────

    def _flash_tick(self):
        self._flash_visible = not self._flash_visible
        self._flash_count += 1
        if self._flash_count >= 8:
            self._flash_timer.stop()
            self._flash_idx = -1
            self._flash_visible = False
        self.update()

    # ── Utility ─────────────────────────────────────────────

    @staticmethod
    def _cv_to_pixmap(cv_img):
        if cv_img is None:
            return None
        if len(cv_img.shape) == 3 and cv_img.shape[2] == 3:
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv_img
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)


# ── HUD Overlay ────────────────────────────────────────────

class HUDOverlay(QWidget):
    """Transparent heads-up display showing FPS, latency, device."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFixedSize(200, 82)
        self._fps = 0.0
        self._latency = 0.0
        self._device = "—"

    def update_stats(self, fps=None, latency_ms=None, device=None):
        if fps is not None:
            self._fps = fps
        if latency_ms is not None:
            self._latency = latency_ms
        if device is not None:
            self._device = device
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1), 8, 8)
        painter.setPen(QPen(QColor(255, 255, 255, 25), 1))
        painter.setBrush(QColor(0, 0, 0, 175))
        painter.drawPath(path)

        mono = QFont("Menlo", 11)
        mono.setStyleHint(QFont.StyleHint.Monospace)
        painter.setFont(mono)
        x, y = 14, 22

        # FPS
        painter.setPen(QColor(COLORS["success"]))
        painter.drawText(x, y, f"FPS      {self._fps:>6.1f}")
        y += 20

        # Latency
        c = COLORS["success"] if self._latency < 200 else COLORS["warning"] if self._latency < 800 else COLORS["danger"]
        painter.setPen(QColor(c))
        painter.drawText(x, y, f"Latency  {self._latency:>5.0f}ms")
        y += 20

        # Device
        painter.setPen(QColor(COLORS["info"]))
        painter.drawText(x, y, f"Device   {self._device:>6s}")
        painter.end()

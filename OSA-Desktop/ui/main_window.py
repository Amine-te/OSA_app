# ─────────────────────────────────────────────────────────────
# main_window.py — Industrial Control Center for OSA Desktop
# Split-view, HUD, ROI, Heatmap, Deep Linking, Toasts, Logging
# ─────────────────────────────────────────────────────────────

import os
import sys
import time
import traceback
from pathlib import Path

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTabWidget, QFileDialog,
    QStatusBar, QProgressBar, QScrollArea,
    QSpinBox, QComboBox, QSizePolicy, QFrame, QLineEdit, QApplication
)
from PyQt6.QtCore import Qt, pyqtSlot, QTimer
from PyQt6.QtGui import QFont, QPixmap

import numpy as np
import cv2

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from workers.pipeline_worker import PipelineWorker
from ui.styles import COLORS
from ui.sidebar import SidebarPanel
from ui.viewer import SplitCompareViewer, HUDOverlay
from ui.widgets import (
    GradientHeader, MetricCard, GaugeWidget,
    StockTable, ExportBar, RealTimePlotWidget,
    StaticReportCanvas, ToastManager, LogConsole,
)


class MainWindow(QMainWindow):
    """Industrial Control Center — OSA Desktop."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.worker = None
        self.pipeline_ready = False
        self.current_results = None
        self.video_frames = []
        self.current_frame_idx = 0

        # FPS tracking
        self._last_frame_time = 0
        self._fps = 0.0

        self._setup_window()
        self._build_ui()

        # Toast manager
        self.toasts = ToastManager(self)

        # Auto-initialize pipeline on startup
        QTimer.singleShot(500, lambda: self._on_initialize_pipeline(self.config))

    # ── Window ──────────────────────────────────────────────

    def _setup_window(self):
        self.setWindowTitle("OSA Desktop — Intelligent Retail Shelf Analysis")
        self.resize(1440, 900)
        self.setMinimumSize(1024, 700)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.status_bar.showMessage("⚙️  Configure and initialize pipeline to begin")

    # ── Build UI ────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Sidebar
        self.sidebar = SidebarPanel(self.config)
        self.sidebar.initialize_requested.connect(self._on_initialize_pipeline)

        separator = QFrame()
        separator.setFixedWidth(1)
        separator.setStyleSheet(f"background: {COLORS['border']};")

        # Main content
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(20, 16, 20, 0)
        content_layout.setSpacing(12)

        self.header = GradientHeader()
        content_layout.addWidget(self.header)

        # Pipeline banner
        self.pipeline_banner = QLabel("⚠️  Pipeline not initialized — configure in sidebar")
        self.pipeline_banner.setStyleSheet(f"""
            background: {COLORS['bg_card']}; color: {COLORS['warning']};
            padding: 12px 16px; border: 1px solid {COLORS['border']};
            border-left: 4px solid {COLORS['warning']}; border-radius: 8px;
            font-size: 13px; font-weight: 500;
        """)
        content_layout.addWidget(self.pipeline_banner)

        # Tabs
        self.tabs = QTabWidget()
        content_layout.addWidget(self.tabs, stretch=1)

        self._build_image_tab()
        self._build_video_tab()

        # Log console at bottom
        self.log_console = LogConsole()
        content_layout.addWidget(self.log_console)

        root.addWidget(self.sidebar)
        root.addWidget(separator)
        root.addWidget(content, stretch=1)

    # ── Image Tab ───────────────────────────────────────────

    def _build_image_tab(self):
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(tab)
        scroll.setStyleSheet(f"QScrollArea {{ border: none; background: {COLORS['bg_secondary']}; }}")

        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Upload area
        upload_area = QWidget()
        upload_area.setStyleSheet(f"""
            QWidget {{ background: {COLORS['bg_card']}; border: 2px dashed {COLORS['border']};
            border-radius: 12px; }}
        """)
        ul = QVBoxLayout(upload_area)
        ul.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ul.setSpacing(10)
        ul.setContentsMargins(40, 24, 40, 24)
        icon = QLabel("📷")
        icon.setStyleSheet("font-size: 36px; border:none; background:transparent;")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.btn_load_image = QPushButton("📂  Choose Image")
        self.btn_load_image.setFixedHeight(42)
        self.btn_load_image.setFixedWidth(200)
        self.btn_load_image.clicked.connect(self._on_load_image)
        ul.addWidget(icon)
        ul.addWidget(self.btn_load_image, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(upload_area)

        # ── Toolbar: Analyze / ROI / Heatmap / Reset ──
        self.img_toolbar = QWidget()
        self.img_toolbar.setVisible(False)
        tb = QHBoxLayout(self.img_toolbar)
        tb.setContentsMargins(0, 0, 0, 0)
        tb.setSpacing(8)

        self.btn_analyze = QPushButton("🔍  Analyze Image")
        self.btn_analyze.setFixedHeight(40)
        self.btn_analyze.clicked.connect(self._on_analyze_image)

        self.btn_roi = QPushButton("✂️  Draw ROI")
        self.btn_roi.setFixedHeight(40)
        self.btn_roi.setProperty("class", "secondary")
        self.btn_roi.clicked.connect(self._on_roi_mode)

        self.btn_heatmap = QPushButton("🌡️  Heatmap")
        self.btn_heatmap.setFixedHeight(40)
        self.btn_heatmap.setProperty("class", "secondary")
        self.btn_heatmap.setCheckable(True)
        self.btn_heatmap.toggled.connect(self._on_heatmap_toggle)

        self.btn_reset_view = QPushButton("🔄  Reset View")
        self.btn_reset_view.setFixedHeight(40)
        self.btn_reset_view.setProperty("class", "secondary")
        self.btn_reset_view.clicked.connect(lambda: self.img_viewer.reset_view())

        tb.addWidget(self.btn_analyze)
        tb.addWidget(self.btn_roi)
        tb.addWidget(self.btn_heatmap)
        tb.addWidget(self.btn_reset_view)
        tb.addStretch()
        layout.addWidget(self.img_toolbar)

        # ── Interactive Viewer (Split-View + Zoom) ──
        viewer_container = QWidget()
        viewer_container.setMinimumHeight(420)
        vc_layout = QVBoxLayout(viewer_container)
        vc_layout.setContentsMargins(0, 0, 0, 0)

        self.img_viewer = SplitCompareViewer()
        self.img_viewer.setMinimumHeight(400)
        self.img_viewer.detection_clicked.connect(self._on_image_detection_clicked)
        self.img_viewer.roi_drawn.connect(self._on_roi_drawn)
        vc_layout.addWidget(self.img_viewer)

        # HUD overlay — positioned top-right of viewer
        self.img_hud = HUDOverlay(self.img_viewer)
        self.img_hud.move(self.img_viewer.width() - 210, 8)
        self.img_hud.setVisible(False)

        self.img_viewer_container = viewer_container
        self.img_viewer_container.setVisible(False)
        layout.addWidget(self.img_viewer_container)

        # ── Metrics Row ──
        self.img_metrics_area = QWidget()
        self.img_metrics_area.setVisible(False)
        ml = QHBoxLayout(self.img_metrics_area)
        ml.setSpacing(12)
        ml.setContentsMargins(0, 0, 0, 0)
        self.card_total = MetricCard("📦", "Total Products", "—", COLORS["accent_start"])
        self.card_missing = MetricCard("⚠️", "Missing Products", "—", COLORS["danger"])
        self.card_stock = MetricCard("📈", "Overall Stock", "—", COLORS["success"])
        ml.addWidget(self.card_total)
        ml.addWidget(self.card_missing)
        ml.addWidget(self.card_stock)
        layout.addWidget(self.img_metrics_area)

        # ── Gauges ──
        self.img_gauges_area = QWidget()
        self.img_gauges_area.setVisible(False)
        self.img_gauges_layout = QHBoxLayout(self.img_gauges_area)
        self.img_gauges_layout.setSpacing(8)
        self.img_gauges_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.img_gauges_area)

        # ── Bar Chart ──
        self.img_chart = StaticReportCanvas(width=8, height=3, dpi=100)
        self.img_chart.setVisible(False)
        layout.addWidget(self.img_chart)

        # ── Stock Table (with click → image highlighting) ──
        self.img_table_header = QLabel("📋  Product Summary")
        self.img_table_header.setStyleSheet(f"font-size: 15px; font-weight: 600; color: {COLORS['text_primary']};")
        self.img_table_header.setVisible(False)
        layout.addWidget(self.img_table_header)

        self.img_table = StockTable()
        self.img_table.setVisible(False)
        self.img_table.cellClicked.connect(self._on_table_row_clicked)
        layout.addWidget(self.img_table)

        # ── Export ──
        self.img_export = ExportBar()
        self.img_export.setVisible(False)
        layout.addWidget(self.img_export)

        layout.addStretch()
        self.tabs.addTab(scroll, "📷  Image Analysis")
        self._loaded_image_path = None
        self._current_detections = []

    # ── Video Tab ───────────────────────────────────────────

    def _build_video_tab(self):
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(tab)
        scroll.setStyleSheet(f"QScrollArea {{ border: none; background: {COLORS['bg_secondary']}; }}")

        layout = QVBoxLayout(tab)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Upload
        upload_area = QWidget()
        upload_area.setStyleSheet(f"""
            QWidget {{ background: {COLORS['bg_card']}; border: 2px dashed {COLORS['border']};
            border-radius: 12px; }}
        """)
        ul = QVBoxLayout(upload_area)
        ul.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ul.setSpacing(10)
        ul.setContentsMargins(40, 24, 40, 24)
        vi = QLabel("🎥")
        vi.setStyleSheet("font-size: 36px; border:none; background:transparent;")
        vi.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ul.addWidget(vi)

        btn_row = QHBoxLayout()
        self.btn_load_video = QPushButton("📂  Choose Video")
        self.btn_load_video.setFixedHeight(42)
        self.btn_load_video.clicked.connect(self._on_load_video)
        self.input_rtsp = QLineEdit()
        self.input_rtsp.setPlaceholderText("rtsp://user:pass@ip/stream")
        self.input_rtsp.setFixedHeight(42)
        self.input_rtsp.setMinimumWidth(260)
        self.btn_rtsp = QPushButton("📡  Connect")
        self.btn_rtsp.setFixedHeight(42)
        self.btn_rtsp.clicked.connect(self._on_connect_rtsp)
        btn_row.addWidget(self.btn_load_video)
        btn_row.addWidget(self.input_rtsp)
        btn_row.addWidget(self.btn_rtsp)
        ul.addLayout(btn_row)
        layout.addWidget(upload_area)

        # Controls
        self.video_controls = QWidget()
        self.video_controls.setVisible(False)
        cl = QHBoxLayout(self.video_controls)
        cl.setSpacing(12)
        cl.setContentsMargins(0, 0, 0, 0)

        for lbl_text, attr, lo, hi, default in [
            ("Interval (s):", "spin_interval", 1, 60, 5),
            ("Max Frames:", "spin_max", 1, 100, 10),
        ]:
            l = QLabel(lbl_text)
            l.setStyleSheet(f"color:{COLORS['text_secondary']}; font-size:12px;")
            s = QSpinBox()
            s.setRange(lo, hi)
            s.setValue(default)
            s.setFixedWidth(80)
            setattr(self, attr, s)
            cl.addWidget(l)
            cl.addWidget(s)

        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Frame-by-Frame", "Trend Analysis"])
        self.combo_mode.setFixedWidth(160)
        cl.addWidget(QLabel("Mode:"))
        cl.addWidget(self.combo_mode)
        cl.addStretch()

        self.btn_analyze_video = QPushButton("🎬  Analyze")
        self.btn_analyze_video.setFixedHeight(38)
        self.btn_analyze_video.clicked.connect(self._on_analyze_video)
        self.btn_clear_video = QPushButton("🧹  Clear")
        self.btn_clear_video.setFixedHeight(38)
        self.btn_clear_video.setProperty("class", "secondary")
        self.btn_clear_video.clicked.connect(self._on_clear_video)
        cl.addWidget(self.btn_analyze_video)
        cl.addWidget(self.btn_clear_video)
        layout.addWidget(self.video_controls)

        # Video viewer + metrics
        self.vid_display = QWidget()
        self.vid_display.setVisible(False)
        vd = QHBoxLayout(self.vid_display)
        vd.setSpacing(12)

        self.vid_viewer = SplitCompareViewer()
        self.vid_viewer.enable_split(False)
        self.vid_viewer.setMinimumHeight(360)

        self.vid_hud = HUDOverlay(self.vid_viewer)
        self.vid_hud.move(10, 8)
        self.vid_hud.setVisible(False)

        vcol = QVBoxLayout()
        vcol.setSpacing(8)
        self.vid_card_total = MetricCard("📦", "Products", "—", COLORS["accent_start"])
        self.vid_card_missing = MetricCard("⚠️", "Missing", "—", COLORS["danger"])
        self.vid_card_stock = MetricCard("📈", "Stock %", "—", COLORS["success"])
        vcol.addWidget(self.vid_card_total)
        vcol.addWidget(self.vid_card_missing)
        vcol.addWidget(self.vid_card_stock)
        vcol.addStretch()

        vd.addWidget(self.vid_viewer, stretch=2)
        vd.addLayout(vcol, stretch=0)
        layout.addWidget(self.vid_display)

        # Frame nav
        self.frame_nav = QWidget()
        self.frame_nav.setVisible(False)
        nl = QHBoxLayout(self.frame_nav)
        nl.setSpacing(8)
        nl.setContentsMargins(0, 0, 0, 0)
        for text, slot in [("⏮️", lambda: self._goto_frame(0)),
                           ("◀️", lambda: self._goto_frame(self.current_frame_idx - 1))]:
            b = QPushButton(text)
            b.setProperty("class", "secondary")
            b.setFixedHeight(34)
            b.clicked.connect(slot)
            nl.addWidget(b)
        self.combo_frame = QComboBox()
        self.combo_frame.setMinimumWidth(200)
        self.combo_frame.currentIndexChanged.connect(self._on_frame_selected)
        nl.addWidget(self.combo_frame, stretch=1)
        for text, slot in [("▶️", lambda: self._goto_frame(self.current_frame_idx + 1)),
                           ("⏭️", lambda: self._goto_frame(len(self.video_frames) - 1))]:
            b = QPushButton(text)
            b.setProperty("class", "secondary")
            b.setFixedHeight(34)
            b.clicked.connect(slot)
            nl.addWidget(b)
        self.lbl_frame_info = QLabel("")
        self.lbl_frame_info.setStyleSheet(f"color:{COLORS['text_muted']}; font-size:12px;")
        nl.addWidget(self.lbl_frame_info)
        layout.addWidget(self.frame_nav)

        # Trend chart
        self.vid_trend = RealTimePlotWidget()
        self.vid_trend.setVisible(False)
        self.vid_trend.setMinimumHeight(220)
        layout.addWidget(self.vid_trend)

        # Table
        self.vid_table_header = QLabel("📋  Frame Comparison")
        self.vid_table_header.setStyleSheet(f"font-size:15px; font-weight:600; color:{COLORS['text_primary']};")
        self.vid_table_header.setVisible(False)
        layout.addWidget(self.vid_table_header)
        self.vid_table = StockTable()
        self.vid_table.setVisible(False)
        layout.addWidget(self.vid_table)

        layout.addStretch()
        self.tabs.addTab(scroll, "🎥  Video Analysis")
        self._loaded_video_path = None

    # ════════════════════════════════════════════════════════
    # PIPELINE
    # ════════════════════════════════════════════════════════

    def _on_initialize_pipeline(self, config):
        self.config = config
        self.sidebar.btn_init.setEnabled(False)
        self.sidebar.set_status("⏳ Initializing...")
        self.status_bar.showMessage("⏳ Initializing pipeline...")
        self.log_console.log("Pipeline initialization started", "info")
        QApplication.processEvents()

        try:
            from utils.path_utils import resolve_path
            paths = {
                "YOLO": resolve_path(config["models"]["yolo_product"]),
                "CNN": resolve_path(config["models"]["cnn_class"]),
                "Void": resolve_path(config["models"]["yolo_void"]),
            }
            missing = [f"{k}: {v}" for k, v in paths.items() if not v.exists()]
            if missing:
                msg = f"❌ Missing: {', '.join(missing)}"
                self.sidebar.set_status(msg, is_error=True)
                self.log_console.log(msg, "error")
                self.sidebar.btn_init.setEnabled(True)
                return

            self.pipeline_ready = True
            self.pipeline_banner.setText("✅  Pipeline initialized — ready to analyze")
            self.pipeline_banner.setStyleSheet(f"""
                background: {COLORS['bg_card']}; color: {COLORS['success']};
                padding: 12px 16px; border: 1px solid {COLORS['border']};
                border-left: 4px solid {COLORS['success']}; border-radius: 8px;
                font-size: 13px; font-weight: 500;
            """)
            self.sidebar.set_status("✅ Pipeline initialized!")
            self.status_bar.showMessage("✅ Pipeline ready")
            self.log_console.log("Pipeline initialized successfully", "success")
            for k, v in paths.items():
                self.log_console.log(f"  {k} model: {v.name}", "debug")
            self.toasts.show_toast("Pipeline initialized successfully", "✅")

        except Exception as e:
            self.sidebar.set_status(f"❌ {e}", is_error=True)
            self.log_console.log(f"Pipeline error: {e}", "error")
            traceback.print_exc()
        finally:
            self.sidebar.btn_init.setEnabled(True)

    # ── Image Actions ───────────────────────────────────────

    def _on_load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self._loaded_image_path = path
            raw = cv2.imread(path)
            self.img_viewer.set_images(raw, None)
            self.img_viewer.enable_split(False)
            self.img_viewer_container.setVisible(True)
            self.img_toolbar.setVisible(True)
            self.status_bar.showMessage(f"📷 Loaded: {os.path.basename(path)}")
            self.log_console.log(f"Image loaded: {os.path.basename(path)}", "info")

    def _on_analyze_image(self):
        if not self._loaded_image_path or not self.pipeline_ready:
            self.status_bar.showMessage("⚠️ Load an image and initialize pipeline first")
            return
        self._start_processing(self._loaded_image_path, is_rtsp=False)

    def _on_roi_mode(self):
        self.img_viewer.enable_roi_mode(True)
        self.status_bar.showMessage("✂️ Draw a rectangle on the image to define ROI")
        self.log_console.log("ROI drawing mode enabled", "info")

    def _on_roi_drawn(self, coords):
        self.log_console.log(f"ROI defined: {coords}", "success")
        self.toasts.show_toast(f"ROI set: [{coords[0]},{coords[1]}]-[{coords[2]},{coords[3]}]", "✂️")

    def _on_heatmap_toggle(self, checked):
        self.img_viewer.toggle_heatmap(checked)
        self.log_console.log(f"Heatmap {'enabled' if checked else 'disabled'}", "info")

    def _on_image_detection_clicked(self, idx):
        """Image bbox clicked → highlight table row."""
        if idx < self.img_table.rowCount():
            self.img_table.selectRow(idx)
            self.img_table.scrollToItem(self.img_table.item(idx, 0))
            self.log_console.log(f"Detection #{idx} selected from image", "debug")

    def _on_table_row_clicked(self, row, col):
        """Table row clicked → flash bbox in image."""
        self.img_viewer.flash_detection(row)
        self.log_console.log(f"Flashing detection #{row} from table click", "debug")

    # ── Video Actions ───────────────────────────────────────

    def _on_load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video (*.mp4 *.avi *.mov *.mkv)")
        if path:
            self._loaded_video_path = path
            self.video_controls.setVisible(True)
            self.status_bar.showMessage(f"🎥 Loaded: {os.path.basename(path)}")
            self.log_console.log(f"Video loaded: {os.path.basename(path)}", "info")

    def _on_connect_rtsp(self):
        url = self.input_rtsp.text().strip()
        if url:
            self._loaded_video_path = url
            self.video_controls.setVisible(True)
            self.log_console.log(f"RTSP URL set: {url}", "info")

    def _on_analyze_video(self):
        if not self._loaded_video_path or not self.pipeline_ready:
            return
        self.video_frames.clear()
        self.current_frame_idx = 0
        self.vid_trend.clear_plot()
        self.combo_frame.clear()
        is_rtsp = self._loaded_video_path.startswith("rtsp://")
        self._start_processing(self._loaded_video_path, is_rtsp=is_rtsp)

    def _on_clear_video(self):
        self.video_frames.clear()
        self.vid_trend.clear_plot()
        self.combo_frame.clear()
        self.vid_display.setVisible(False)
        self.frame_nav.setVisible(False)
        self.vid_trend.setVisible(False)
        self.vid_table.setVisible(False)
        self.vid_table_header.setVisible(False)

    # ── Processing ──────────────────────────────────────────

    def _start_processing(self, path, is_rtsp=False):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        self.worker = PipelineWorker(media_path=path, config=self.config, is_rtsp=is_rtsp)
        self.worker.started_processing.connect(self._on_started)
        self.worker.frame_processed.connect(self._on_frame_processed)
        self.worker.finished_processing.connect(self._on_finished)
        self.worker.error_occurred.connect(self._on_error)
        self._last_frame_time = time.time()
        self.worker.start()
        self.log_console.log(f"Processing started: {os.path.basename(path)}", "info")

    def _on_started(self):
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_bar.showMessage("⏳ Processing...")

    @pyqtSlot(dict)
    def _on_frame_processed(self, results):
        # FPS calculation
        now = time.time()
        dt = now - self._last_frame_time
        self._fps = 1.0 / dt if dt > 0 else 0
        self._last_frame_time = now

        self.current_results = results
        is_image = (self._loaded_image_path and not self.worker.is_rtsp and
                    any(self._loaded_image_path.lower().endswith(e) for e in ['.jpg', '.jpeg', '.png', '.bmp']))

        if is_image:
            self._display_image_results(results)
        else:
            self._display_video_frame(results)

    def _on_finished(self):
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("✅ Processing complete")
        self.log_console.log("Processing finished", "success")
        self.toasts.show_toast("Analysis complete!", "✅")
        if self.video_frames and self.combo_mode.currentText() == "Trend Analysis":
            self.vid_trend.setVisible(True)

    def _on_error(self, msg):
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage(f"❌ {msg}")
        self.log_console.log(f"Error: {msg}", "error")
        self.toasts.show_toast(f"Error: {msg[:50]}", "❌")

    # ── Display Image Results ───────────────────────────────

    def _display_image_results(self, results):
        raw = results.get('raw_image')
        ann = results.get('image')
        self.img_viewer.set_images(raw, ann)
        self.img_viewer.enable_split(True)
        self.img_viewer_container.setVisible(True)

        # Set detections for interactive click
        dets = results.get('product_detections', [])
        voids = results.get('void_detections', [])
        self._current_detections = dets
        self.img_viewer.set_detections(dets, voids)

        # HUD
        latency = results.get('inference_time_ms', 0)
        device = results.get('device', 'CPU')
        self.img_hud.update_stats(fps=self._fps, latency_ms=latency, device=device)
        self.img_hud.setVisible(True)
        self.img_hud.move(self.img_viewer.width() - 210, 8)

        # Metrics
        summary = results.get('summary', {})
        self.card_total.set_value(summary.get('total_products_detected', 0))
        self.card_missing.set_value(summary.get('estimated_missing_products', 0))
        pct = summary.get('overall_stock_percentage', 0)
        self.card_stock.set_value(f"{pct:.1f}%")
        self.img_metrics_area.setVisible(True)

        # Gauges
        self._clear_layout(self.img_gauges_layout)
        stock_levels = summary.get('stock_levels', {})
        if stock_levels:
            for product, data in stock_levels.items():
                g = GaugeWidget(product.title(), data.get('stock_percentage', 0))
                g.setFixedHeight(200)
                self.img_gauges_layout.addWidget(g)
            self.img_gauges_area.setVisible(True)

        self.img_chart.plot_summary(summary)
        self.img_chart.setVisible(True)

        if stock_levels:
            self.img_table.load_data(stock_levels)
            self.img_table.setVisible(True)
            self.img_table_header.setVisible(True)

        self.img_export.set_results(results)
        self.img_export.setVisible(True)

        self.log_console.log(
            f"Image analysis: {summary.get('total_products_detected', 0)} products, "
            f"{pct:.1f}% stock, {latency:.0f}ms latency", "success"
        )

    # ── Display Video Frame ─────────────────────────────────

    def _display_video_frame(self, results):
        idx = len(self.video_frames)
        results['frame_number'] = idx + 1
        self.video_frames.append(results)

        summary = results.get('summary', {})
        pct = summary.get('overall_stock_percentage', 0)
        self.combo_frame.addItem(f"Frame {idx + 1} — {pct:.1f}%")
        self.vid_trend.update_plot(idx + 1, pct)

        self._show_video_frame_at(idx)
        self.vid_display.setVisible(True)
        self.frame_nav.setVisible(True)
        self.vid_trend.setVisible(self.combo_mode.currentText() == "Trend Analysis")

    def _show_video_frame_at(self, idx):
        if idx < 0 or idx >= len(self.video_frames):
            return
        self.current_frame_idx = idx
        r = self.video_frames[idx]

        ann = r.get('image')
        if ann is not None:
            self.vid_viewer.set_images(None, ann)

        # HUD
        self.vid_hud.update_stats(fps=self._fps, latency_ms=r.get('inference_time_ms', 0),
                                   device=r.get('device', 'CPU'))
        self.vid_hud.setVisible(True)

        s = r.get('summary', {})
        self.vid_card_total.set_value(s.get('total_products_detected', 0))
        self.vid_card_missing.set_value(s.get('estimated_missing_products', 0))
        self.vid_card_stock.set_value(f"{s.get('overall_stock_percentage', 0):.1f}%")
        self.lbl_frame_info.setText(f"Frame {idx + 1} / {len(self.video_frames)}")

        sl = s.get('stock_levels', {})
        if sl:
            self.vid_table.load_data(sl)
            self.vid_table.setVisible(True)
            self.vid_table_header.setVisible(True)

        self.combo_frame.blockSignals(True)
        self.combo_frame.setCurrentIndex(idx)
        self.combo_frame.blockSignals(False)

    def _goto_frame(self, idx):
        self._show_video_frame_at(max(0, min(idx, len(self.video_frames) - 1)))

    def _on_frame_selected(self, idx):
        if idx >= 0:
            self._show_video_frame_at(idx)

    # ── Utilities ───────────────────────────────────────────

    def _clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Reposition HUD overlays
        if hasattr(self, 'img_hud'):
            self.img_hud.move(self.img_viewer.width() - 210, 8)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        super().closeEvent(event)

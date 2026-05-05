# ─────────────────────────────────────────────────────────────
# main_window.py — OSA Industrial Control Center (dock workspaces)
# ─────────────────────────────────────────────────────────────

import sys
import time
import traceback
from pathlib import Path

from datetime import datetime

from PyQt6.QtCore import QByteArray, Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QAction, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QDockWidget,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

# Ensure OSA-Desktop root is importable when this module is loaded in unusual contexts.
_app_root = Path(__file__).resolve().parent.parent
if str(_app_root) not in sys.path:
    sys.path.insert(0, str(_app_root))

from core.app_state import AppState, PipelineState, SourceType
from core.event_bus import EventBus
from core import session_manager
from core.history_store import HistoryStore

from workers.pipeline_worker import PipelineWorker
from ui.styles import COLORS, toggle_theme, current_theme
from ui.auxiliary_windows import AnalyticsWindow, ConfigWindow, InventoryReportWindow
from ui.viewer import SplitCompareViewer, HUDOverlay
from ui.widgets import ToastManager, LogConsole
from ui.error_banner import ErrorBanner


class MainWindow(QMainWindow):
    """Dock-based control center; workers emit through EventBus."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.app_state = AppState()
        self.bus = EventBus()
        self.worker = None
        self.pipeline_ready = False
        self.current_results = None
        self._last_frame_time = time.time()
        self._fps = 0.0
        self._base_dir = Path(__file__).resolve().parent.parent
        self._history = HistoryStore(session_manager.sessions_root(self._base_dir) / "history.db")
        self._analytics_frame_idx = 0

        self._setup_window()
        self._build_ui()
        self._wire_bus()
        self._register_shortcuts()

        self.toasts = ToastManager(self)

        QTimer.singleShot(400, self._auto_init_pipeline)
        QTimer.singleShot(600, self._try_restore_session)

    # ── Window chrome ───────────────────────────────────────

    def _setup_window(self):
        self.setWindowTitle("OSA")
        self.resize(1480, 920)
        self.setMinimumSize(1100, 720)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(220)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        self.setDockNestingEnabled(True)
        self.setCorner(Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.LeftDockWidgetArea)

        self._build_menubar()

    def _build_menubar(self):
        mb = self.menuBar()

        a_cfg = QAction("Configuration", self)
        a_cfg.setShortcut(QKeySequence("Ctrl+,"))
        a_cfg.triggered.connect(self._show_config_window)
        mb.addAction(a_cfg)

        a_an = QAction("Analytics", self)
        a_an.triggered.connect(self._show_analytics_window)
        mb.addAction(a_an)

        a_inv = QAction("Inventory / report", self)
        a_inv.triggered.connect(self._show_inventory_window)
        mb.addAction(a_inv)

        view = mb.addMenu("&View")
        a_log = QAction("Log console", self)
        a_log.setShortcut(QKeySequence("Ctrl+`"))
        a_log.triggered.connect(self._toggle_log_dock)
        view.addAction(a_log)
        a_feed = QAction("Live detection feed", self)
        a_feed.triggered.connect(self._toggle_void_feed_dock)
        view.addAction(a_feed)

        view.addSeparator()
        self.act_theme = QAction("🌙  Switch to Dark Theme", self)
        self.act_theme.setShortcut(QKeySequence("Ctrl+Shift+T"))
        self.act_theme.triggered.connect(self._on_toggle_theme)
        view.addAction(self.act_theme)

    def _setup_auxiliary_windows(self):
        self._config_win = ConfigWindow(self.config, self)
        self._config_win.sidebar.initialize_requested.connect(self._on_initialize_pipeline_config)
        self.sidebar = self._config_win.sidebar

        self._analytics_win = AnalyticsWindow(self)
        self.perf_panel = self._analytics_win.perf_panel

        self._inventory_win = InventoryReportWindow(self)
        self.img_table = self._inventory_win.img_table
        self.report_panel = self._inventory_win.report_panel

        # Bind analytics history store (video/live)
        self._analytics_win.bind_history(self._history, self.app_state.analytics_session_id)

    def _show_config_window(self):
        self._config_win.show()
        self._config_win.raise_()
        self._config_win.activateWindow()

    def _show_analytics_window(self):
        # Ensure current session is bound before showing.
        self._analytics_win.bind_history(self._history, self.app_state.analytics_session_id)
        self._analytics_win.show()
        self._analytics_win.raise_()
        self._analytics_win.activateWindow()

    def _show_inventory_window(self):
        self._inventory_win.show()
        self._inventory_win.raise_()
        self._inventory_win.activateWindow()

    def _snap_log_dock(self):
        if not self.dock_log.isVisible():
            return
        self.dock_log.setFloating(False)
        h = min(200, max(130, int(self.height() * 0.2)))
        self.resizeDocks([self.dock_log], [h], Qt.Orientation.Vertical)

    def _snap_void_feed_dock(self):
        if not self.dock_void_feed.isVisible():
            return
        self.dock_void_feed.setFloating(False)
        w = min(360, max(220, int(self.width() * 0.22)))
        self.resizeDocks([self.dock_void_feed], [w], Qt.Orientation.Horizontal)

    def _toggle_log_dock(self):
        show = not self.dock_log.isVisible()
        self.dock_log.setFloating(False)
        self.dock_log.setVisible(show)
        if show:
            self.dock_log.raise_()
            QTimer.singleShot(0, self._snap_log_dock)

    def _toggle_void_feed_dock(self):
        show = not self.dock_void_feed.isVisible()
        self.dock_void_feed.setFloating(False)
        self.dock_void_feed.setVisible(show)
        if show:
            self.dock_void_feed.raise_()
            QTimer.singleShot(0, self._snap_void_feed_dock)

    def _collect_auxiliary_ui_snapshot(self) -> dict:
        def pack(win: QMainWindow) -> dict:
            g = win.saveGeometry()
            return {
                "geometry": bytes(g.toBase64()).decode("ascii"),
                "visible": win.isVisible(),
            }

        return {
            "config": pack(self._config_win),
            "analytics": pack(self._analytics_win),
            "inventory": pack(self._inventory_win),
            "log_dock_visible": self.dock_log.isVisible(),
            "void_feed_visible": self.dock_void_feed.isVisible(),
        }

    def _restore_auxiliary_ui(self, payload: dict) -> None:
        if not payload:
            return
        mapping = (
            ("config", self._config_win),
            ("analytics", self._analytics_win),
            ("inventory", self._inventory_win),
        )
        for key, win in mapping:
            block = payload.get(key)
            if not isinstance(block, dict):
                continue
            geo = block.get("geometry") or ""
            if geo:
                win.restoreGeometry(QByteArray.fromBase64(geo.encode("ascii")))
            if bool(block.get("visible")):
                win.show()
        if "log_dock_visible" in payload:
            vis = bool(payload["log_dock_visible"])
            self.dock_log.setFloating(False)
            self.dock_log.setVisible(vis)
            if vis:
                QTimer.singleShot(0, self._snap_log_dock)
        if "void_feed_visible" in payload:
            vis = bool(payload["void_feed_visible"])
            self.dock_void_feed.setFloating(False)
            self.dock_void_feed.setVisible(vis)
            if vis:
                QTimer.singleShot(0, self._snap_void_feed_dock)

    def _build_ui(self):
        self._setup_auxiliary_windows()

        wrap = QWidget()
        root = QVBoxLayout(wrap)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)



        self.error_banner = ErrorBanner()
        self.error_banner.retry_clicked.connect(self._retry_last_action)
        self.error_banner.clear_clicked.connect(self._clear_error_state)
        root.addWidget(self.error_banner)

        self.setCentralWidget(wrap)


        self._build_workspaces()
        self._build_docks()

        inner = QVBoxLayout()
        inner.setContentsMargins(12, 4, 12, 10)
        inner.setSpacing(8)
        inner.addWidget(self.main_stack, stretch=1)
        wrap.layout().addLayout(inner)





    def _build_workspaces(self):
        self.main_stack = QStackedWidget()

        # Live (RTSP) — only workspace
        self.ws_live = QWidget()
        ll = QVBoxLayout(self.ws_live)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(10)

        live_ctrl = QHBoxLayout()
        live_ctrl.setSpacing(8)
        self.input_rtsp_live = QLineEdit()
        self.input_rtsp_live.setPlaceholderText("rtsp://…")
        self.btn_rtsp_connect = QPushButton("Connect")
        self.btn_rtsp_connect.setProperty("class", "secondary")
        self.btn_rtsp_connect.clicked.connect(self._on_connect_rtsp_live)
        self.btn_run_live = QPushButton("Run live analysis")
        self.btn_run_live.clicked.connect(self._on_analyze_live)
        self.btn_clear_live = QPushButton("Clear")
        self.btn_clear_live.setProperty("class", "secondary")
        self.btn_clear_live.clicked.connect(self._on_clear_live)
        live_ctrl.addWidget(QLabel("RTSP"))
        live_ctrl.addWidget(self.input_rtsp_live, stretch=1)
        live_ctrl.addWidget(self.btn_rtsp_connect)
        live_ctrl.addWidget(self.btn_run_live)
        live_ctrl.addWidget(self.btn_clear_live)
        live_ctrl.addStretch()
        ll.addLayout(live_ctrl)

        self.live_viewer = SplitCompareViewer()
        self.live_viewer.enable_split(False)
        self.live_viewer.detection_clicked.connect(self._on_image_detection_clicked)
        ll.addWidget(self.live_viewer, stretch=1)

        self.live_hud = HUDOverlay(self.live_viewer)
        self.live_hud.setParent(self.live_viewer)
        self.live_hud.move(8, 8)
        self.live_hud.setVisible(False)

        note = QLabel("Live camera analysis (RTSP). History is recorded in Analytics.")
        note.setStyleSheet(f"color:{COLORS['text_muted']}; font-size: 11px; padding:6px 0;")
        ll.addWidget(note)

        self.main_stack.addWidget(self.ws_live)

    def _compact_dock_title_bar(self, dock: QDockWidget, title: str) -> None:
        """Compact dock title: small label + close (less height than native title bar)."""
        dock.setWindowTitle(title)
        strip = QWidget()
        strip.setFixedHeight(20)
        strip.setStyleSheet(
            f"background:{COLORS['bg_card']}; border-bottom:1px solid {COLORS['border']};"
        )
        hl = QHBoxLayout(strip)
        hl.setContentsMargins(8, 0, 4, 0)
        hl.setSpacing(6)
        lbl = QLabel(title)
        lbl.setStyleSheet(
            f"color:{COLORS['text_primary']}; font-size:11px; font-weight:600; "
            f"background:transparent; border:none; padding:0;"
        )
        hl.addWidget(lbl, alignment=Qt.AlignmentFlag.AlignVCenter)
        hl.addStretch()
        btn = QPushButton("×")
        btn.setFixedSize(20, 16)
        btn.setFlat(True)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setToolTip("Hide panel")
        btn.setStyleSheet(
            f"color:{COLORS['text_muted']}; font-size:13px; font-weight:700; "
            f"border:none; padding:0; margin:0; background:transparent;"
        )
        btn.clicked.connect(dock.hide)
        hl.addWidget(btn, alignment=Qt.AlignmentFlag.AlignVCenter)
        dock.setTitleBarWidget(strip)

    def _build_docks(self):
        self.img_table.cellClicked.connect(self._on_table_row_clicked)
        self.img_table.itemEntered.connect(self._on_table_hover)

        _dock_flags = (
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )

        self.dock_log = QDockWidget(self)
        self.dock_log.setObjectName("dock_log")
        self.dock_log.setFeatures(_dock_flags)
        self.dock_log.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.log_console = LogConsole()
        self.dock_log.setWidget(self.log_console)
        self._compact_dock_title_bar(self.dock_log, "Log console")
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.dock_log)
        self.dock_log.setVisible(False)

        self.dock_void_feed = QDockWidget(self)
        self.dock_void_feed.setObjectName("dock_void_feed")
        self.dock_void_feed.setFeatures(_dock_flags)
        self.dock_void_feed.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        feed_host = QWidget()
        fl = QVBoxLayout(feed_host)
        fl.setContentsMargins(4, 4, 4, 4)
        self.void_feed_list = QListWidget()
        fl.addWidget(self.void_feed_list)
        self.dock_void_feed.setWidget(feed_host)
        self._compact_dock_title_bar(self.dock_void_feed, "Live detection feed")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock_void_feed)
        self.dock_void_feed.setVisible(False)

    def _wire_bus(self):
        self.bus.frame_updated.connect(self._on_bus_frame)
        self.bus.detections_updated.connect(self._on_bus_detections)
        self.bus.pipeline_status_changed.connect(self._on_bus_pipeline_state)
        self.bus.error_occurred.connect(self._on_bus_error)

    def _register_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+,"), self, activated=self._show_config_window)
        QShortcut(QKeySequence("Ctrl+E"), self, activated=self._export_quick)
        QShortcut(QKeySequence("Meta+E"), self, activated=self._export_quick)
        QShortcut(QKeySequence("Ctrl+`"), self, activated=self._toggle_log_dock)

    # ── Event bus handlers ────────────────────────────────────

    @pyqtSlot(object)
    def _on_bus_frame(self, payload):
        self.current_results = payload
        self.app_state.last_results = payload

    @pyqtSlot(list, list)
    def _on_bus_detections(self, dets, voids):
        self.app_state.detections = list(dets or [])
        self.app_state.void_detections = list(voids or [])
        t = datetime.now().strftime("%H:%M:%S")
        n_d = len(dets or [])
        n_v = len(voids or [])
        self.void_feed_list.insertItem(0, f"{t}  ·  products {n_d}  ·  voids {n_v}")
        while self.void_feed_list.count() > 80:
            self.void_feed_list.takeItem(self.void_feed_list.count() - 1)

    @pyqtSlot(object)
    def _on_bus_pipeline_state(self, state: PipelineState):
        self.app_state.set_pipeline(state)


    @pyqtSlot(str)
    def _on_bus_error(self, msg: str):
        self.error_banner.show_error(msg)

    # ── Pipeline state machine ──────────────────────────────

    def _set_pipeline_state(self, state: PipelineState, err: str = ""):
        self.app_state.set_pipeline(state, err)
        self.bus.pipeline_status_changed.emit(state)
        if state == PipelineState.ERROR and err:
            self.bus.error_occurred.emit(err)


    def _on_start_clicked(self):
        if not self.pipeline_ready:
            self._on_initialize_pipeline_config(self._gather_config_from_sidebar())
            return
        if self.app_state.pipeline_state == PipelineState.PAUSED:
            self._resume_processing()
            return
        self._on_analyze_live()

    def _on_pause_clicked(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker = None
        self._set_pipeline_state(PipelineState.PAUSED)

    def _on_stop_clicked(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker = None
        self.progress_bar.setVisible(False)
        self._set_pipeline_state(PipelineState.READY if self.pipeline_ready else PipelineState.IDLE)

    def _retry_last_action(self):
        self.error_banner.setVisible(False)
        self._on_start_clicked()

    def _clear_error_state(self):
        self.app_state.last_error = ""
        self._set_pipeline_state(PipelineState.READY if self.pipeline_ready else PipelineState.IDLE)

    def _gather_config_from_sidebar(self):
        class_text = self.sidebar.input_classes.text()
        class_names = [n.strip() for n in class_text.split(",") if n.strip()]
        return {
            "models": {
                "yolo_product": self.sidebar.input_yolo.text(),
                "cnn_class": self.sidebar.input_cnn.text(),
                "yolo_void": self.sidebar.input_void.text(),
            },
            "class_names": class_names,
            "thresholds": {
                "confidence": self.sidebar.slider_conf.value() / 100.0,
                "void_confidence": self.sidebar.slider_void.value() / 100.0,
            },
            "ui": self.config.get("ui", {}),
        }

    def _auto_init_pipeline(self):
        self._on_initialize_pipeline_config(self._gather_config_from_sidebar())

    def _on_initialize_pipeline_config(self, config):
        self.config = config
        self._set_pipeline_state(PipelineState.LOADING)
        self.sidebar.btn_init.setEnabled(False)
        self.sidebar.set_status("⏳ Initializing…")
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
                msg = "Missing models: " + ", ".join(missing)
                self.sidebar.set_status(msg, is_error=True)
                self.log_console.log(msg, "error")
                self._set_pipeline_state(PipelineState.ERROR, msg)
                self.sidebar.btn_init.setEnabled(True)
                return

            self.pipeline_ready = True
            self.sidebar.set_status("✅ Pipeline ready")
            self.log_console.log("Pipeline initialized", "info")
            self.toasts.show_toast("Pipeline ready", "✅")
            self._set_pipeline_state(PipelineState.READY)
        except Exception as e:
            self.sidebar.set_status(f"❌ {e}", is_error=True)
            self.log_console.log(f"Pipeline error: {e}", "error")
            traceback.print_exc()
            self._set_pipeline_state(PipelineState.ERROR, str(e))
        finally:
            self.sidebar.btn_init.setEnabled(True)

    # ── Media actions ───────────────────────────────────────

    def _on_connect_rtsp_live(self):
        url = self.input_rtsp_live.text().strip()
        if url:
            self.app_state.current_source = SourceType.RTSP
            self.app_state.source_path = url
            self.log_console.log("Live RTSP URL set", "info")

    def _on_analyze_live(self):
        url = self.input_rtsp_live.text().strip()
        if not url or not self.pipeline_ready:
            return
        self.app_state.current_source = SourceType.RTSP
        self.app_state.source_path = url
        self._start_processing(url, is_rtsp=True)

    def _on_clear_live(self):
        self.live_hud.setVisible(False)

    def _on_image_detection_clicked(self, idx: int):
        self.app_state.selected_detection_index = idx
        self.img_table.selectRow(idx)
        self.img_table.scrollToItem(self.img_table.item(idx, 0))
        self.live_viewer.highlight_detection(idx)

    def _on_table_row_clicked(self, row, col):
        self.live_viewer.flash_detection(row)
        self.live_viewer.highlight_detection(row)

    def _on_table_hover(self, item):
        if item is None:
            return
        self.live_viewer.highlight_detection(item.row())

    def _resume_processing(self):
        self._on_analyze_live()

    def _start_processing(self, path, is_rtsp=False):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        self.app_state.analytics_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._analytics_frame_idx = 0
        self._analytics_win.bind_history(self._history, self.app_state.analytics_session_id)
        self.worker = PipelineWorker(rtsp_url=path, config=self.config)
        self.worker.started_processing.connect(self._worker_started)
        self.worker.frame_processed.connect(self._worker_frame)
        self.worker.finished_processing.connect(self._worker_finished)
        self.worker.error_occurred.connect(self._worker_error)
        self._last_frame_time = time.time()
        self._set_pipeline_state(PipelineState.RUNNING)
        self.worker.start()
        self.log_console.log(f"Processing started: {str(path)}", "info")

    def _worker_started(self):
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

    @pyqtSlot(dict)
    def _worker_frame(self, results):
        now = time.time()
        dt = now - self._last_frame_time
        self._fps = 1.0 / dt if dt > 0 else 0.0
        self._last_frame_time = now

        device = results.get("device", "CPU")
        lat = float(results.get("inference_time_ms", 0))
        self.app_state.device = device
        self.perf_panel.set_device(device)
        self.perf_panel.push_sample(self._fps, lat)
        try:
            self._history.append_result(
                session_id=self.app_state.analytics_session_id or "default",
                ts_ms=int(time.time() * 1000),
                source_type=self.app_state.current_source.name.lower(),
                source_id=str(self.app_state.source_path or ""),
                frame_index=int(self._analytics_frame_idx),
                results=results,
            )
            self._analytics_frame_idx += 1
            self._analytics_win.bind_history(self._history, self.app_state.analytics_session_id)
            self._analytics_win.refresh()
        except Exception as e:
            # History must never break processing; log once per error type.
            self.log_console.log(f"Analytics history write skipped: {e}", "warning")

        self.bus.frame_updated.emit(results)
        dets = results.get("product_detections", [])
        voids = results.get("void_detections", [])
        self.bus.detections_updated.emit(dets, voids)

        self._display_live_frame(results)

    def _worker_finished(self):
        self.progress_bar.setVisible(False)
        self._set_pipeline_state(PipelineState.READY)
        self.log_console.log("Processing finished", "info")
        self.toasts.show_toast("Analysis complete", "✅")

    def _worker_error(self, msg: str):
        self.progress_bar.setVisible(False)
        self.log_console.log(f"Error: {msg}", "error")
        self._set_pipeline_state(PipelineState.ERROR, msg)

    def _display_live_frame(self, results):
        ann = results.get("image")
        if ann is not None:
            self.live_viewer.set_images(None, ann)
        self.live_viewer.set_detections(
            results.get("product_detections", []), results.get("void_detections", [])
        )
        self.live_hud.update_stats(
            fps=self._fps,
            latency_ms=results.get("inference_time_ms", 0),
            device=results.get("device", "CPU"),
        )
        self.live_hud.setVisible(True)

        summary = results.get("summary", {})
        sl = summary.get("stock_levels", {})
        if sl:
            self.img_table.load_data(sl)
        self.report_panel.set_results(results)

    def _export_quick(self):
        self.report_panel._export_pdf()

    def _on_toggle_theme(self):
        app = QApplication.instance()
        new_theme = toggle_theme(app)

        # Update menu label
        if new_theme == "dark":
            self.act_theme.setText("☀️  Switch to Light Theme")
        else:
            self.act_theme.setText("🌙  Switch to Dark Theme")

        # Refresh pyqtgraph plot backgrounds, log console, and inventory window
        self.perf_panel.apply_theme()
        self._analytics_win.apply_theme()
        self._inventory_win.apply_theme()
        self.log_console.apply_theme()

        # Rebuild dock title bar styles (they use inline COLORS)
        self._compact_dock_title_bar(self.dock_log, "Log console")
        self._compact_dock_title_bar(self.dock_void_feed, "Live detection feed")

        # Repaint all custom widgets that use COLORS in paintEvent
        for w in self.findChildren(QWidget):
            w.update()

        self.toasts.show_toast(
            f"{'Dark' if new_theme == 'dark' else 'Light'} theme applied",
            "🌙" if new_theme == "dark" else "☀️"
        )

    def _try_restore_session(self):
        folder = session_manager.load_last_session_path(self._base_dir)
        if not folder:
            return
        try:
            session_manager.load_session_state(folder, self.app_state)
            session_manager.restore_window_layout(self, folder)
            aux = session_manager.load_auxiliary_windows_payload(folder)
            self._restore_auxiliary_ui(aux)
            self.log_console.log(f"Restored session {folder.name}", "info")
        except Exception as e:
            self.log_console.log(f"Session restore skipped: {e}", "warning")

    def closeEvent(self, event):
        try:
            session_manager.save_session(
                self._base_dir,
                self.app_state,
                self.config,
                self,
                detections_payload=self.current_results,
                auxiliary_windows=self._collect_auxiliary_ui_snapshot(),
            )
        except Exception as e:
            print("Session save:", e)
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        super().closeEvent(event)
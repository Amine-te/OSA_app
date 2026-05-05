# ─────────────────────────────────────────────────────────────
# auxiliary_windows.py — detached Configuration, Analytics, Inventory
# ─────────────────────────────────────────────────────────────

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QComboBox,
    QGridLayout,
    QMainWindow,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ui.performance_panel import PerformancePanel
from ui.report_panel import ReportPanel
from ui.sidebar import SidebarPanel
from ui.styles import COLORS
from ui.widgets import StockTable

import pyqtgraph as pg
import time
import statistics


class ConfigWindow(QMainWindow):
    """Pipeline paths, thresholds, and initialize — separate from main canvas."""

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OSA — Configuration")
        self.resize(560, 820)
        self.setMinimumSize(420, 500)
        self.sidebar = SidebarPanel(config, full_window=True)
        self.setCentralWidget(self.sidebar)


class AnalyticsWindow(QMainWindow):
    """Video/Live analytics: history, KPIs, and trends (performance is secondary)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OSA — Analytics")
        self.resize(540, 780)
        self.setMinimumSize(360, 480)

        root = QWidget()
        lay = QVBoxLayout(root)
        lay.setSpacing(10)
        lay.setContentsMargins(12, 12, 12, 12)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        controls.addWidget(QLabel("Range"))
        self.range_combo = QComboBox()
        self.range_combo.addItems(["Last 5 min", "Last 15 min", "Last 60 min", "Full session"])
        self.range_combo.currentIndexChanged.connect(self.refresh)
        controls.addWidget(self.range_combo)
        controls.addSpacing(12)
        controls.addWidget(QLabel("Product trend"))
        self.product_combo = QComboBox()
        self.product_combo.addItem("All products")
        self.product_combo.currentIndexChanged.connect(self.refresh)
        controls.addWidget(self.product_combo, stretch=1)
        lay.addLayout(controls)

        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, stretch=1)

        self.page_history = QWidget()
        hl = QVBoxLayout(self.page_history)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(10)
        self.history_hint = QLabel("Run Video Monitoring or Live Camera to populate analytics history.")
        self.history_hint.setWordWrap(True)
        self.history_hint.setStyleSheet(
            f"color:{COLORS['text_secondary']}; background:{COLORS['bg_card']}; "
            f"border:1px solid {COLORS['border']}; border-radius:8px; padding:12px;"
        )
        hl.addWidget(self.history_hint)

        # Session summary (without live inventory KPIs)
        self.kpi_lbl = QLabel("Samples: —   Duration: —   Last update: —")
        self.kpi_lbl.setStyleSheet("font-size: 12px;")
        hl.addWidget(self.kpi_lbl)

        # Trend plots (clear labels + legend)
        stock_title = QLabel("Stock Health Trend")
        stock_title.setStyleSheet("font-weight: 600;")
        stock_sub = QLabel("Overall stock percentage over monitoring time.")
        stock_sub.setProperty("class", "muted")
        hl.addWidget(stock_title)
        hl.addWidget(stock_sub)

        self.plot_stock = pg.PlotWidget()
        self.plot_stock.setBackground(COLORS.get("plot_bg", COLORS["bg_card"]))
        self.plot_stock.showGrid(x=True, y=True, alpha=0.12)
        self.plot_stock.setLabel("left", "Stock (%)", color=COLORS["text_muted"])
        self.plot_stock.setLabel("bottom", "Elapsed time (s)", color=COLORS["text_muted"])
        self.plot_stock.setYRange(0, 100)
        self.plot_stock.addLegend(offset=(8, 8))
        self.stock_curve = self.plot_stock.plot(
            pen=pg.mkPen(color=COLORS["success"], width=2),
            name="Stock %",
        )
        self.product_curve = self.plot_stock.plot(
            pen=pg.mkPen(color=COLORS["warning"], width=2, style=Qt.PenStyle.DashLine),
            name="Selected product %",
        )
        hl.addWidget(self.plot_stock, stretch=1)

        events_title = QLabel("Shelf Events Trend")
        events_title.setStyleSheet("font-weight: 600;")
        events_sub = QLabel("Missing products and void detections over monitoring time.")
        events_sub.setProperty("class", "muted")
        hl.addWidget(events_title)
        hl.addWidget(events_sub)

        self.plot_events = pg.PlotWidget()
        self.plot_events.setBackground(COLORS.get("plot_bg", COLORS["bg_card"]))
        self.plot_events.showGrid(x=True, y=True, alpha=0.12)
        self.plot_events.setLabel("left", "Count", color=COLORS["text_muted"])
        self.plot_events.setLabel("bottom", "Elapsed time (s)", color=COLORS["text_muted"])
        self.plot_events.addLegend(offset=(8, 8))
        self.missing_curve = self.plot_events.plot(
            pen=pg.mkPen(color=COLORS["danger"], width=2),
            name="Missing products",
        )
        self.void_curve = self.plot_events.plot(
            pen=pg.mkPen(color=COLORS["accent_start"], width=2),
            name="Void detections",
        )
        hl.addWidget(self.plot_events, stretch=1)

        self.page_performance = QWidget()
        pl = QVBoxLayout(self.page_performance)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(10)
        self.perf_panel = PerformancePanel()
        pl.addWidget(self.perf_panel)

        self.page_kpis = QWidget()
        kl = QVBoxLayout(self.page_kpis)
        kl.setContentsMargins(0, 0, 0, 0)
        kl.setSpacing(10)
        kpi_title = QLabel("Operational KPIs")
        kpi_title.setStyleSheet("font-weight: 600; font-size: 13px;")
        kpi_sub = QLabel("Computed from selected range and product filter.")
        kpi_sub.setProperty("class", "muted")
        kl.addWidget(kpi_title)
        kl.addWidget(kpi_sub)
        self.kpi_grid_host = QWidget()
        grid = QGridLayout(self.kpi_grid_host)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.setContentsMargins(0, 0, 0, 0)
        self.kpi_value_labels = {}
        kpi_defs = [
            ("avg_stock", "Average Stock %"),
            ("stock_vol", "Stock Volatility (std)"),
            ("time_under", "Time Under 80%"),
            ("max_missing", "Max Missing (overall)"),
            ("avg_void", "Average Voids (overall)"),
            ("trend", "Trend Direction"),
        ]
        for idx, (key, title) in enumerate(kpi_defs):
            r = idx // 2
            c = (idx % 2) * 2
            t = QLabel(title)
            t.setProperty("class", "muted")
            v = QLabel("—")
            v.setStyleSheet("font-size: 16px; font-weight: 700;")
            grid.addWidget(t, r, c)
            grid.addWidget(v, r, c + 1)
            self.kpi_value_labels[key] = v
        kl.addWidget(self.kpi_grid_host)
        kl.addStretch()

        self.tabs.addTab(self.page_history, "History")
        self.tabs.addTab(self.page_kpis, "KPIs")
        self.tabs.addTab(self.page_performance, "Performance")

        self.setCentralWidget(root)

        self._store = None
        self._session_id = ""
        self._suspend_product_events = False

    def apply_theme(self) -> None:
        """Refresh pyqtgraph plot backgrounds after a theme switch."""
        bg = COLORS.get("plot_bg", COLORS["bg_card"])
        self.plot_stock.setBackground(bg)
        self.plot_events.setBackground(bg)
        
        # Also refresh axis pen colors if possible, but at least text colors will update via global QSS
        
        self.history_hint.setStyleSheet(
            f"background:{COLORS['bg_card']}; border:1px solid {COLORS['border']}; "
            f"border-radius:8px; padding:12px;"
        )

    def set_enabled_for_source(self, enabled: bool) -> None:
        """When disabled (Image Inspection), show an empty-state message."""
        self.tabs.setEnabled(enabled)
        if enabled:
            self.history_hint.setText(
                "Run Video Monitoring or Live Camera to populate analytics history."
            )
        else:
            self.history_hint.setText(
                "Analytics is intended for Video Monitoring and Live Camera (history over time)."
            )

    def bind_history(self, store, session_id: str) -> None:
        self._store = store
        self._session_id = session_id or ""
        self.refresh()

    def refresh(self) -> None:
        if self._suspend_product_events:
            return
        if not self._store or not self._session_id:
            return
        try:
            k = self._store.query_kpis(session_id=self._session_id)
            range_minutes = {0: 5, 1: 15, 2: 60}.get(self.range_combo.currentIndex())
            now_ms = time.time() * 1000.0
            since_ts_ms = int(now_ms - (range_minutes * 60 * 1000)) if range_minutes else None

            # Keep product list aligned with selected time range.
            names = self._store.list_products(session_id=self._session_id, since_ts_ms=since_ts_ms)
            current = self.product_combo.currentText()
            self._suspend_product_events = True
            self.product_combo.clear()
            self.product_combo.addItem("All products")
            for n in names:
                self.product_combo.addItem(n)
            idx = self.product_combo.findText(current)
            self.product_combo.setCurrentIndex(idx if idx >= 0 else 0)
            self._suspend_product_events = False

            series_stock = self._store.query_series(
                session_id=self._session_id,
                metric="stock_pct",
                since_ts_ms=since_ts_ms,
            )
            duration_s = 0.0
            last_update = "—"
            if series_stock:
                duration_s = max(0.0, (series_stock[-1][0] - series_stock[0][0]) / 1000.0)
                last_update = f"{series_stock[-1][0] / 1000.0:.1f}s"
            self.kpi_lbl.setText(
                f"Samples: {k.total_samples}   Duration: {duration_s:.1f}s   Last update: {last_update}"
            )

            series_missing = self._store.query_series(
                session_id=self._session_id,
                metric="missing_products",
                since_ts_ms=since_ts_ms,
            )
            series_voids = self._store.query_series(
                session_id=self._session_id,
                metric="void_count",
                since_ts_ms=since_ts_ms,
            )
            selected_product = self.product_combo.currentText()
            series_stock_for_kpi = series_stock
            if selected_product and selected_product != "All products":
                series_stock_for_kpi = self._store.query_product_series(
                    session_id=self._session_id,
                    product_name=selected_product,
                    since_ts_ms=since_ts_ms,
                )

            self._refresh_kpi_tab(
                stock_series=series_stock_for_kpi,
                missing_series=series_missing,
                void_series=series_voids,
            )

            if series_stock:
                t0 = series_stock[0][0]
                xs = [(ts - t0) / 1000.0 for ts, _ in series_stock]
                self.stock_curve.setData(xs, [v for _, v in series_stock])
                if selected_product and selected_product != "All products":
                    pseries = series_stock_for_kpi
                    if pseries:
                        px = [(ts - t0) / 1000.0 for ts, _ in pseries]
                        self.product_curve.setData(px, [v for _, v in pseries])
                    else:
                        self.product_curve.setData([], [])
                else:
                    self.product_curve.setData([], [])
            else:
                self.stock_curve.setData([], [])
                self.product_curve.setData([], [])
            if series_missing:
                t0 = series_missing[0][0]
                xs = [(ts - t0) / 1000.0 for ts, _ in series_missing]
                self.missing_curve.setData(xs, [v for _, v in series_missing])
            else:
                self.missing_curve.setData([], [])
            if series_voids:
                t0 = series_voids[0][0]
                xs = [(ts - t0) / 1000.0 for ts, _ in series_voids]
                self.void_curve.setData(xs, [v for _, v in series_voids])
            else:
                self.void_curve.setData([], [])
        except Exception:
            # Keep UI resilient; failures here should not break the app.
            return

    def _refresh_kpi_tab(self, *, stock_series, missing_series, void_series):
        stocks = [float(v) for _, v in (stock_series or [])]
        miss = [float(v) for _, v in (missing_series or [])]
        voids = [float(v) for _, v in (void_series or [])]

        if stocks:
            avg_stock = sum(stocks) / len(stocks)
            vol = statistics.pstdev(stocks) if len(stocks) > 1 else 0.0
            under = sum(1 for v in stocks if v < 80.0)
            under_pct = (under / len(stocks)) * 100.0
            trend = self._trend_label(stocks)
            self.kpi_value_labels["avg_stock"].setText(f"{avg_stock:.1f}%")
            self.kpi_value_labels["stock_vol"].setText(f"{vol:.2f}")
            self.kpi_value_labels["time_under"].setText(f"{under_pct:.1f}%")
            self.kpi_value_labels["trend"].setText(trend)
        else:
            self.kpi_value_labels["avg_stock"].setText("—")
            self.kpi_value_labels["stock_vol"].setText("—")
            self.kpi_value_labels["time_under"].setText("—")
            self.kpi_value_labels["trend"].setText("—")

        if miss:
            self.kpi_value_labels["max_missing"].setText(f"{max(miss):.0f}")
        else:
            self.kpi_value_labels["max_missing"].setText("—")
        if voids:
            self.kpi_value_labels["avg_void"].setText(f"{(sum(voids)/len(voids)):.2f}")
        else:
            self.kpi_value_labels["avg_void"].setText("—")

    def _trend_label(self, ys):
        if len(ys) < 2:
            return "Stable"
        n = len(ys)
        x_mean = (n - 1) / 2.0
        y_mean = sum(ys) / n
        num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(ys))
        den = sum((i - x_mean) ** 2 for i in range(n)) or 1.0
        slope = num / den
        if slope > 0.05:
            return "Improving"
        if slope < -0.05:
            return "Declining"
        return "Stable"


class InventoryReportWindow(QMainWindow):
    """Stock table and session report / exports."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("OSA — Inventory & report")
        self.resize(540, 720)
        self.setMinimumSize(400, 400)

        inv_split = QSplitter(Qt.Orientation.Vertical)
        self.img_table = StockTable()
        self.report_panel = ReportPanel()
        inv_split.addWidget(self.img_table)
        inv_split.addWidget(self.report_panel)
        inv_split.setStretchFactor(0, 3)
        inv_split.setStretchFactor(1, 2)
        self.setCentralWidget(inv_split)

    def apply_theme(self) -> None:
        """Propagate theme switch to child widgets."""
        self.img_table.apply_theme()
        self.report_panel.apply_theme()

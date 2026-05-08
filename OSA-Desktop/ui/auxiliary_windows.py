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
    QTableWidget,
    QPushButton,
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
        self.kpi_table = QTableWidget()
        self.kpi_table.setColumnCount(5)
        self.kpi_table.setHorizontalHeaderLabels([
            "Product", "OSA Rate", "OOS Rate", "Peak Missing", "Time Under 80%"
        ])
        self.kpi_table.horizontalHeader().setStretchLastSection(True)
        from PyQt6.QtWidgets import QHeaderView, QTableWidgetItem
        self.kpi_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.kpi_table.verticalHeader().setVisible(False)
        self.kpi_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.kpi_table.setAlternatingRowColors(True)
        self.kpi_table.setStyleSheet(f"""
            QTableWidget {{
                alternate-background-color: {COLORS["bg_secondary"]};
            }}
        """)
        kl.addWidget(self.kpi_table)

        # KPI Evolution Graphs Container
        self.kpi_graphs_container = QWidget()
        kpi_grid = QGridLayout(self.kpi_graphs_container)
        kpi_grid.setContentsMargins(0, 0, 0, 0)
        kpi_grid.setSpacing(8)
        
        self.plot_osa = pg.PlotWidget(title="OSA Rate (%)")
        self.plot_oos = pg.PlotWidget(title="OOS Rate (%)")
        self.plot_peak = pg.PlotWidget(title="Missing Items")
        self.plot_under = pg.PlotWidget(title="Threshold Events (<80%)")
        
        for p in (self.plot_osa, self.plot_oos, self.plot_peak, self.plot_under):
            p.setBackground(COLORS.get("plot_bg", COLORS["bg_card"]))
            p.showGrid(x=True, y=True, alpha=0.12)
            p.setLabel("bottom", "Elapsed time (s)", color=COLORS["text_muted"])
            p.setTitle(p.plotItem.titleLabel.text, color=COLORS["text_primary"], size="10pt")
            
        self.plot_osa.setYRange(0, 105)
        self.plot_oos.setYRange(0, 105)
        self.plot_under.setYRange(-0.1, 1.1)

        self.curve_osa = self.plot_osa.plot(pen=pg.mkPen(color=COLORS["success"], width=2))
        self.curve_oos = self.plot_oos.plot(pen=pg.mkPen(color=COLORS["danger"], width=2))
        self.curve_peak = self.plot_peak.plot(pen=pg.mkPen(color=COLORS["warning"], width=2))
        self.curve_under = self.plot_under.plot(pen=pg.mkPen(color=COLORS["accent_start"], width=2))

        kpi_grid.addWidget(self.plot_osa, 0, 0)
        kpi_grid.addWidget(self.plot_oos, 0, 1)
        kpi_grid.addWidget(self.plot_peak, 1, 0)
        kpi_grid.addWidget(self.plot_under, 1, 1)

        kl.addWidget(self.kpi_graphs_container, stretch=2)
        self.kpi_graphs_container.setVisible(False)

        # Export row
        exp_row = QHBoxLayout()
        exp_lbl = QLabel("📤 Export Data")
        exp_lbl.setStyleSheet(f"font-weight: 600; color: {COLORS['text_secondary']}; font-size: 13px;")
        exp_row.addWidget(exp_lbl)
        exp_row.addStretch()
        self.btn_export_kpi_csv = QPushButton("📊 CSV")
        self.btn_export_kpi_csv.setProperty("class", "secondary")
        self.btn_export_kpi_csv.clicked.connect(self._export_kpis_csv)
        self.btn_export_kpi_json = QPushButton("💾 JSON")
        self.btn_export_kpi_json.setProperty("class", "secondary")
        self.btn_export_kpi_json.clicked.connect(self._export_kpis_json)
        exp_row.addWidget(self.btn_export_kpi_csv)
        exp_row.addWidget(self.btn_export_kpi_json)
        kl.addLayout(exp_row)

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
        
        if hasattr(self, 'plot_osa'):
            for p in (self.plot_osa, self.plot_oos, self.plot_peak, self.plot_under):
                p.setBackground(bg)
                title = p.plotItem.titleLabel.text
                p.setTitle(title, color=COLORS["text_primary"], size="10pt")
        
        # Also refresh axis pen colors if possible, but at least text colors will update via global QSS
        
        self.history_hint.setStyleSheet(
            f"background:{COLORS['bg_card']}; border:1px solid {COLORS['border']}; "
            f"border-radius:8px; padding:12px;"
        )
        self.kpi_table.setStyleSheet(f"""
            QTableWidget {{
                alternate-background-color: {COLORS["bg_secondary"]};
            }}
        """)

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

            self._refresh_kpi_tab(since_ts_ms=since_ts_ms)

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

    def _refresh_kpi_tab(self, since_ts_ms):
        selected = self.product_combo.currentText()
        
        # Clear table
        self.kpi_table.setRowCount(0)
        self._current_kpis = [] # save for export
        
        products_to_eval = []
        if selected == "All products":
            products_to_eval = [self.product_combo.itemText(i) for i in range(1, self.product_combo.count())]
        else:
            products_to_eval = [selected]
            
        self.kpi_table.setRowCount(len(products_to_eval))
        
        for row, prod_name in enumerate(products_to_eval):
            kpis = self._store.compute_osa_kpis(
                session_id=self._session_id, 
                product_name=prod_name, 
                since_ts_ms=since_ts_ms
            )
            kpis["product"] = prod_name
            self._current_kpis.append(kpis)
            
            # Format data
            osa_str = f"{kpis['osa_rate']:.1f}%"
            oos_str = f"{kpis['oos_rate']:.1f}%"
            peak_str = str(kpis['peak_missing'])
            under_str = f"{kpis['time_under_threshold']:.1f}%"
            
            from PyQt6.QtWidgets import QTableWidgetItem
            from PyQt6.QtCore import Qt
            from PyQt6.QtGui import QColor, QBrush
            
            # Row items
            items = [
                QTableWidgetItem(prod_name),
                QTableWidgetItem(osa_str),
                QTableWidgetItem(oos_str),
                QTableWidgetItem(peak_str),
                QTableWidgetItem(under_str),
            ]
            
            # Color coding OSA Rate
            if kpis['osa_rate'] >= 90:
                color = QColor(COLORS["success"])
            elif kpis['osa_rate'] >= 70:
                color = QColor(COLORS["warning"])
            else:
                color = QColor(COLORS["danger"])
            
            for col, item in enumerate(items):
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if col == 1: # Highlight OSA rate column
                    item.setForeground(QBrush(color))
                self.kpi_table.setItem(row, col, item)
                
        # Handle graphs
        if selected == "All products":
            self.kpi_graphs_container.setVisible(False)
            self.curve_osa.setData([], [])
            self.curve_oos.setData([], [])
            self.curve_peak.setData([], [])
            self.curve_under.setData([], [])
        else:
            self.kpi_graphs_container.setVisible(True)
            series = self._store.query_kpi_evolution_series(
                session_id=self._session_id,
                product_name=selected,
                since_ts_ms=since_ts_ms
            )
            
            osa = series["osa_rate"]
            if osa:
                t0 = osa[-1][0] if osa else 0  # the timeseries comes in reversed order from query_series? Wait, query_kpi_evolution_series does `reversed(rows)` which makes it ascending if the original query was `ORDER BY ts_ms DESC`.
                t0 = osa[0][0]
                
                x_osa = [(ts - t0) / 1000.0 for ts, _ in osa]
                y_osa = [v for _, v in osa]
                self.curve_osa.setData(x_osa, y_osa)
                
                oos = series["oos_rate"]
                x_oos = [(ts - t0) / 1000.0 for ts, _ in oos]
                y_oos = [v for _, v in oos]
                self.curve_oos.setData(x_oos, y_oos)
                
                miss = series["missing_count"]
                x_miss = [(ts - t0) / 1000.0 for ts, _ in miss]
                y_miss = [v for _, v in miss]
                self.curve_peak.setData(x_miss, y_miss)
                
                under = series["threshold_events"]
                x_under = [(ts - t0) / 1000.0 for ts, _ in under]
                y_under = [v for _, v in under]
                self.curve_under.setData(x_under, y_under)
            else:
                self.curve_osa.setData([], [])
                self.curve_oos.setData([], [])
                self.curve_peak.setData([], [])
                self.curve_under.setData([], [])

    def _export_kpis_csv(self):
        if not getattr(self, "_current_kpis", None):
            return
        from PyQt6.QtWidgets import QFileDialog
        from datetime import datetime
        import csv
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Save KPIs to CSV",
            f"osa_kpis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV Files (*.csv)"
        )
        if path:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Product", "OSA Rate (%)", "OOS Rate (%)", "Peak Missing", "Time Under 80% (%)"])
                for k in self._current_kpis:
                    writer.writerow([
                        k["product"], 
                        f"{k['osa_rate']:.2f}", 
                        f"{k['oos_rate']:.2f}", 
                        k["peak_missing"], 
                        f"{k['time_under_threshold']:.2f}"
                    ])

    def _export_kpis_json(self):
        if not getattr(self, "_current_kpis", None):
            return
        from PyQt6.QtWidgets import QFileDialog
        from datetime import datetime
        import json
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Save KPIs to JSON",
            f"osa_kpis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json)"
        )
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self._current_kpis, f, indent=2)


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

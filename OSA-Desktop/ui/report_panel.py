"""In-app report preview with PDF / CSV export (PyQt print stack)."""

from datetime import datetime

from PyQt6.QtPrintSupport import QPrinter
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ui.styles import COLORS
from ui.widgets import StaticReportCanvas


class ReportPanel(QWidget):
    """Summary text + chart preview + export actions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        head = QLabel("Session report")
        head.setProperty("class", "heading")
        root.addWidget(head)

        self.summary = QTextBrowser()
        self.summary.setOpenExternalLinks(False)
        self.summary.setMinimumHeight(120)
        root.addWidget(self.summary)

        self.chart = StaticReportCanvas(width=7, height=3, dpi=96)
        root.addWidget(self.chart)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.btn_pdf = QPushButton("Export PDF")
        self.btn_csv = QPushButton("Export CSV")
        self.btn_json = QPushButton("Export JSON")
        for b in (self.btn_pdf, self.btn_csv, self.btn_json):
            b.setProperty("class", "secondary")
            row.addWidget(b)
        row.addStretch()
        root.addLayout(row)

        self.btn_pdf.clicked.connect(self._export_pdf)
        self.btn_csv.clicked.connect(self._export_csv)
        self.btn_json.clicked.connect(self._export_json)

    def set_results(self, results: dict):
        self._results = results
        if not results:
            self.summary.setText("No results available.")
            self.chart.plot_summary({})
            return

        self._build_summary(results)
        self.chart.plot_summary(results.get("summary", {}))

    def apply_theme(self) -> None:
        """Refresh static chart background after theme toggle."""
        if self._results:
            self.chart.plot_summary(self._results.get("summary", {}))
        else:
            self.chart.plot_summary({})

    def _build_summary(self, results: dict):
        s = results.get("summary", {})
        lines = [
            f"<p><b>Total products detected:</b> {s.get('total_products_detected', 0)}</p>",
            f"<p><b>Estimated missing:</b> {s.get('estimated_missing_products', 0)}</p>",
            f"<p><b>Overall stock:</b> {s.get('overall_stock_percentage', 0):.1f}%</p>",
            f"<p><b>Inference:</b> {results.get('inference_time_ms', 0):.0f} ms on {results.get('device', '—')}</p>",
        ]
        self.summary.setHtml("".join(lines))

    def _export_pdf(self):
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save PDF",
            f"osa_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            "PDF (*.pdf)",
        )
        if not path:
            return
        from PyQt6.QtGui import QTextDocument

        html = self.summary.toHtml()
        doc = QTextDocument()
        doc.setHtml(html)
        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
        printer.setOutputFileName(path)
        doc.print(printer)

    def _export_csv(self):
        if not self._results:
            return
        import csv

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save CSV",
            f"osa_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV (*.csv)",
        )
        if not path:
            return
        summary = self._results.get("summary", {})
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Product", "Current", "Missing", "Capacity", "Stock%"])
            for product, data in summary.get("stock_levels", {}).items():
                current = data.get("current_count", 0)
                missing = data.get("missing_count", 0)
                cap = data.get("full_capacity", current + missing)
                pct = data.get("stock_percentage", 0)
                w.writerow([product.title(), current, missing, cap, f"{pct:.1f}"])

    def _export_json(self):
        if not self._results:
            return
        import json

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save JSON",
            f"osa_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON (*.json)",
        )
        if not path:
            return
        with open(path, "w") as f:
            json.dump(self._results, f, indent=2, default=str)

"""Non-blocking top error strip with retry and clear."""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton

from ui.styles import COLORS


class ErrorBanner(QFrame):
    """Dismissible error banner docked below the header."""

    retry_clicked = pyqtSignal()
    clear_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setVisible(False)
        self.setStyleSheet(
            f"""
            QFrame {{
                background: rgba(248, 81, 73, 0.12);
                border: 1px solid {COLORS['danger']};
                border-radius: 8px;
            }}
            """
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(8)

        self.msg = QLabel("")
        self.msg.setWordWrap(True)
        self.msg.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 13px;")
        lay.addWidget(self.msg, stretch=1)

        self.btn_retry = QPushButton("Retry")
        self.btn_retry.setProperty("class", "secondary")
        self.btn_retry.clicked.connect(self.retry_clicked.emit)
        lay.addWidget(self.btn_retry)

        self.btn_clear = QPushButton("Dismiss")
        self.btn_clear.setProperty("class", "secondary")
        self.btn_clear.clicked.connect(self._on_clear)
        lay.addWidget(self.btn_clear)

    def _on_clear(self):
        self.setVisible(False)
        self.clear_clicked.emit()

    def show_error(self, text: str):
        self.msg.setText(text)
        self.setVisible(True)

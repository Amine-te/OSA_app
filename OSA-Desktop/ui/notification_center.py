# ─────────────────────────────────────────────────────────────
# notification_center.py — Slide-in notification panel + alert cards
# Displays alert history, severity badges, and notification settings.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import time
from datetime import datetime
from typing import List

from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtSignal
from PyQt6.QtGui import (
    QPainter, QColor, QFont, QBrush, QPen, QPainterPath, QLinearGradient,
)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QSizePolicy, QSpinBox, QCheckBox,
    QGraphicsDropShadowEffect, QApplication,
)

from ui.styles import COLORS
from core.notification_engine import Alert, AlertSeverity, NotificationConfig


# ── Single Alert Card ───────────────────────────────────────

class AlertCard(QWidget):
    """Compact card displaying one alert with severity color bar."""

    dismissed = pyqtSignal(object)  # emits the Alert

    _SEVERITY_COLORS = {
        AlertSeverity.CRITICAL: "#DC2626",
        AlertSeverity.WARNING:  "#D97706",
        AlertSeverity.INFO:     "#16A34A",
    }

    def __init__(self, alert: Alert, parent=None):
        super().__init__(parent)
        self.alert = alert
        self.setFixedHeight(72)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._build_ui()
        self.apply_theme()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 8, 8)
        layout.setSpacing(10)

        # Severity dot
        self._dot = QLabel("●")
        self._dot.setFixedWidth(18)
        self._dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._dot)

        # Text block
        text_block = QVBoxLayout()
        text_block.setContentsMargins(0, 0, 0, 0)
        text_block.setSpacing(2)

        # Header: severity + product
        self._header_lbl = QLabel(f"{self.alert.icon}  {self.alert.severity_label}")
        text_block.addWidget(self._header_lbl)

        # Message
        self._msg_lbl = QLabel(self.alert.message)
        self._msg_lbl.setWordWrap(True)
        text_block.addWidget(self._msg_lbl)

        # Timestamp
        ts_str = datetime.fromtimestamp(self.alert.timestamp).strftime("%H:%M:%S")
        self._ts_lbl = QLabel(ts_str)
        text_block.addWidget(self._ts_lbl)

        layout.addLayout(text_block, stretch=1)

        # Dismiss button
        self._btn_x = QPushButton("×")
        self._btn_x.setFixedSize(22, 22)
        self._btn_x.setFlat(True)
        self._btn_x.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_x.clicked.connect(lambda: self.dismissed.emit(self.alert))
        layout.addWidget(self._btn_x, alignment=Qt.AlignmentFlag.AlignTop)

    def apply_theme(self):
        """Re-apply all inline styles from current COLORS."""
        severity_color = self._SEVERITY_COLORS.get(self.alert.severity, COLORS["text_muted"])
        self._dot.setStyleSheet(
            f"color: {severity_color}; font-size: 16px; background: transparent;"
        )
        self._header_lbl.setStyleSheet(
            f"font-size: 12px; font-weight: 700; color: {severity_color}; background: transparent;"
        )
        self._msg_lbl.setStyleSheet(
            f"font-size: 11px; color: {COLORS['text_primary']}; background: transparent;"
        )
        self._ts_lbl.setStyleSheet(
            f"font-size: 10px; color: {COLORS['text_muted']}; background: transparent;"
        )
        self._btn_x.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 14px; font-weight: 700; "
            f"border: none; background: transparent;"
        )
        self.update()  # repaint the card background / severity bar

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()

        # Card background
        path = QPainterPath()
        path.addRoundedRect(
            rect.x() + 1, rect.y() + 1, rect.width() - 2, rect.height() - 2,
            8, 8
        )
        painter.setPen(QPen(QColor(COLORS["border"]), 1))
        painter.setBrush(QColor(COLORS["bg_card"]))
        painter.drawPath(path)

        # Left severity bar
        severity_color = self._SEVERITY_COLORS.get(self.alert.severity, COLORS["text_muted"])
        bar = QPainterPath()
        bar.addRoundedRect(2, 8, 4, rect.height() - 16, 2, 2)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(severity_color))
        painter.drawPath(bar)

        painter.end()
        super().paintEvent(event)


# ── Settings Panel (inline in the notification center) ──────

class NotificationSettingsPanel(QWidget):
    """Inline settings for notification thresholds and behavior."""

    settings_changed = pyqtSignal(object)  # emits NotificationConfig

    def __init__(self, config: NotificationConfig, parent=None):
        super().__init__(parent)
        self._config = config
        self._build_ui()
        self.apply_theme()
        self.setVisible(False)  # collapsed by default

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 12)
        layout.setSpacing(8)

        # Section header
        self._hdr = QLabel("Notification Settings")
        layout.addWidget(self._hdr)

        # ── Warning threshold ────────────────────────────────
        row1 = QHBoxLayout()
        self._lbl_warning = QLabel("Warning threshold (%)")
        row1.addWidget(self._lbl_warning)
        self.spin_warning = QSpinBox()
        self.spin_warning.setRange(10, 95)
        self.spin_warning.setValue(int(self._config.warning_threshold))
        self.spin_warning.setSuffix("%")
        self.spin_warning.setFixedWidth(80)
        self.spin_warning.valueChanged.connect(self._on_change)
        row1.addWidget(self.spin_warning)
        layout.addLayout(row1)

        # ── Critical threshold ───────────────────────────────
        row2 = QHBoxLayout()
        self._lbl_critical = QLabel("Critical threshold (%)")
        row2.addWidget(self._lbl_critical)
        self.spin_critical = QSpinBox()
        self.spin_critical.setRange(5, 90)
        self.spin_critical.setValue(int(self._config.critical_threshold))
        self.spin_critical.setSuffix("%")
        self.spin_critical.setFixedWidth(80)
        self.spin_critical.valueChanged.connect(self._on_change)
        row2.addWidget(self.spin_critical)
        layout.addLayout(row2)

        # ── Cooldown ─────────────────────────────────────────
        row3 = QHBoxLayout()
        self._lbl_cooldown = QLabel("Cooldown (seconds)")
        row3.addWidget(self._lbl_cooldown)
        self.spin_cooldown = QSpinBox()
        self.spin_cooldown.setRange(5, 300)
        self.spin_cooldown.setValue(int(self._config.cooldown_seconds))
        self.spin_cooldown.setSuffix("s")
        self.spin_cooldown.setFixedWidth(80)
        self.spin_cooldown.valueChanged.connect(self._on_change)
        row3.addWidget(self.spin_cooldown)
        layout.addLayout(row3)

        # ── Checkboxes ───────────────────────────────────────
        self.chk_aggregate = QCheckBox("Alert on overall stock level")
        self.chk_aggregate.setChecked(self._config.alert_on_aggregate)
        self.chk_aggregate.toggled.connect(self._on_change)
        layout.addWidget(self.chk_aggregate)

        self.chk_products = QCheckBox("Alert on individual products")
        self.chk_products.setChecked(self._config.alert_on_products)
        self.chk_products.toggled.connect(self._on_change)
        layout.addWidget(self.chk_products)

        self.chk_recovery = QCheckBox("Alert on stock recovery")
        self.chk_recovery.setChecked(self._config.alert_on_recovery)
        self.chk_recovery.toggled.connect(self._on_change)
        layout.addWidget(self.chk_recovery)

        self.chk_sound = QCheckBox("Enable notification sound")
        self.chk_sound.setChecked(self._config.sound_enabled)
        self.chk_sound.toggled.connect(self._on_change)
        layout.addWidget(self.chk_sound)

        # Separator
        self._sep = QFrame()
        self._sep.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(self._sep)

    def apply_theme(self):
        """Re-apply all inline styles from current COLORS."""
        self._hdr.setStyleSheet(
            f"font-size: 13px; font-weight: 700; color: {COLORS['text_primary']}; "
            f"background: transparent;"
        )
        _lbl_ss = (
            f"font-size: 12px; color: {COLORS['text_secondary']}; background: transparent;"
        )
        self._lbl_warning.setStyleSheet(_lbl_ss)
        self._lbl_critical.setStyleSheet(_lbl_ss)
        self._lbl_cooldown.setStyleSheet(_lbl_ss)
        self._sep.setStyleSheet(f"color: {COLORS['border']};")
        self.setStyleSheet(f"background: {COLORS['bg_secondary']};")

    def _on_change(self, *_):
        self._config.warning_threshold = float(self.spin_warning.value())
        self._config.critical_threshold = float(self.spin_critical.value())
        self._config.cooldown_seconds = float(self.spin_cooldown.value())
        self._config.alert_on_aggregate = self.chk_aggregate.isChecked()
        self._config.alert_on_products = self.chk_products.isChecked()
        self._config.alert_on_recovery = self.chk_recovery.isChecked()
        self._config.sound_enabled = self.chk_sound.isChecked()
        self.settings_changed.emit(self._config)

    def get_config(self) -> NotificationConfig:
        return self._config


# ── Notification Center Panel ───────────────────────────────

class NotificationCenter(QWidget):
    """
    Slide-in panel showing alert history, unread badge count,
    inline settings, and clear-all functionality.

    Designed to be placed in a QDockWidget or shown as a popup.
    """

    alert_count_changed = pyqtSignal(int)  # emits unread count

    def __init__(self, config: NotificationConfig = None, parent=None):
        super().__init__(parent)
        self._config = config or NotificationConfig()
        self._alerts: List[Alert] = []
        self._unread_count = 0
        self._settings_visible = False

        self._build_ui()
        self.apply_theme()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Header bar ───────────────────────────────────────
        self._header = QWidget()
        self._header.setFixedHeight(44)
        hlay = QHBoxLayout(self._header)
        hlay.setContentsMargins(12, 0, 8, 0)
        hlay.setSpacing(8)

        self._title_lbl = QLabel("🔔 Notifications")
        hlay.addWidget(self._title_lbl)

        self._badge = QLabel("")
        self._badge.setFixedSize(24, 20)
        self._badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._badge.setVisible(False)
        hlay.addWidget(self._badge)

        hlay.addStretch()

        # Settings toggle
        self._btn_settings = QPushButton("⚙")
        self._btn_settings.setFixedSize(28, 28)
        self._btn_settings.setFlat(True)
        self._btn_settings.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_settings.setToolTip("Notification settings")
        self._btn_settings.clicked.connect(self._toggle_settings)
        hlay.addWidget(self._btn_settings)

        # Clear all
        self._btn_clear = QPushButton("Clear all")
        self._btn_clear.setProperty("class", "secondary")
        self._btn_clear.setFixedHeight(28)
        self._btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_clear.clicked.connect(self.clear_all)
        hlay.addWidget(self._btn_clear)

        layout.addWidget(self._header)

        # ── Settings panel (collapsible) ─────────────────────
        self._settings_panel = NotificationSettingsPanel(self._config)
        self._settings_panel.settings_changed.connect(self._on_settings_changed)
        layout.addWidget(self._settings_panel)

        # ── Scrollable alert list ────────────────────────────
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._list_container = QWidget()
        self._list_layout = QVBoxLayout(self._list_container)
        self._list_layout.setContentsMargins(8, 8, 8, 8)
        self._list_layout.setSpacing(6)
        self._list_layout.addStretch()  # push cards to top

        self._scroll.setWidget(self._list_container)
        layout.addWidget(self._scroll, stretch=1)

        # ── Empty state ──────────────────────────────────────
        self._empty_lbl = QLabel(
            "No notifications yet.\n"
            "Alerts will appear here when stock\n"
            "drops below configured thresholds."
        )
        self._empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._list_layout.insertWidget(0, self._empty_lbl)

    def _toggle_settings(self):
        self._settings_visible = not self._settings_visible
        self._settings_panel.setVisible(self._settings_visible)

    def _on_settings_changed(self, config: NotificationConfig):
        self._config = config

    def get_config(self) -> NotificationConfig:
        return self._settings_panel.get_config()

    # ── Public API ───────────────────────────────────────────

    def push_alerts(self, alerts: List[Alert]):
        """Add new alerts to the panel. Called by MainWindow."""
        if not alerts:
            return

        # Hide empty state
        self._empty_lbl.setVisible(False)

        for alert in alerts:
            card = AlertCard(alert)
            card.dismissed.connect(self._on_dismiss_card)
            # Insert before the stretch at the end
            self._list_layout.insertWidget(self._list_layout.count() - 1, card)
            self._alerts.append(alert)

        self._unread_count += len(alerts)
        self._update_badge()

        # Scroll to bottom (latest)
        QTimer.singleShot(50, self._scroll_to_bottom)

        # Trim old cards if too many
        max_cards = self._config.max_history
        while len(self._alerts) > max_cards:
            self._alerts.pop(0)
            item = self._list_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

    def mark_read(self):
        """Mark all alerts as read (clears badge)."""
        self._unread_count = 0
        self._update_badge()

    def clear_all(self):
        """Remove all alert cards."""
        while self._list_layout.count() > 1:  # keep the stretch
            item = self._list_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        self._alerts.clear()
        self._unread_count = 0
        self._update_badge()
        self._empty_lbl.setVisible(True)
        self._list_layout.insertWidget(0, self._empty_lbl)

    def _on_dismiss_card(self, alert: Alert):
        """Remove a single alert card."""
        if alert in self._alerts:
            self._alerts.remove(alert)
        # Find and remove the card widget
        for i in range(self._list_layout.count()):
            item = self._list_layout.itemAt(i)
            if item and item.widget() and isinstance(item.widget(), AlertCard):
                if item.widget().alert is alert:
                    w = self._list_layout.takeAt(i).widget()
                    w.deleteLater()
                    break
        if not self._alerts:
            self._empty_lbl.setVisible(True)

    def _update_badge(self):
        if self._unread_count > 0:
            self._badge.setText(str(min(self._unread_count, 99)))
            self._badge.setVisible(True)
        else:
            self._badge.setVisible(False)
        self.alert_count_changed.emit(self._unread_count)

    def _scroll_to_bottom(self):
        sb = self._scroll.verticalScrollBar()
        sb.setValue(sb.maximum())

    def apply_theme(self):
        """Refresh ALL colors across the entire panel on theme change."""
        # ── Own background ───────────────────────────────────
        self.setStyleSheet(f"background: {COLORS['bg_primary']};")

        # ── Header bar ───────────────────────────────────────
        self._header.setStyleSheet(
            f"background: {COLORS['bg_card']}; "
            f"border-bottom: 1px solid {COLORS['border']};"
        )
        self._title_lbl.setStyleSheet(
            f"font-size: 14px; font-weight: 700; color: {COLORS['text_primary']}; "
            f"background: transparent; border: none;"
        )
        self._badge.setStyleSheet(
            f"background: {COLORS['danger']}; color: white; "
            f"font-size: 11px; font-weight: 700; border-radius: 10px; border: none;"
        )
        self._btn_settings.setStyleSheet(
            f"font-size: 16px; color: {COLORS['text_muted']}; "
            f"border: none; background: transparent;"
        )

        # ── Scroll area + list container ─────────────────────
        self._scroll.setStyleSheet(
            f"QScrollArea {{ background: {COLORS['bg_primary']}; border: none; }}"
        )
        self._list_container.setStyleSheet(
            f"background: {COLORS['bg_primary']};"
        )

        # ── Empty state label ────────────────────────────────
        self._empty_lbl.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 12px; padding: 40px; "
            f"background: transparent;"
        )

        # ── Settings panel ───────────────────────────────────
        self._settings_panel.apply_theme()

        # ── All alert cards ──────────────────────────────────
        for i in range(self._list_layout.count()):
            item = self._list_layout.itemAt(i)
            if item and item.widget() and isinstance(item.widget(), AlertCard):
                item.widget().apply_theme()

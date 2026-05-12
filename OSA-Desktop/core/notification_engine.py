# ─────────────────────────────────────────────────────────────
# notification_engine.py — Alert evaluation engine for OSA monitoring
# Evaluates per-product and aggregate stock against configurable
# thresholds and emits alerts with cooldown to prevent spam.
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional


class AlertSeverity(Enum):
    """Three-tier severity mapping to visual treatment."""
    INFO = auto()       # Stock recovered / threshold cleared
    WARNING = auto()    # Stock dropped below warning threshold
    CRITICAL = auto()   # Stock dropped below critical threshold


@dataclass
class Alert:
    """Immutable snapshot of a single alert event."""
    timestamp: float              # time.time()
    product: str                  # product name or "__aggregate__"
    severity: AlertSeverity
    message: str
    stock_pct: float              # current stock percentage
    threshold: float              # threshold that was violated
    is_recovery: bool = False     # True if this is a "stock recovered" alert

    @property
    def severity_label(self) -> str:
        return self.severity.name.capitalize()

    @property
    def icon(self) -> str:
        if self.is_recovery:
            return "✅"
        return {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🚨",
        }[self.severity]


@dataclass
class NotificationConfig:
    """User-configurable notification thresholds and behavior."""
    enabled: bool = True
    # Stock percentage thresholds
    warning_threshold: float = 70.0     # Below this → WARNING
    critical_threshold: float = 50.0    # Below this → CRITICAL
    # Cooldown: minimum seconds between repeated alerts for the same product
    cooldown_seconds: float = 30.0
    # Whether to alert on aggregate (overall) stock level
    alert_on_aggregate: bool = True
    # Whether to alert on individual product stock levels
    alert_on_products: bool = True
    # Whether to emit recovery alerts when stock returns above threshold
    alert_on_recovery: bool = True
    # Sound
    sound_enabled: bool = True
    # Max alerts to keep in history
    max_history: int = 200


class NotificationEngine:
    """
    Stateful alert evaluator. Call `evaluate(results_dict)` on each
    inference frame; it returns a list of new `Alert` objects (possibly empty).
    
    Internally tracks per-product cooldowns and previous-state to detect
    threshold crossings (both downward and recovery).
    """

    def __init__(self, config: Optional[NotificationConfig] = None):
        self.config = config or NotificationConfig()
        # product_name -> last alert timestamp (to enforce cooldown)
        self._last_alert_time: Dict[str, float] = {}
        # product_name -> was_below_threshold (to detect crossings)
        self._prev_below_warning: Dict[str, bool] = {}
        self._prev_below_critical: Dict[str, bool] = {}
        # Full alert history
        self.history: List[Alert] = []

    def update_config(self, config: NotificationConfig) -> None:
        """Hot-update configuration without resetting state."""
        self.config = config

    def clear_state(self) -> None:
        """Reset all tracking state (e.g. on new session)."""
        self._last_alert_time.clear()
        self._prev_below_warning.clear()
        self._prev_below_critical.clear()

    def evaluate(self, results: dict) -> List[Alert]:
        """
        Evaluate a single frame's results dict and return any new alerts.
        
        Expected keys in `results`:
            - summary.overall_stock_percentage  (float)
            - summary.stock_levels              (dict[str, dict])
              Each stock_level entry has 'stock_percentage' (float).
        """
        if not self.config.enabled:
            return []

        alerts: List[Alert] = []
        now = time.time()
        summary = results.get("summary", {})
        if not summary:
            return []

        # ── Aggregate stock check ────────────────────────────
        if self.config.alert_on_aggregate:
            overall_pct = summary.get("overall_stock_percentage", 100.0)
            agg_alerts = self._check_product(
                "__aggregate__", overall_pct, now,
                display_name="Overall Stock"
            )
            alerts.extend(agg_alerts)

        # ── Per-product stock check ──────────────────────────
        if self.config.alert_on_products:
            stock_levels = summary.get("stock_levels", {})
            for product_name, data in stock_levels.items():
                pct = data.get("stock_percentage", 100.0)
                prod_alerts = self._check_product(
                    product_name, pct, now,
                    display_name=product_name.title()
                )
                alerts.extend(prod_alerts)

        # Trim history
        self.history.extend(alerts)
        if len(self.history) > self.config.max_history:
            self.history = self.history[-self.config.max_history:]

        return alerts

    def _check_product(
        self, key: str, pct: float, now: float, display_name: str
    ) -> List[Alert]:
        """Check a single product/aggregate against thresholds."""
        alerts: List[Alert] = []

        was_below_warning = self._prev_below_warning.get(key, False)
        was_below_critical = self._prev_below_critical.get(key, False)

        is_below_critical = pct < self.config.critical_threshold
        is_below_warning = pct < self.config.warning_threshold

        # Update state
        self._prev_below_warning[key] = is_below_warning
        self._prev_below_critical[key] = is_below_critical

        # Check cooldown
        last_time = self._last_alert_time.get(key, 0)
        if (now - last_time) < self.config.cooldown_seconds:
            return []

        # ── Critical crossing (downward) ─────────────────────
        if is_below_critical and not was_below_critical:
            alert = Alert(
                timestamp=now,
                product=key,
                severity=AlertSeverity.CRITICAL,
                message=f"{display_name} dropped to {pct:.1f}% — below critical threshold ({self.config.critical_threshold:.0f}%)",
                stock_pct=pct,
                threshold=self.config.critical_threshold,
            )
            alerts.append(alert)
            self._last_alert_time[key] = now

        # ── Warning crossing (downward, not already critical) ─
        elif is_below_warning and not was_below_warning and not is_below_critical:
            alert = Alert(
                timestamp=now,
                product=key,
                severity=AlertSeverity.WARNING,
                message=f"{display_name} dropped to {pct:.1f}% — below warning threshold ({self.config.warning_threshold:.0f}%)",
                stock_pct=pct,
                threshold=self.config.warning_threshold,
            )
            alerts.append(alert)
            self._last_alert_time[key] = now

        # ── Recovery (upward crossing) ───────────────────────
        elif self.config.alert_on_recovery:
            if was_below_critical and not is_below_critical and not is_below_warning:
                alert = Alert(
                    timestamp=now,
                    product=key,
                    severity=AlertSeverity.INFO,
                    message=f"{display_name} recovered to {pct:.1f}% — above thresholds",
                    stock_pct=pct,
                    threshold=self.config.warning_threshold,
                    is_recovery=True,
                )
                alerts.append(alert)
                self._last_alert_time[key] = now
            elif was_below_warning and not is_below_warning:
                alert = Alert(
                    timestamp=now,
                    product=key,
                    severity=AlertSeverity.INFO,
                    message=f"{display_name} recovered to {pct:.1f}% — above warning threshold",
                    stock_pct=pct,
                    threshold=self.config.warning_threshold,
                    is_recovery=True,
                )
                alerts.append(alert)
                self._last_alert_time[key] = now

        return alerts

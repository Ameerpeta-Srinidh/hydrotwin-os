"""
HydroTwin OS — Plane 2: Alert Engine

Processes raw anomaly detections into structured, deduplicated alerts
with severity classification, throttling, and escalation logic.

Features:
    - Severity classification based on anomaly type + confidence
    - Alert deduplication within configurable cooldown windows
    - Time-based escalation (warning → critical if unresolved)
    - Formatted output for Kafka events and operator display
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """A structured operator alert."""
    alert_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    anomaly_type: str = ""          # leak, hotspot, vibration, flow_deviation
    severity: str = "info"          # info, warning, critical
    confidence: float = 0.0
    location: str = ""
    source: str = ""                # sensor, vision, vibration, fusion
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    escalation_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "timestamp": str(self.timestamp),
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "confidence": self.confidence,
            "location": self.location,
            "source": self.source,
            "message": self.message,
            "details": self.details,
            "acknowledged": self.acknowledged,
        }


# ─────────────────────── Severity Rules ───────────────────────

SEVERITY_RULES: dict[str, dict[str, str]] = {
    "leak": {"high": "critical", "medium": "critical", "low": "warning"},
    "hotspot": {"high": "critical", "medium": "warning", "low": "info"},
    "vibration_anomaly": {"high": "critical", "medium": "warning", "low": "info"},
    "vibration_fault": {"high": "critical", "medium": "warning", "low": "info"},
    "flow_deviation": {"high": "warning", "medium": "warning", "low": "info"},
    "corrosion": {"high": "warning", "medium": "info", "low": "info"},
    "cable_damage": {"high": "warning", "medium": "info", "low": "info"},
    "floor_water": {"high": "critical", "medium": "critical", "low": "warning"},
    "spike": {"high": "warning", "medium": "info", "low": "info"},
    "temporal": {"high": "warning", "medium": "info", "low": "info"},
    "multivariate": {"high": "warning", "medium": "info", "low": "info"},
}

MESSAGE_TEMPLATES: dict[str, str] = {
    "leak": "⚠️ Water leak detected at {location} (confidence: {confidence:.0%})",
    "hotspot": "🔥 Thermal hotspot at {location} — {details}",
    "vibration_anomaly": "📳 Vibration anomaly on {location} — {details}",
    "vibration_fault": "📳 Equipment fault on {location} — {details}",
    "flow_deviation": "💧 Flow deviation at {location} (confidence: {confidence:.0%})",
    "corrosion": "🟤 Corrosion detected at {location}",
    "floor_water": "🌊 Floor water detected at {location}",
    "spike": "📈 Sensor spike at {location} — value: {details}",
    "temporal": "⏱️ Temporal pattern anomaly at {location}",
    "multivariate": "🔗 Correlated sensor anomaly detected",
}


class AlertEngine:
    """
    Processes raw detections into structured, deduplicated alerts.
    """

    def __init__(
        self,
        cooldown_seconds: int = 300,
        escalation_timeout_seconds: int = 900,
        max_active_alerts: int = 100,
    ):
        self.cooldown = timedelta(seconds=cooldown_seconds)
        self.escalation_timeout = timedelta(seconds=escalation_timeout_seconds)
        self.max_active = max_active_alerts

        self._active_alerts: dict[str, Alert] = {}
        self._cooldown_tracker: dict[str, datetime] = {}  # dedup key → last alert time
        self._alert_counter = 0

    def process(
        self,
        anomaly_type: str,
        confidence: float,
        location: str = "",
        source: str = "sensor",
        details: dict[str, Any] | None = None,
    ) -> Alert | None:
        """
        Process a raw detection and emit an alert if appropriate.

        Returns None if the detection is deduplicated (within cooldown).
        """
        # Determine severity
        conf_level = "high" if confidence > 0.8 else ("medium" if confidence > 0.5 else "low")
        rules = SEVERITY_RULES.get(anomaly_type, {"high": "info", "medium": "info", "low": "info"})
        severity = rules.get(conf_level, "info")

        # Deduplication check
        dedup_key = f"{anomaly_type}:{location}"
        last_time = self._cooldown_tracker.get(dedup_key)

        now = datetime.utcnow()
        if last_time and (now - last_time) < self.cooldown:
            # Within cooldown — check for escalation instead
            existing = self._active_alerts.get(dedup_key)
            if existing and not existing.acknowledged:
                self._check_escalation(existing, now)
            return None

        # Create alert
        self._alert_counter += 1
        alert_id = f"ALT-{self._alert_counter:05d}"

        detail_str = ""
        if details:
            detail_str = ", ".join(f"{k}={v}" for k, v in list(details.items())[:3])

        template = MESSAGE_TEMPLATES.get(anomaly_type, "Anomaly detected at {location}")
        message = template.format(
            location=location or "unknown",
            confidence=confidence,
            details=detail_str,
        )

        alert = Alert(
            alert_id=alert_id,
            anomaly_type=anomaly_type,
            severity=severity,
            confidence=confidence,
            location=location,
            source=source,
            message=message,
            details=details or {},
        )

        # Track
        self._active_alerts[dedup_key] = alert
        self._cooldown_tracker[dedup_key] = now

        # Evict old alerts
        if len(self._active_alerts) > self.max_active:
            oldest_key = min(self._active_alerts, key=lambda k: self._active_alerts[k].timestamp)
            del self._active_alerts[oldest_key]

        logger.info(f"ALERT [{severity.upper()}] {message}")
        return alert

    def acknowledge(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged."""
        for key, alert in self._active_alerts.items():
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                logger.info(f"Alert {alert_id} acknowledged")
                return True
        return False

    def resolve(self, alert_id: str) -> bool:
        """Resolve and remove an alert."""
        for key, alert in list(self._active_alerts.items()):
            if alert.alert_id == alert_id:
                del self._active_alerts[key]
                if key in self._cooldown_tracker:
                    del self._cooldown_tracker[key]
                logger.info(f"Alert {alert_id} resolved")
                return True
        return False

    @property
    def active_alerts(self) -> list[Alert]:
        return list(self._active_alerts.values())

    @property
    def critical_alerts(self) -> list[Alert]:
        return [a for a in self._active_alerts.values() if a.severity == "critical"]

    def _check_escalation(self, alert: Alert, now: datetime) -> None:
        """Escalate unacknowledged alerts after timeout."""
        age = now - alert.timestamp
        if age > self.escalation_timeout and alert.severity != "critical":
            alert.severity = "critical"
            alert.escalation_count += 1
            alert.message += f" [ESCALATED after {int(age.total_seconds())}s]"
            logger.warning(f"Alert {alert.alert_id} escalated to CRITICAL")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> AlertEngine:
        alert_cfg = config.get("anomaly_detection", {}).get("alert_engine", {})
        return cls(
            cooldown_seconds=alert_cfg.get("cooldown_seconds", 300),
            escalation_timeout_seconds=alert_cfg.get("escalation_timeout_seconds", 900),
            max_active_alerts=alert_cfg.get("max_active_alerts", 100),
        )

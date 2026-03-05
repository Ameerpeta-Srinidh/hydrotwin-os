"""
HydroTwin OS — Plane 2: Incident Tracker

Manages the lifecycle of detected incidents from detection through resolution.
Provides root cause analysis hints by correlating related sensor anomalies
and maintains a searchable historical log.

Incident Lifecycle:
    detected → acknowledged → investigating → resolved
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class IncidentStatus(str, Enum):
    DETECTED = "detected"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"


@dataclass
class IncidentEvent:
    """A timestamped event in the incident's lifecycle."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    status: IncidentStatus = IncidentStatus.DETECTED
    actor: str = "system"          # who made the change
    notes: str = ""


@dataclass
class RootCauseHint:
    """A hypothesis about the root cause of an incident."""
    hypothesis: str
    confidence: float = 0.0        # 0–1
    supporting_evidence: list[str] = field(default_factory=list)
    suggested_action: str = ""


@dataclass
class Incident:
    """A tracked incident."""
    incident_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: IncidentStatus = IncidentStatus.DETECTED

    # Classification
    anomaly_type: str = ""          # leak, hotspot, vibration, etc.
    severity: str = "warning"
    location: str = ""

    # Related alerts
    alert_ids: list[str] = field(default_factory=list)

    # Timeline
    events: list[IncidentEvent] = field(default_factory=list)

    # Root cause analysis
    root_cause_hints: list[RootCauseHint] = field(default_factory=list)
    correlated_sensors: list[str] = field(default_factory=list)

    # Resolution
    resolved_at: datetime | None = None
    resolution_notes: str = ""
    time_to_resolve_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "created_at": str(self.created_at),
            "status": self.status.value,
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "location": self.location,
            "alert_ids": self.alert_ids,
            "events": [{"timestamp": str(e.timestamp), "status": e.status.value,
                         "actor": e.actor, "notes": e.notes} for e in self.events],
            "root_cause_hints": [{"hypothesis": h.hypothesis, "confidence": h.confidence,
                                   "suggested_action": h.suggested_action}
                                  for h in self.root_cause_hints],
            "resolved_at": str(self.resolved_at) if self.resolved_at else None,
            "resolution_notes": self.resolution_notes,
        }


# ─────────────────────── Root Cause Analysis ───────────────────────

# Known correlations: if anomaly X at location Y, check these sensors
ROOT_CAUSE_PATTERNS: dict[str, list[RootCauseHint]] = {
    "leak": [
        RootCauseHint(
            hypothesis="Pipe joint failure or seal degradation",
            confidence=0.7,
            supporting_evidence=["Check pressure sensors in adjacent pipes",
                                  "Review vibration data for water hammer"],
            suggested_action="Isolate pipe section and inspect joints",
        ),
        RootCauseHint(
            hypothesis="Condensation from temperature differential",
            confidence=0.3,
            supporting_evidence=["Compare inlet/outlet temperatures with dewpoint"],
            suggested_action="Adjust supply air temperature or add insulation",
        ),
    ],
    "hotspot": [
        RootCauseHint(
            hypothesis="Airflow obstruction or blanking panel missing",
            confidence=0.6,
            supporting_evidence=["Check adjacent rack temperatures",
                                  "Verify CRAH supply air direction"],
            suggested_action="Inspect rack for blocked airflow and missing blanking panels",
        ),
        RootCauseHint(
            hypothesis="CRAH failure or reduced cooling capacity",
            confidence=0.4,
            supporting_evidence=["Check CRAH supply air temperature",
                                  "Check compressor and fan status"],
            suggested_action="Verify CRAH operating parameters",
        ),
    ],
    "vibration_anomaly": [
        RootCauseHint(
            hypothesis="Pump/fan bearing wear or imbalance",
            confidence=0.7,
            supporting_evidence=["Check bearing temperature",
                                  "Review vibration trend over past 7 days"],
            suggested_action="Schedule preventive bearing replacement",
        ),
    ],
    "flow_deviation": [
        RootCauseHint(
            hypothesis="Partial pipe blockage or valve malfunction",
            confidence=0.6,
            supporting_evidence=["Check pressure difference across the segment",
                                  "Verify valve position sensors"],
            suggested_action="Inspect pipe routing and valve positions",
        ),
    ],
}


class IncidentTracker:
    """
    Manages tracking of anomaly incidents from detection to resolution.
    """

    def __init__(self):
        self._incidents: dict[str, Incident] = {}
        self._resolved: list[Incident] = []
        self._counter = 0

    def create_incident(
        self,
        anomaly_type: str,
        severity: str,
        location: str,
        alert_ids: list[str] | None = None,
    ) -> Incident:
        """Create a new incident from a detected anomaly."""
        self._counter += 1
        incident_id = f"INC-{self._counter:05d}"

        # Get root cause hints
        hints = ROOT_CAUSE_PATTERNS.get(anomaly_type, [])

        incident = Incident(
            incident_id=incident_id,
            anomaly_type=anomaly_type,
            severity=severity,
            location=location,
            alert_ids=alert_ids or [],
            root_cause_hints=hints.copy(),
            events=[IncidentEvent(status=IncidentStatus.DETECTED, notes="Auto-detected by Plane 2")],
        )

        self._incidents[incident_id] = incident
        logger.info(f"Incident created: {incident_id} | {anomaly_type} at {location} [{severity}]")
        return incident

    def transition(
        self,
        incident_id: str,
        new_status: IncidentStatus,
        actor: str = "operator",
        notes: str = "",
    ) -> bool:
        """
        Transition an incident to a new status.

        Valid transitions:
            detected → acknowledged → investigating → resolved/false_positive
        """
        incident = self._incidents.get(incident_id)
        if incident is None:
            logger.warning(f"Incident {incident_id} not found")
            return False

        VALID_TRANSITIONS: dict[IncidentStatus, list[IncidentStatus]] = {
            IncidentStatus.DETECTED: [IncidentStatus.ACKNOWLEDGED, IncidentStatus.FALSE_POSITIVE],
            IncidentStatus.ACKNOWLEDGED: [IncidentStatus.INVESTIGATING, IncidentStatus.RESOLVED, IncidentStatus.FALSE_POSITIVE],
            IncidentStatus.INVESTIGATING: [IncidentStatus.RESOLVED, IncidentStatus.FALSE_POSITIVE],
        }

        allowed = VALID_TRANSITIONS.get(incident.status, [])
        if new_status not in allowed:
            logger.warning(f"Invalid transition: {incident.status.value} → {new_status.value}")
            return False

        incident.status = new_status
        incident.events.append(IncidentEvent(status=new_status, actor=actor, notes=notes))

        if new_status in (IncidentStatus.RESOLVED, IncidentStatus.FALSE_POSITIVE):
            incident.resolved_at = datetime.utcnow()
            incident.resolution_notes = notes
            incident.time_to_resolve_seconds = (
                incident.resolved_at - incident.created_at
            ).total_seconds()

            # Move to resolved list
            self._resolved.append(incident)
            del self._incidents[incident_id]

        logger.info(f"Incident {incident_id}: {incident.status.value} by {actor}")
        return True

    def get_incident(self, incident_id: str) -> Incident | None:
        return self._incidents.get(incident_id)

    @property
    def active_incidents(self) -> list[Incident]:
        return list(self._incidents.values())

    @property
    def resolved_incidents(self) -> list[Incident]:
        return list(self._resolved)

    def search(
        self,
        anomaly_type: str | None = None,
        severity: str | None = None,
        location: str | None = None,
        include_resolved: bool = True,
    ) -> list[Incident]:
        """Search incidents by criteria."""
        all_incidents = list(self._incidents.values())
        if include_resolved:
            all_incidents.extend(self._resolved)

        results = []
        for inc in all_incidents:
            if anomaly_type and inc.anomaly_type != anomaly_type:
                continue
            if severity and inc.severity != severity:
                continue
            if location and location not in inc.location:
                continue
            results.append(inc)

        return results

    def summary(self) -> dict[str, Any]:
        return {
            "active_count": len(self._incidents),
            "resolved_count": len(self._resolved),
            "by_severity": {
                "critical": len([i for i in self._incidents.values() if i.severity == "critical"]),
                "warning": len([i for i in self._incidents.values() if i.severity == "warning"]),
                "info": len([i for i in self._incidents.values() if i.severity == "info"]),
            },
            "by_type": {
                t: len([i for i in self._incidents.values() if i.anomaly_type == t])
                for t in set(i.anomaly_type for i in self._incidents.values())
            } if self._incidents else {},
        }

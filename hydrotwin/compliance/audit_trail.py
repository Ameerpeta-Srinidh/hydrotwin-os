"""
HydroTwin OS — Plane 4: Audit Trail

Immutable, hash-chained event log that records every action and decision
across all planes. Provides tamper-evident storage with query and export
capabilities for regulatory audits.

Features:
    - SHA-256 hash chain for integrity verification
    - Timestamped entries with source plane attribution
    - Query by date range, category, severity, or plane
    - Export to CSV and JSON for external auditors
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    """A single entry in the immutable audit log."""
    entry_id: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source_plane: int = 0          # 1=physics, 2=detection, 3=carbon, 4=compliance
    category: str = ""             # action, decision, violation, alert, calibration
    severity: str = "info"         # info, warning, critical
    actor: str = "system"          # system, operator, rl_agent, regulator
    action: str = ""               # what happened
    details: dict[str, Any] = field(default_factory=dict)
    correlation_id: str = ""       # links to Kafka event
    previous_hash: str = ""        # hash of previous entry
    entry_hash: str = ""           # hash of this entry

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "timestamp": str(self.timestamp),
            "source_plane": self.source_plane,
            "category": self.category,
            "severity": self.severity,
            "actor": self.actor,
            "action": self.action,
            "details": self.details,
            "correlation_id": self.correlation_id,
            "previous_hash": self.previous_hash,
            "entry_hash": self.entry_hash,
        }

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of this entry's content."""
        content = (
            f"{self.entry_id}|{self.timestamp}|{self.source_plane}|"
            f"{self.category}|{self.severity}|{self.actor}|{self.action}|"
            f"{json.dumps(self.details, sort_keys=True, default=str)}|"
            f"{self.correlation_id}|{self.previous_hash}"
        )
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


class AuditTrail:
    """
    Immutable, hash-chained audit log.

    Every entry includes a SHA-256 hash of its contents plus the previous
    entry's hash, forming a tamper-evident chain that auditors can verify.
    """

    def __init__(self):
        self._entries: list[AuditEntry] = []
        self._counter = 0

    def log(
        self,
        action: str,
        source_plane: int = 0,
        category: str = "action",
        severity: str = "info",
        actor: str = "system",
        details: dict[str, Any] | None = None,
        correlation_id: str = "",
    ) -> AuditEntry:
        """
        Add a new entry to the audit trail.

        Returns the created entry with its hash.
        """
        self._counter += 1
        prev_hash = self._entries[-1].entry_hash if self._entries else "GENESIS"

        entry = AuditEntry(
            entry_id=self._counter,
            source_plane=source_plane,
            category=category,
            severity=severity,
            actor=actor,
            action=action,
            details=details or {},
            correlation_id=correlation_id,
            previous_hash=prev_hash,
        )
        entry.entry_hash = entry.compute_hash()
        self._entries.append(entry)

        logger.debug(f"Audit [{entry.entry_id}] {category}/{severity}: {action}")
        return entry

    def verify_integrity(self) -> tuple[bool, list[int]]:
        """
        Verify the hash chain integrity.

        Returns:
            (is_valid, list of tampered entry IDs)
        """
        tampered = []

        for i, entry in enumerate(self._entries):
            # Verify self-hash
            expected_hash = entry.compute_hash()
            if entry.entry_hash != expected_hash:
                tampered.append(entry.entry_id)
                continue

            # Verify chain link
            if i > 0:
                if entry.previous_hash != self._entries[i - 1].entry_hash:
                    tampered.append(entry.entry_id)

        return len(tampered) == 0, tampered

    def query(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        source_plane: int | None = None,
        category: str | None = None,
        severity: str | None = None,
        actor: str | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Query the audit trail with filters."""
        results = []
        for entry in reversed(self._entries):
            if start_date and entry.timestamp < start_date:
                continue
            if end_date and entry.timestamp > end_date:
                continue
            if source_plane is not None and entry.source_plane != source_plane:
                continue
            if category and entry.category != category:
                continue
            if severity and entry.severity != severity:
                continue
            if actor and entry.actor != actor:
                continue
            results.append(entry)
            if len(results) >= limit:
                break
        return results

    def export_json(self, entries: list[AuditEntry] | None = None) -> str:
        """Export entries to JSON string."""
        target = entries or self._entries
        return json.dumps(
            [e.to_dict() for e in target],
            indent=2,
            default=str,
        )

    def export_csv(self, entries: list[AuditEntry] | None = None) -> str:
        """Export entries to CSV string."""
        target = entries or self._entries
        if not target:
            return ""

        output = io.StringIO()
        fieldnames = [
            "entry_id", "timestamp", "source_plane", "category",
            "severity", "actor", "action", "correlation_id",
            "previous_hash", "entry_hash",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for entry in target:
            row = entry.to_dict()
            row.pop("details", None)
            writer.writerow(row)

        return output.getvalue()

    @property
    def entries(self) -> list[AuditEntry]:
        return list(self._entries)

    @property
    def size(self) -> int:
        return len(self._entries)

    def summary(self) -> dict[str, Any]:
        """Summary statistics of the audit trail."""
        by_category: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        by_plane: dict[int, int] = {}

        for e in self._entries:
            by_category[e.category] = by_category.get(e.category, 0) + 1
            by_severity[e.severity] = by_severity.get(e.severity, 0) + 1
            by_plane[e.source_plane] = by_plane.get(e.source_plane, 0) + 1

        return {
            "total_entries": len(self._entries),
            "by_category": by_category,
            "by_severity": by_severity,
            "by_plane": by_plane,
        }

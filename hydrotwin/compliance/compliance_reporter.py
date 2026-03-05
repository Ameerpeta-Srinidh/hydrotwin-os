"""
HydroTwin OS — Plane 4: Compliance Reporter

Auto-generates structured compliance reports covering water usage,
carbon emissions, thermal performance, energy efficiency, and
incident history. Supports daily, weekly, and monthly intervals.

Report Sections:
    1. Executive Summary (compliance score, violations, trends)
    2. Water Usage (WUE, consumption, discharge compliance)
    3. Carbon Emissions (intensity, total, cap utilization)
    4. Thermal Compliance (ASHRAE range, hotspot frequency)
    5. Energy Efficiency (PUE trend, cooling breakdown)
    6. Incidents (anomalies, response times, root causes)
    7. Recommendations (data-driven improvement suggestions)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ReportMetrics:
    """Aggregated metrics for a reporting period."""
    period_start: datetime = field(default_factory=datetime.utcnow)
    period_end: datetime = field(default_factory=datetime.utcnow)
    period_type: str = "daily"     # daily, weekly, monthly

    # Water
    avg_wue: float = 0.0
    max_wue: float = 0.0
    total_water_liters: float = 0.0
    water_violations: int = 0

    # Carbon
    total_carbon_kg: float = 0.0
    avg_carbon_intensity: float = 0.0
    carbon_cap_utilization_pct: float = 0.0
    carbon_violations: int = 0

    # Thermal
    avg_inlet_temp_c: float = 0.0
    max_inlet_temp_c: float = 0.0
    min_inlet_temp_c: float = 0.0
    thermal_violations: int = 0
    hotspot_events: int = 0

    # Energy
    avg_pue: float = 0.0
    min_pue: float = 0.0
    total_it_kwh: float = 0.0
    total_cooling_kwh: float = 0.0

    # Incidents
    total_incidents: int = 0
    critical_incidents: int = 0
    avg_resolution_time_hours: float = 0.0


@dataclass
class ComplianceReport:
    """A structured compliance report."""
    report_id: str = ""
    generated_at: datetime = field(default_factory=datetime.utcnow)
    period_type: str = "daily"
    jurisdiction: str = "EPA_FEDERAL"
    facility_id: str = "HydroTwin-DC-01"
    compliance_score: float = 1.0
    metrics: ReportMetrics = field(default_factory=ReportMetrics)
    violations: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    sections: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "generated_at": str(self.generated_at),
            "period_type": self.period_type,
            "jurisdiction": self.jurisdiction,
            "facility_id": self.facility_id,
            "compliance_score": round(self.compliance_score, 3),
            "violations_count": len(self.violations),
            "recommendations": self.recommendations,
        }


class ComplianceReporter:
    """
    Generates compliance reports from facility metrics and regulation results.
    """

    def __init__(self, facility_id: str = "HydroTwin-DC-01"):
        self.facility_id = facility_id
        self._report_counter = 0
        self._history: list[ComplianceReport] = []

    def generate_report(
        self,
        metrics: ReportMetrics,
        compliance_results: list[dict[str, Any]] | None = None,
        incidents: list[dict[str, Any]] | None = None,
        jurisdiction: str = "EPA_FEDERAL",
    ) -> ComplianceReport:
        """
        Generate a compliance report for a given period.

        Args:
            metrics: aggregated metrics for the period
            compliance_results: list of RuleResult.to_dict() outputs
            incidents: list of incident summaries
            jurisdiction: jurisdiction for the report header
        """
        self._report_counter += 1
        report_id = f"RPT-{self._report_counter:05d}"
        compliance_results = compliance_results or []
        incidents = incidents or []

        # Count violations
        violations = [r for r in compliance_results if r.get("status") == "fail"]

        # Compute compliance score
        if compliance_results:
            total = len(compliance_results)
            passed = sum(1 for r in compliance_results if r.get("status") == "pass")
            warned = sum(1 for r in compliance_results if r.get("status") == "warn")
            score = (passed + 0.5 * warned) / max(total, 1)
        else:
            score = 1.0  # no rules evaluated = fully compliant

        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, violations)

        # Generate report sections
        sections = {
            "executive_summary": self._executive_summary(metrics, score, violations),
            "water_usage": self._water_section(metrics),
            "carbon_emissions": self._carbon_section(metrics),
            "thermal_compliance": self._thermal_section(metrics),
            "energy_efficiency": self._energy_section(metrics),
            "incidents": self._incidents_section(metrics, incidents),
        }

        report = ComplianceReport(
            report_id=report_id,
            period_type=metrics.period_type,
            jurisdiction=jurisdiction,
            facility_id=self.facility_id,
            compliance_score=score,
            metrics=metrics,
            violations=violations,
            recommendations=recommendations,
            sections=sections,
        )

        self._history.append(report)
        logger.info(f"Report {report_id} generated: score={score:.2%}, violations={len(violations)}")
        return report

    def _executive_summary(self, m: ReportMetrics, score: float, violations: list) -> str:
        status = "COMPLIANT" if not violations else f"NON-COMPLIANT ({len(violations)} violations)"
        return (
            f"Compliance Score: {score:.1%} | Status: {status}\n"
            f"Period: {m.period_type.title()} | "
            f"PUE: {m.avg_pue:.2f} | WUE: {m.avg_wue:.2f} | "
            f"Carbon: {m.total_carbon_kg:.0f} kgCO₂ | "
            f"Incidents: {m.total_incidents}"
        )

    def _water_section(self, m: ReportMetrics) -> str:
        return (
            f"Average WUE: {m.avg_wue:.2f} L/kWh | Peak WUE: {m.max_wue:.2f} L/kWh\n"
            f"Total Consumption: {m.total_water_liters:.0f} L | "
            f"Violations: {m.water_violations}"
        )

    def _carbon_section(self, m: ReportMetrics) -> str:
        return (
            f"Total Emissions: {m.total_carbon_kg:.0f} kgCO₂ | "
            f"Avg Intensity: {m.avg_carbon_intensity:.1f} gCO₂/kWh\n"
            f"Cap Utilization: {m.carbon_cap_utilization_pct:.1f}% | "
            f"Violations: {m.carbon_violations}"
        )

    def _thermal_section(self, m: ReportMetrics) -> str:
        return (
            f"Inlet Temp: avg={m.avg_inlet_temp_c:.1f}°C, "
            f"max={m.max_inlet_temp_c:.1f}°C, min={m.min_inlet_temp_c:.1f}°C\n"
            f"Hotspot Events: {m.hotspot_events} | "
            f"Violations: {m.thermal_violations}"
        )

    def _energy_section(self, m: ReportMetrics) -> str:
        return (
            f"Average PUE: {m.avg_pue:.2f} | Best PUE: {m.min_pue:.2f}\n"
            f"IT Energy: {m.total_it_kwh:.0f} kWh | "
            f"Cooling Energy: {m.total_cooling_kwh:.0f} kWh"
        )

    def _incidents_section(self, m: ReportMetrics, incidents: list) -> str:
        return (
            f"Total Incidents: {m.total_incidents} | Critical: {m.critical_incidents}\n"
            f"Avg Resolution Time: {m.avg_resolution_time_hours:.1f} hours"
        )

    def _generate_recommendations(
        self, m: ReportMetrics, violations: list,
    ) -> list[str]:
        """Generate data-driven recommendations."""
        recs = []

        if m.avg_pue > 1.4:
            recs.append(
                f"PUE averaging {m.avg_pue:.2f} — consider increasing economizer utilization "
                f"or optimizing supply air temperature setpoints."
            )

        if m.avg_wue > 1.5:
            recs.append(
                f"WUE averaging {m.avg_wue:.2f} L/kWh — evaluate shifting to mechanical "
                f"cooling during high water stress periods."
            )

        if m.hotspot_events > 5:
            recs.append(
                f"{m.hotspot_events} hotspot events detected — review airflow management "
                f"and blanking panel coverage."
            )

        if m.total_carbon_kg > 0 and m.carbon_cap_utilization_pct > 80:
            recs.append(
                f"Carbon cap at {m.carbon_cap_utilization_pct:.0f}% utilization — "
                f"consider workload migration to lower-carbon regions during peak grid hours."
            )

        for v in violations:
            rule_name = v.get("rule_name", "Unknown")
            recs.append(f"Address violation: {rule_name} — review operational parameters.")

        if not recs:
            recs.append("Facility operating within all compliance thresholds. No actions required.")

        return recs

    @property
    def history(self) -> list[ComplianceReport]:
        return list(self._history)

    def trend_analysis(self) -> dict[str, list[float]]:
        """Return compliance score trend from report history."""
        return {
            "scores": [r.compliance_score for r in self._history],
            "violation_counts": [len(r.violations) for r in self._history],
            "pue_values": [r.metrics.avg_pue for r in self._history],
            "wue_values": [r.metrics.avg_wue for r in self._history],
        }

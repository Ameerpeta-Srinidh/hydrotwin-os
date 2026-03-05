"""
HydroTwin OS — Plane 4: Regulation Engine

Evaluates real-time facility metrics against regulatory requirements
from multiple jurisdictions. Supports water discharge limits, carbon
emission caps, thermal compliance, and noise ordinances.

Jurisdiction Profiles:
    EPA_FEDERAL   — US EPA Clean Water Act + ENERGY STAR
    EU_WFD        — EU Water Framework Directive + EED
    CALIFORNIA    — CA Water Code + Title 24
    SINGAPORE     — PUB Water + NEA Carbon Tax
    CUSTOM        — User-defined rules
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────── Data Types ───────────────────────

class ComplianceStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class RuleCategory(str, Enum):
    WATER = "water"
    CARBON = "carbon"
    THERMAL = "thermal"
    ENERGY = "energy"
    NOISE = "noise"
    REPORTING = "reporting"


@dataclass
class RegulationRule:
    """A single regulatory rule to evaluate."""
    rule_id: str
    name: str
    category: RuleCategory
    jurisdiction: str
    description: str = ""
    metric_key: str = ""           # key in metrics dict
    limit_value: float = 0.0       # threshold
    limit_type: str = "max"        # max, min, range
    range_low: float = 0.0         # for range type
    range_high: float = 0.0        # for range type
    unit: str = ""
    warn_threshold_pct: float = 0.8  # warn at 80% of limit
    citation: str = ""             # legal reference
    penalty_per_violation: str = "" # e.g., "$10,000/day"


@dataclass
class RuleResult:
    """Result of evaluating a single regulation rule."""
    rule: RegulationRule
    status: ComplianceStatus
    actual_value: float = 0.0
    margin_pct: float = 0.0        # how close to limit (0=at limit, 1=at warn)
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule.rule_id,
            "rule_name": self.rule.name,
            "category": self.rule.category.value,
            "jurisdiction": self.rule.jurisdiction,
            "status": self.status.value,
            "actual_value": self.actual_value,
            "limit_value": self.rule.limit_value,
            "margin_pct": round(self.margin_pct, 2),
            "message": self.message,
            "citation": self.rule.citation,
            "timestamp": str(self.timestamp),
        }


@dataclass
class ComplianceScore:
    """Overall compliance score for a jurisdiction."""
    jurisdiction: str
    score: float = 1.0             # 0–1 (1=fully compliant)
    total_rules: int = 0
    passed: int = 0
    warned: int = 0
    failed: int = 0
    results: list[RuleResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "jurisdiction": self.jurisdiction,
            "score": round(self.score, 3),
            "total_rules": self.total_rules,
            "passed": self.passed,
            "warned": self.warned,
            "failed": self.failed,
            "timestamp": str(self.timestamp),
        }


# ─────────────────────── Jurisdiction Profiles ───────────────────────

EPA_RULES: list[RegulationRule] = [
    RegulationRule(
        rule_id="EPA-CWA-001", name="Cooling Water Discharge Temperature",
        category=RuleCategory.WATER, jurisdiction="EPA_FEDERAL",
        description="Max temperature of cooling water discharge",
        metric_key="discharge_temp_c", limit_value=35.0, limit_type="max",
        unit="°C", citation="Clean Water Act §316(a)",
        penalty_per_violation="$25,000/day",
    ),
    RegulationRule(
        rule_id="EPA-CWA-002", name="Daily Water Consumption Limit",
        category=RuleCategory.WATER, jurisdiction="EPA_FEDERAL",
        description="Maximum daily freshwater withdrawal",
        metric_key="daily_water_liters", limit_value=500000.0, limit_type="max",
        unit="liters/day", citation="Clean Water Act §402",
    ),
    RegulationRule(
        rule_id="EPA-ES-001", name="PUE ENERGY STAR Threshold",
        category=RuleCategory.ENERGY, jurisdiction="EPA_FEDERAL",
        description="Power Usage Effectiveness must meet ENERGY STAR",
        metric_key="pue", limit_value=1.5, limit_type="max",
        unit="ratio", warn_threshold_pct=0.9,
        citation="ENERGY STAR Data Center Rating",
    ),
    RegulationRule(
        rule_id="EPA-GHG-001", name="Annual Carbon Emissions Cap",
        category=RuleCategory.CARBON, jurisdiction="EPA_FEDERAL",
        description="Annual CO₂ equivalent emissions limit",
        metric_key="annual_carbon_tonnes", limit_value=50000.0, limit_type="max",
        unit="tCO₂e/year", citation="EPA 40 CFR Part 98",
    ),
]

EU_RULES: list[RegulationRule] = [
    RegulationRule(
        rule_id="EU-WFD-001", name="Water Abstraction Limit",
        category=RuleCategory.WATER, jurisdiction="EU_WFD",
        description="Maximum daily water abstraction",
        metric_key="daily_water_liters", limit_value=400000.0, limit_type="max",
        unit="liters/day", citation="Water Framework Directive 2000/60/EC",
    ),
    RegulationRule(
        rule_id="EU-EED-001", name="Energy Efficiency — PUE Target",
        category=RuleCategory.ENERGY, jurisdiction="EU_WFD",
        description="Data center PUE target under Energy Efficiency Directive",
        metric_key="pue", limit_value=1.3, limit_type="max",
        unit="ratio", citation="Energy Efficiency Directive 2023/1791",
    ),
    RegulationRule(
        rule_id="EU-ETS-001", name="Carbon Emissions Allowance",
        category=RuleCategory.CARBON, jurisdiction="EU_WFD",
        description="EU ETS carbon emissions cap",
        metric_key="annual_carbon_tonnes", limit_value=30000.0, limit_type="max",
        unit="tCO₂e/year", citation="EU ETS Directive 2003/87/EC",
    ),
    RegulationRule(
        rule_id="EU-TEMP-001", name="Server Inlet Temperature — ASHRAE",
        category=RuleCategory.THERMAL, jurisdiction="EU_WFD",
        description="ASHRAE A1 recommended inlet temperature range",
        metric_key="inlet_temp_c", limit_type="range",
        range_low=18.0, range_high=27.0,
        unit="°C", citation="EN 50600-2-3 / ASHRAE TC 9.9",
    ),
]

CALIFORNIA_RULES: list[RegulationRule] = [
    RegulationRule(
        rule_id="CA-WC-001", name="Water Efficiency Standard",
        category=RuleCategory.WATER, jurisdiction="CALIFORNIA",
        description="WUE must not exceed state limit in drought zones",
        metric_key="wue", limit_value=1.8, limit_type="max",
        unit="L/kWh", citation="CA Water Code §10608.48",
    ),
    RegulationRule(
        rule_id="CA-T24-001", name="Title 24 Energy Efficiency",
        category=RuleCategory.ENERGY, jurisdiction="CALIFORNIA",
        description="PUE target per Title 24 building energy code",
        metric_key="pue", limit_value=1.2, limit_type="max",
        unit="ratio", citation="CA Title 24, Part 6 §140.9",
    ),
]

SINGAPORE_RULES: list[RegulationRule] = [
    RegulationRule(
        rule_id="SG-PUB-001", name="PUB Water Conservation",
        category=RuleCategory.WATER, jurisdiction="SINGAPORE",
        description="Water consumption efficiency target",
        metric_key="wue", limit_value=2.0, limit_type="max",
        unit="L/kWh", citation="PUB Water Efficiency Labelling Scheme",
    ),
    RegulationRule(
        rule_id="SG-CT-001", name="Carbon Tax Threshold",
        category=RuleCategory.CARBON, jurisdiction="SINGAPORE",
        description="Carbon tax applies above this threshold",
        metric_key="annual_carbon_tonnes", limit_value=25000.0, limit_type="max",
        unit="tCO₂e/year", citation="Carbon Pricing Act 2018",
        penalty_per_violation="SGD 25/tCO₂e",
    ),
]

JURISDICTION_RULES: dict[str, list[RegulationRule]] = {
    "EPA_FEDERAL": EPA_RULES,
    "EU_WFD": EU_RULES,
    "CALIFORNIA": CALIFORNIA_RULES,
    "SINGAPORE": SINGAPORE_RULES,
}


# ─────────────────────── Regulation Engine ───────────────────────

class RegulationEngine:
    """
    Evaluates facility metrics against regulatory rules.
    Supports multiple jurisdictions evaluated simultaneously.
    """

    def __init__(self, jurisdictions: list[str] | None = None):
        self.jurisdictions = jurisdictions if jurisdictions is not None else ["EPA_FEDERAL"]
        self._rules: list[RegulationRule] = []
        self._custom_rules: list[RegulationRule] = []
        self._load_rules()

    def _load_rules(self):
        """Load rules for all configured jurisdictions."""
        self._rules = []
        for j in self.jurisdictions:
            rules = JURISDICTION_RULES.get(j, [])
            self._rules.extend(rules)
        self._rules.extend(self._custom_rules)
        logger.info(f"Loaded {len(self._rules)} rules for {self.jurisdictions}")

    def add_custom_rule(self, rule: RegulationRule) -> None:
        """Add a custom regulatory rule."""
        self._custom_rules.append(rule)
        self._rules.append(rule)

    def evaluate(self, metrics: dict[str, float]) -> list[RuleResult]:
        """
        Evaluate all rules against current facility metrics.

        Args:
            metrics: dict with keys matching rule metric_keys
                     (pue, wue, inlet_temp_c, daily_water_liters, etc.)
        """
        results = []
        for rule in self._rules:
            result = self._evaluate_rule(rule, metrics)
            results.append(result)
        return results

    def evaluate_by_jurisdiction(self, metrics: dict[str, float]) -> dict[str, ComplianceScore]:
        """Evaluate and group results by jurisdiction."""
        all_results = self.evaluate(metrics)
        scores: dict[str, ComplianceScore] = {}

        for result in all_results:
            j = result.rule.jurisdiction
            if j not in scores:
                scores[j] = ComplianceScore(jurisdiction=j)
            scores[j].results.append(result)
            scores[j].total_rules += 1

            if result.status == ComplianceStatus.PASS:
                scores[j].passed += 1
            elif result.status == ComplianceStatus.WARN:
                scores[j].warned += 1
            else:
                scores[j].failed += 1

        # Compute scores
        for s in scores.values():
            if s.total_rules > 0:
                s.score = (s.passed + 0.5 * s.warned) / s.total_rules

        return scores

    def _evaluate_rule(self, rule: RegulationRule, metrics: dict[str, float]) -> RuleResult:
        """Evaluate a single rule."""
        value = metrics.get(rule.metric_key)

        if value is None:
            return RuleResult(
                rule=rule,
                status=ComplianceStatus.PASS,
                message=f"Metric '{rule.metric_key}' not available — skipped",
            )

        if rule.limit_type == "max":
            return self._check_max(rule, value)
        elif rule.limit_type == "min":
            return self._check_min(rule, value)
        elif rule.limit_type == "range":
            return self._check_range(rule, value)

        return RuleResult(rule=rule, status=ComplianceStatus.PASS, actual_value=value)

    def _check_max(self, rule: RegulationRule, value: float) -> RuleResult:
        """Check if value is within maximum limit."""
        if value > rule.limit_value:
            margin = (value - rule.limit_value) / max(rule.limit_value, 1e-8)
            return RuleResult(
                rule=rule, status=ComplianceStatus.FAIL, actual_value=value,
                margin_pct=-margin,
                message=f"VIOLATION: {rule.name} = {value:.2f} exceeds limit {rule.limit_value} {rule.unit}",
            )

        warn_level = rule.limit_value * rule.warn_threshold_pct
        if value > warn_level:
            margin = (rule.limit_value - value) / max(rule.limit_value - warn_level, 1e-8)
            return RuleResult(
                rule=rule, status=ComplianceStatus.WARN, actual_value=value,
                margin_pct=margin,
                message=f"WARNING: {rule.name} = {value:.2f} approaching limit {rule.limit_value} {rule.unit}",
            )

        margin = (rule.limit_value - value) / max(rule.limit_value, 1e-8)
        return RuleResult(
            rule=rule, status=ComplianceStatus.PASS, actual_value=value,
            margin_pct=margin,
            message=f"COMPLIANT: {rule.name} = {value:.2f} within limit {rule.limit_value} {rule.unit}",
        )

    def _check_min(self, rule: RegulationRule, value: float) -> RuleResult:
        """Check if value meets minimum requirement."""
        if value < rule.limit_value:
            return RuleResult(
                rule=rule, status=ComplianceStatus.FAIL, actual_value=value,
                message=f"VIOLATION: {rule.name} = {value:.2f} below minimum {rule.limit_value} {rule.unit}",
            )
        return RuleResult(
            rule=rule, status=ComplianceStatus.PASS, actual_value=value,
            message=f"COMPLIANT: {rule.name} = {value:.2f} meets minimum {rule.limit_value} {rule.unit}",
        )

    def _check_range(self, rule: RegulationRule, value: float) -> RuleResult:
        """Check if value is within an acceptable range."""
        if value < rule.range_low or value > rule.range_high:
            return RuleResult(
                rule=rule, status=ComplianceStatus.FAIL, actual_value=value,
                message=f"VIOLATION: {rule.name} = {value:.2f} outside range [{rule.range_low}, {rule.range_high}] {rule.unit}",
            )

        # Warn if near edges
        range_span = rule.range_high - rule.range_low
        margin_low = (value - rule.range_low) / range_span
        margin_high = (rule.range_high - value) / range_span
        margin = min(margin_low, margin_high)

        if margin < 0.1:
            return RuleResult(
                rule=rule, status=ComplianceStatus.WARN, actual_value=value,
                margin_pct=margin,
                message=f"WARNING: {rule.name} = {value:.2f} near edge of range [{rule.range_low}, {rule.range_high}] {rule.unit}",
            )

        return RuleResult(
            rule=rule, status=ComplianceStatus.PASS, actual_value=value,
            margin_pct=margin,
            message=f"COMPLIANT: {rule.name} = {value:.2f} within range [{rule.range_low}, {rule.range_high}] {rule.unit}",
        )

    @property
    def rules(self) -> list[RegulationRule]:
        return list(self._rules)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> RegulationEngine:
        comp_cfg = config.get("compliance", {})
        jurisdictions = comp_cfg.get("jurisdictions", ["EPA_FEDERAL"])
        return cls(jurisdictions=jurisdictions)

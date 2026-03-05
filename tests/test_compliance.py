"""
Tests for Plane 4: Regulatory Compliance

Covers regulation engine, audit trail, compliance reporter,
and explainability module.
"""

import json
import pytest
from datetime import datetime, timedelta

from hydrotwin.compliance.regulation_engine import (
    RegulationEngine, RegulationRule, RuleCategory,
    ComplianceStatus, ComplianceScore, RuleResult,
    JURISDICTION_RULES,
)
from hydrotwin.compliance.audit_trail import AuditTrail, AuditEntry
from hydrotwin.compliance.compliance_reporter import (
    ComplianceReporter, ComplianceReport, ReportMetrics,
)
from hydrotwin.compliance.explainability import (
    ExplainabilityEngine, DecisionExplanation, CounterfactualResult,
)


# ═══════════════════════════════════════════════════════════
#  Regulation Engine
# ═══════════════════════════════════════════════════════════

class TestRegulationEngine:

    def test_load_epa_rules(self):
        engine = RegulationEngine(jurisdictions=["EPA_FEDERAL"])
        assert len(engine.rules) == 4

    def test_load_multiple_jurisdictions(self):
        engine = RegulationEngine(jurisdictions=["EPA_FEDERAL", "EU_WFD"])
        assert len(engine.rules) == 8  # 4 EPA + 4 EU

    def test_compliant_metrics_pass(self):
        engine = RegulationEngine(jurisdictions=["EPA_FEDERAL"])
        metrics = {
            "discharge_temp_c": 25.0,
            "daily_water_liters": 100000,
            "pue": 1.2,
            "annual_carbon_tonnes": 20000,
        }
        results = engine.evaluate(metrics)
        assert all(r.status == ComplianceStatus.PASS for r in results)

    def test_violation_detected(self):
        engine = RegulationEngine(jurisdictions=["EPA_FEDERAL"])
        metrics = {
            "discharge_temp_c": 40.0,  # exceeds 35°C limit
            "pue": 1.2,
        }
        results = engine.evaluate(metrics)
        violations = [r for r in results if r.status == ComplianceStatus.FAIL]
        assert len(violations) >= 1
        assert any("VIOLATION" in v.message for v in violations)

    def test_warning_near_limit(self):
        engine = RegulationEngine(jurisdictions=["EPA_FEDERAL"])
        metrics = {
            "discharge_temp_c": 30.0,  # 85% of 35°C limit
            "pue": 1.2,
        }
        results = engine.evaluate(metrics)
        warnings = [r for r in results if r.status == ComplianceStatus.WARN]
        assert len(warnings) >= 1

    def test_range_rule_compliance(self):
        engine = RegulationEngine(jurisdictions=["EU_WFD"])
        metrics = {"inlet_temp_c": 22.0}
        results = engine.evaluate(metrics)
        temp_result = [r for r in results if r.rule.metric_key == "inlet_temp_c"]
        assert len(temp_result) == 1
        assert temp_result[0].status == ComplianceStatus.PASS

    def test_range_rule_violation(self):
        engine = RegulationEngine(jurisdictions=["EU_WFD"])
        metrics = {"inlet_temp_c": 35.0}  # above 27°C
        results = engine.evaluate(metrics)
        temp_result = [r for r in results if r.rule.metric_key == "inlet_temp_c"]
        assert temp_result[0].status == ComplianceStatus.FAIL

    def test_compliance_score_by_jurisdiction(self):
        engine = RegulationEngine(jurisdictions=["EPA_FEDERAL", "CALIFORNIA"])
        metrics = {"pue": 1.2, "wue": 1.0}
        scores = engine.evaluate_by_jurisdiction(metrics)
        assert "EPA_FEDERAL" in scores
        assert "CALIFORNIA" in scores
        for s in scores.values():
            assert 0 <= s.score <= 1.0

    def test_custom_rule(self):
        engine = RegulationEngine(jurisdictions=[])
        custom = RegulationRule(
            rule_id="CUSTOM-001", name="Custom Noise Limit",
            category=RuleCategory.NOISE, jurisdiction="CUSTOM",
            metric_key="noise_db", limit_value=70.0, limit_type="max",
            unit="dB",
        )
        engine.add_custom_rule(custom)
        results = engine.evaluate({"noise_db": 75.0})
        assert len(results) == 1
        assert results[0].status == ComplianceStatus.FAIL

    def test_missing_metric_skipped(self):
        engine = RegulationEngine(jurisdictions=["EPA_FEDERAL"])
        results = engine.evaluate({})  # no metrics
        assert all(r.status == ComplianceStatus.PASS for r in results)

    def test_rule_result_serialization(self):
        engine = RegulationEngine(jurisdictions=["EPA_FEDERAL"])
        results = engine.evaluate({"pue": 1.6})
        d = results[0].to_dict()
        assert "rule_id" in d
        assert "status" in d

    def test_from_config(self):
        config = {"compliance": {"jurisdictions": ["SINGAPORE"]}}
        engine = RegulationEngine.from_config(config)
        assert len(engine.rules) == 2  # Singapore has 2 rules

    def test_all_jurisdictions_have_rules(self):
        for j_name, rules in JURISDICTION_RULES.items():
            assert len(rules) > 0, f"Jurisdiction {j_name} has no rules"


# ═══════════════════════════════════════════════════════════
#  Audit Trail
# ═══════════════════════════════════════════════════════════

class TestAuditTrail:

    def test_log_entry(self):
        trail = AuditTrail()
        entry = trail.log("RL action taken", source_plane=3, category="action")
        assert entry.entry_id == 1
        assert entry.entry_hash != ""
        assert entry.previous_hash == "GENESIS"

    def test_hash_chain(self):
        trail = AuditTrail()
        e1 = trail.log("Action 1")
        e2 = trail.log("Action 2")
        e3 = trail.log("Action 3")
        assert e2.previous_hash == e1.entry_hash
        assert e3.previous_hash == e2.entry_hash

    def test_integrity_verification_passes(self):
        trail = AuditTrail()
        for i in range(10):
            trail.log(f"Action {i}", source_plane=i % 4)
        is_valid, tampered = trail.verify_integrity()
        assert is_valid
        assert len(tampered) == 0

    def test_tamper_detection(self):
        trail = AuditTrail()
        trail.log("Action 1")
        trail.log("Action 2")
        trail.log("Action 3")

        # Tamper with entry 2
        trail._entries[1].action = "TAMPERED ACTION"

        is_valid, tampered = trail.verify_integrity()
        assert not is_valid
        assert 2 in tampered  # entry_id 2 was tampered

    def test_query_by_category(self):
        trail = AuditTrail()
        trail.log("RL action", category="action")
        trail.log("Violation found", category="violation")
        trail.log("Another action", category="action")
        results = trail.query(category="action")
        assert len(results) == 2

    def test_query_by_severity(self):
        trail = AuditTrail()
        trail.log("Info event", severity="info")
        trail.log("Critical alert", severity="critical")
        results = trail.query(severity="critical")
        assert len(results) == 1

    def test_query_by_plane(self):
        trail = AuditTrail()
        trail.log("Physics event", source_plane=1)
        trail.log("Detection event", source_plane=2)
        trail.log("RL event", source_plane=3)
        results = trail.query(source_plane=2)
        assert len(results) == 1
        assert results[0].source_plane == 2

    def test_export_json(self):
        trail = AuditTrail()
        trail.log("Test action", details={"key": "value"})
        json_str = trail.export_json()
        parsed = json.loads(json_str)
        assert len(parsed) == 1
        assert parsed[0]["action"] == "Test action"

    def test_export_csv(self):
        trail = AuditTrail()
        trail.log("Action 1")
        trail.log("Action 2")
        csv_str = trail.export_csv()
        assert "entry_id" in csv_str  # header
        assert "Action 1" in csv_str

    def test_summary(self):
        trail = AuditTrail()
        trail.log("Info", severity="info", source_plane=1)
        trail.log("Warning", severity="warning", source_plane=2)
        trail.log("Critical", severity="critical", source_plane=3)
        s = trail.summary()
        assert s["total_entries"] == 3
        assert s["by_severity"]["critical"] == 1


# ═══════════════════════════════════════════════════════════
#  Compliance Reporter
# ═══════════════════════════════════════════════════════════

class TestComplianceReporter:

    def _make_metrics(self) -> ReportMetrics:
        return ReportMetrics(
            period_type="daily",
            avg_wue=1.2, max_wue=1.8, total_water_liters=50000,
            total_carbon_kg=1500, avg_carbon_intensity=180,
            avg_inlet_temp_c=23.0, max_inlet_temp_c=26.0, min_inlet_temp_c=19.0,
            avg_pue=1.25, min_pue=1.15,
            total_it_kwh=120000, total_cooling_kwh=30000,
            total_incidents=2, critical_incidents=0,
        )

    def test_generate_report(self):
        reporter = ComplianceReporter()
        metrics = self._make_metrics()
        report = reporter.generate_report(metrics)
        assert report.report_id.startswith("RPT-")
        # With no compliance results, score defaults to 100%
        assert report.compliance_score == 1.0

    def test_report_with_violations(self):
        reporter = ComplianceReporter()
        metrics = self._make_metrics()
        violations = [
            {"status": "fail", "rule_name": "PUE Limit", "actual_value": 1.6},
            {"status": "pass", "rule_name": "WUE Limit"},
        ]
        report = reporter.generate_report(metrics, compliance_results=violations)
        assert len(report.violations) == 1
        assert report.compliance_score < 1.0

    def test_report_sections_populated(self):
        reporter = ComplianceReporter()
        metrics = self._make_metrics()
        report = reporter.generate_report(metrics)
        assert "executive_summary" in report.sections
        assert "water_usage" in report.sections
        assert "carbon_emissions" in report.sections
        assert "thermal_compliance" in report.sections

    def test_recommendations_generated(self):
        reporter = ComplianceReporter()
        metrics = self._make_metrics()
        metrics.avg_pue = 1.5  # high PUE
        report = reporter.generate_report(metrics)
        assert len(report.recommendations) > 0
        assert any("PUE" in r for r in report.recommendations)

    def test_report_history(self):
        reporter = ComplianceReporter()
        for _ in range(3):
            reporter.generate_report(self._make_metrics())
        assert len(reporter.history) == 3

    def test_trend_analysis(self):
        reporter = ComplianceReporter()
        for pue in [1.3, 1.25, 1.2]:
            m = self._make_metrics()
            m.avg_pue = pue
            reporter.generate_report(m)
        trends = reporter.trend_analysis()
        assert len(trends["pue_values"]) == 3
        assert trends["pue_values"] == [1.3, 1.25, 1.2]

    def test_report_serialization(self):
        reporter = ComplianceReporter()
        report = reporter.generate_report(self._make_metrics())
        d = report.to_dict()
        assert "report_id" in d
        assert "compliance_score" in d


# ═══════════════════════════════════════════════════════════
#  Explainability
# ═══════════════════════════════════════════════════════════

class TestExplainability:

    def test_explain_rl_action(self):
        engine = ExplainabilityEngine()
        explanation = engine.explain_rl_action(
            action={"cooling_mode_mix": 0.8, "supply_air_temp": 18.0, "fan_speed": 0.7, "economizer_damper": 0.0},
            state={"ambient_temp_c": 40.0, "inlet_temp_c": 28.0, "grid_carbon_intensity": 500, "water_stress_index": 2.0},
            reward=-2.5,
            metrics={"pue": 1.4, "wue": 0.8},
        )
        assert explanation.decision_id.startswith("DEC-")
        assert len(explanation.reasoning) > 0
        assert "chiller" in explanation.action_summary.lower() or "mechanical" in explanation.action_summary.lower()

    def test_explain_high_water_stress(self):
        engine = ExplainabilityEngine()
        explanation = engine.explain_rl_action(
            action={"cooling_mode_mix": 0.9},
            state={"water_stress_index": 4.0, "ambient_temp_c": 30, "inlet_temp_c": 22, "grid_carbon_intensity": 200},
            reward=-1.0,
            metrics={"pue": 1.3, "wue": 0.5},
        )
        assert any("water stress" in r.lower() for r in explanation.reasoning)

    def test_plain_text_output(self):
        engine = ExplainabilityEngine()
        explanation = engine.explain_rl_action(
            action={"cooling_mode_mix": 0.5},
            state={"ambient_temp_c": 25, "inlet_temp_c": 22, "grid_carbon_intensity": 200, "water_stress_index": 1},
            reward=-1.0,
            metrics={"pue": 1.2, "wue": 1.0},
        )
        text = explanation.to_plain_text()
        assert "Decision:" in text
        assert "Why this action" in text

    def test_counterfactual(self):
        engine = ExplainabilityEngine()
        result = engine.counterfactual(
            actual_action={"cooling_mode_mix": 0.8},
            actual_metrics={"pue": 1.4, "wue": 0.5, "reward": -2.0},
            alternative_action={"cooling_mode_mix": 0.2},
            alternative_metrics={"pue": 1.6, "wue": 1.5, "reward": -3.0},
        )
        assert isinstance(result, CounterfactualResult)
        assert "optimal" in result.recommendation.lower()
        assert result.differences["pue"] == pytest.approx(0.2)

    def test_explain_anomaly_response(self):
        engine = ExplainabilityEngine()
        explanation = engine.explain_anomaly_response(
            anomaly_type="leak",
            severity="critical",
            location="Pipe-12",
            response_action="Increased monitoring + dispatched team",
        )
        assert explanation.source_plane == 2
        assert any("critical" in r.lower() for r in explanation.reasoning)
        assert len(explanation.regulatory_links) > 0

    def test_explanation_serialization(self):
        engine = ExplainabilityEngine()
        explanation = engine.explain_rl_action(
            action={"cooling_mode_mix": 0.5},
            state={"ambient_temp_c": 25, "inlet_temp_c": 22, "grid_carbon_intensity": 200, "water_stress_index": 1},
            reward=-1.0,
            metrics={"pue": 1.2, "wue": 1.0},
        )
        d = explanation.to_dict()
        assert "reasoning" in d
        assert "regulatory_links" in d

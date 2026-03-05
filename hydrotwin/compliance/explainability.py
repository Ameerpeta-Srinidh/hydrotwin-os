"""
HydroTwin OS — Plane 4: Explainability Module

Generates plain-language explanations for autonomous decisions made by
the RL agent and other planes. Provides counterfactual analysis and
links decisions back to regulatory requirements.

Capabilities:
    - Decision explanation (why did the agent take this action?)
    - Counterfactual analysis (what would have happened otherwise?)
    - Regulatory linkage (which regulations influenced this decision?)
    - Plain-language summaries for operators and regulators
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DecisionExplanation:
    """A plain-language explanation of an autonomous decision."""
    decision_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source_plane: int = 3          # which plane made the decision
    action_summary: str = ""       # what was decided
    reasoning: list[str] = field(default_factory=list)
    alternative_actions: list[str] = field(default_factory=list)
    regulatory_links: list[str] = field(default_factory=list)
    confidence: float = 0.0
    impact_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "timestamp": str(self.timestamp),
            "source_plane": self.source_plane,
            "action_summary": self.action_summary,
            "reasoning": self.reasoning,
            "alternative_actions": self.alternative_actions,
            "regulatory_links": self.regulatory_links,
            "confidence": self.confidence,
            "impact_summary": self.impact_summary,
        }

    def to_plain_text(self) -> str:
        """Format as a human-readable explanation."""
        lines = [
            f"Decision: {self.action_summary}",
            "",
            "Why this action was taken:",
        ]
        for i, reason in enumerate(self.reasoning, 1):
            lines.append(f"  {i}. {reason}")

        if self.alternative_actions:
            lines.append("")
            lines.append("Alternatives considered:")
            for alt in self.alternative_actions:
                lines.append(f"  • {alt}")

        if self.regulatory_links:
            lines.append("")
            lines.append("Regulatory compliance:")
            for reg in self.regulatory_links:
                lines.append(f"  📋 {reg}")

        lines.append("")
        lines.append(f"Expected impact: {self.impact_summary}")
        return "\n".join(lines)


@dataclass
class CounterfactualResult:
    """Result of a counterfactual analysis."""
    scenario: str
    actual_outcome: dict[str, float]
    counterfactual_outcome: dict[str, float]
    differences: dict[str, float] = field(default_factory=dict)
    recommendation: str = ""


class ExplainabilityEngine:
    """
    Generates explanations for autonomous decisions.
    """

    COOLING_MODE_DESCRIPTIONS = {
        (0.0, 0.3): "primarily evaporative cooling (water-intensive, low-energy)",
        (0.3, 0.7): "a balanced mix of evaporative and mechanical cooling",
        (0.7, 1.0): "primarily mechanical chiller cooling (energy-intensive, water-saving)",
    }

    def __init__(self):
        self._explanations: list[DecisionExplanation] = []
        self._counter = 0

    def explain_rl_action(
        self,
        action: dict[str, float],
        state: dict[str, float],
        reward: float,
        metrics: dict[str, float],
    ) -> DecisionExplanation:
        """
        Explain an RL agent's cooling action.

        Args:
            action: {cooling_mode_mix, supply_air_temp, fan_speed, economizer_damper}
            state: current facility state
            reward: reward received
            metrics: current facility metrics (pue, wue, etc.)
        """
        self._counter += 1
        decision_id = f"DEC-{self._counter:05d}"

        cooling_mix = action.get("cooling_mode_mix", 0.5)
        supply_temp = action.get("supply_air_temp", 20.0)
        fan_speed = action.get("fan_speed", 0.5)
        economizer = action.get("economizer_damper", 0.0)

        # Describe the cooling mode
        mode_desc = "balanced cooling"
        for (low, high), desc in self.COOLING_MODE_DESCRIPTIONS.items():
            if low <= cooling_mix <= high:
                mode_desc = desc
                break

        # Build action summary
        action_summary = (
            f"Set cooling to {mode_desc} "
            f"(mix={cooling_mix:.0%}), "
            f"supply air at {supply_temp:.1f}°C, "
            f"fan at {fan_speed:.0%}, "
            f"economizer at {economizer:.0%}"
        )

        # Build reasoning based on conditions
        reasoning = []
        ambient = state.get("ambient_temp_c", 30.0)
        inlet = state.get("inlet_temp_c", 22.0)
        carbon = state.get("grid_carbon_intensity", 200.0)
        water_stress = state.get("water_stress_index", 1.0)

        if ambient > 35:
            reasoning.append(
                f"High ambient temperature ({ambient:.1f}°C) limits evaporative cooling effectiveness"
            )
        elif ambient < 18:
            reasoning.append(
                f"Low ambient temperature ({ambient:.1f}°C) enables free cooling via economizer"
            )

        if carbon > 400:
            reasoning.append(
                f"High grid carbon intensity ({carbon:.0f} gCO₂/kWh) — "
                f"prioritizing evaporative cooling to reduce electricity use"
            )
        elif carbon < 100:
            reasoning.append(
                f"Clean grid ({carbon:.0f} gCO₂/kWh) — "
                f"mechanical cooling acceptable without carbon penalty"
            )

        if water_stress > 3.0:
            reasoning.append(
                f"High water stress ({water_stress:.1f}/5.0) — "
                f"reducing evaporative cooling to conserve water"
            )

        if inlet > 27:
            reasoning.append(
                f"Inlet temperature ({inlet:.1f}°C) approaching ASHRAE A1 limit — "
                f"increasing cooling capacity"
            )
        elif inlet < 18:
            reasoning.append(
                f"Inlet temperature ({inlet:.1f}°C) below optimal — "
                f"reducing cooling to save energy"
            )

        if not reasoning:
            reasoning.append("Operating within normal parameters — maintaining current efficiency")

        # Alternatives
        alternatives = self._generate_alternatives(cooling_mix, state)

        # Regulatory links
        reg_links = self._regulatory_links(metrics, state)

        # Impact summary
        pue = metrics.get("pue", 1.3)
        wue = metrics.get("wue", 1.0)
        impact = (
            f"PUE={pue:.2f}, WUE={wue:.2f} L/kWh, "
            f"reward={reward:.3f}"
        )

        explanation = DecisionExplanation(
            decision_id=decision_id,
            source_plane=3,
            action_summary=action_summary,
            reasoning=reasoning,
            alternative_actions=alternatives,
            regulatory_links=reg_links,
            confidence=min(1.0, abs(reward) / 5.0),
            impact_summary=impact,
        )

        self._explanations.append(explanation)
        return explanation

    def counterfactual(
        self,
        actual_action: dict[str, float],
        actual_metrics: dict[str, float],
        alternative_action: dict[str, float],
        alternative_metrics: dict[str, float],
    ) -> CounterfactualResult:
        """
        Compare actual action outcome with a hypothetical alternative.
        """
        diffs = {}
        for key in actual_metrics:
            if key in alternative_metrics:
                diffs[key] = alternative_metrics[key] - actual_metrics[key]

        # Generate recommendation
        actual_reward = actual_metrics.get("reward", 0)
        alt_reward = alternative_metrics.get("reward", 0)

        if alt_reward > actual_reward:
            rec = "The alternative action would have produced a better outcome."
        else:
            rec = "The chosen action was optimal compared to this alternative."

        return CounterfactualResult(
            scenario=f"Mix {alternative_action.get('cooling_mode_mix', 0):.0%} vs actual {actual_action.get('cooling_mode_mix', 0):.0%}",
            actual_outcome=actual_metrics,
            counterfactual_outcome=alternative_metrics,
            differences=diffs,
            recommendation=rec,
        )

    def explain_anomaly_response(
        self,
        anomaly_type: str,
        severity: str,
        location: str,
        response_action: str,
    ) -> DecisionExplanation:
        """Explain the system's response to a detected anomaly."""
        self._counter += 1

        reasoning = []
        if severity == "critical":
            reasoning.append(f"Critical {anomaly_type} detected at {location} — immediate response required")
        else:
            reasoning.append(f"{anomaly_type.title()} detected at {location} (severity: {severity})")

        response_map = {
            "leak": "Increased monitoring frequency and alerted facility team for physical inspection",
            "hotspot": "Increased local cooling and added blanking panels to affected zone",
            "vibration_fault": "Scheduled preventive maintenance and shifted load to backup equipment",
            "flow_deviation": "Adjusted valve positions and verified pipe routing integrity",
        }

        action_desc = response_map.get(anomaly_type, f"Responded to {anomaly_type} with standard protocol")
        reasoning.append(action_desc)

        return DecisionExplanation(
            decision_id=f"DEC-{self._counter:05d}",
            source_plane=2,
            action_summary=f"Anomaly response: {response_action}",
            reasoning=reasoning,
            regulatory_links=[
                f"Action logged per {self._regulatory_requirement_for(anomaly_type)}"
            ],
            confidence=0.9 if severity == "critical" else 0.7,
            impact_summary=f"Mitigating {anomaly_type} at {location}",
        )

    def _generate_alternatives(self, cooling_mix: float, state: dict) -> list[str]:
        """Generate alternative actions that were considered."""
        alts = []
        if cooling_mix > 0.5:
            alts.append(
                "Switch to more evaporative cooling to reduce electricity consumption "
                "(rejected: water stress or ambient conditions unfavorable)"
            )
        else:
            alts.append(
                "Switch to more mechanical cooling to reduce water consumption "
                "(rejected: grid carbon intensity too high)"
            )

        alts.append(
            "Do nothing (maintain current setpoints) — "
            "rejected if conditions have meaningfully changed"
        )
        return alts

    def _regulatory_links(self, metrics: dict, state: dict) -> list[str]:
        """Link decisions to regulatory requirements."""
        links = []
        pue = metrics.get("pue", 1.3)
        wue = metrics.get("wue", 1.0)
        inlet = state.get("inlet_temp_c", 22.0)

        if pue > 1.4:
            links.append("ENERGY STAR PUE target (1.5) — action aims to improve efficiency")
        if wue > 1.5:
            links.append("EPA water discharge limits — action reduces water consumption")
        if inlet > 27 or inlet < 18:
            links.append("ASHRAE TC 9.9 A1 guideline (18–27°C) — action corrects temperature")
        if not links:
            links.append("Operating within all regulatory thresholds")
        return links

    def _regulatory_requirement_for(self, anomaly_type: str) -> str:
        """Map anomaly type to relevant regulation."""
        mapping = {
            "leak": "EPA Clean Water Act §316(a) — water discharge monitoring",
            "hotspot": "ASHRAE TC 9.9 thermal guidelines",
            "vibration_fault": "OSHA workplace safety standards",
            "flow_deviation": "EPA water monitoring requirements",
        }
        return mapping.get(anomaly_type, "standard operating procedures")

    @property
    def explanations(self) -> list[DecisionExplanation]:
        return list(self._explanations)

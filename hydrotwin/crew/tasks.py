"""
HydroTwin OS — Plane 4: CrewAI Task Definitions

One Task per agent, scoped to a single control-cycle decision:
  1. action_execution_task   — validate and dispatch SAC setpoint
  2. demand_forecast_task    — produce load forecast and drift report
  3. compliance_audit_task   — regulatory check and audit record
"""

from __future__ import annotations

from crewai import Agent, Task


def build_action_execution_task(
    agent: Agent,
    action_vector: list[float],
    thermal_state: dict,
) -> Task:
    """
    Ask the NexusExecutor to validate and dispatch a SAC action.

    Parameters
    ----------
    agent:         The NexusExecutor CrewAI Agent.
    action_vector: 4-dim continuous SAC output
                   [cooling_mode_mix, supply_air_temp, fan_speed, economiser_damper]
    thermal_state: Current sensor readings from the data-centre environment.
    """
    return Task(
        description=(
            f"The SAC Plane-3 agent has produced the following continuous action "
            f"vector: {action_vector}.\n\n"
            f"Current thermal state snapshot:\n{thermal_state}\n\n"
            "1. Validate each dimension against its operational bounds.\n"
            "2. Apply any safety clamp required by current thermal constraints.\n"
            "3. Provide a brief justification explaining how delayed thermal "
            "   dynamics (2–5 min lag) influenced the chosen setpoints.\n"
            "4. Return a structured dispatch report with final setpoints."
        ),
        expected_output=(
            "A structured JSON-compatible report containing: validated_setpoints "
            "(dict), clamps_applied (list), thermal_rationale (str)."
        ),
        agent=agent,
    )


def build_demand_forecast_task(
    agent: Agent,
    current_it_load_mw: float,
    season: str,
) -> Task:
    """
    Ask the AquiferForecaster for a load forecast and drift assessment.
    """
    return Task(
        description=(
            f"Current IT load: {current_it_load_mw:.2f} MW. Season: {season}.\n\n"
            "1. Produce a 24-hour probabilistic IT-load forecast (P10/P50/P90).\n"
            "2. Estimate groundwater extraction volume over the forecast window.\n"
            "3. Compare the current demand distribution against the replay-buffer "
            "   baseline and quantify any distribution shift (KL divergence estimate).\n"
            "4. Recommend whether the RL agent should apply importance-weighted "
            "   sampling or trigger a partial buffer purge."
        ),
        expected_output=(
            "A report with: load_forecast_mw (dict with p10/p50/p90), "
            "groundwater_extraction_ml (float), drift_detected (bool), "
            "kl_divergence_estimate (float), buffer_action_recommendation (str)."
        ),
        agent=agent,
    )


def build_compliance_audit_task(
    agent: Agent,
    proposed_setpoints: dict,
    daily_extraction_ml: float,
) -> Task:
    """
    Ask the ComplianceGuardian to audit proposed setpoints.
    """
    return Task(
        description=(
            f"Proposed cooling setpoints: {proposed_setpoints}\n"
            f"Cumulative daily groundwater extraction so far: {daily_extraction_ml:.1f} ML\n\n"
            "1. Check proposed setpoints against CGWA extraction licence limits.\n"
            "2. Verify WUE target compliance (target ≤ 1.2).\n"
            "3. Assess carbon-intensity impact against daily budget.\n"
            "4. Approve or block the action, and produce a structured audit record."
        ),
        expected_output=(
            "A compliance report with: approved (bool), violations (list), "
            "wue_status (str), carbon_status (str), audit_record (dict)."
        ),
        agent=agent,
    )


__all__ = [
    "build_action_execution_task",
    "build_demand_forecast_task",
    "build_compliance_audit_task",
]

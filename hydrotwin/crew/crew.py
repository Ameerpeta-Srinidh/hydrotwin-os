"""
HydroTwin OS — Plane 4: HydroTwinCrew

The top-level orchestrator that assembles agents and tasks into a sequential
CrewAI crew for one control-cycle decision loop.

Usage
-----
    from hydrotwin.crew import HydroTwinCrew

    crew = HydroTwinCrew()                       # uses mock LLM by default
    result = crew.run_control_cycle(
        action_vector=[0.4, 18.5, 0.72, 0.3],
        thermal_state={"supply_temp_c": 19.2, "return_temp_c": 27.1, "it_load_mw": 3.4},
        current_it_load_mw=3.4,
        season="summer",
        daily_extraction_ml=142.5,
    )
"""

from __future__ import annotations

import logging
from typing import Any

from crewai import Crew, Process

from hydrotwin.crew.agents import (
    build_aquifer_forecaster,
    build_compliance_guardian,
    build_nexus_executor,
)
from hydrotwin.crew.tasks import (
    build_action_execution_task,
    build_compliance_audit_task,
    build_demand_forecast_task,
)

logger = logging.getLogger(__name__)


class HydroTwinCrew:
    """
    Plane 4 multi-agent coordination layer.

    Orchestrates three specialised CrewAI agents in a sequential process:
      1. AquiferForecaster  → demand forecast + replay-buffer drift detection
      2. NexusExecutor      → SAC action validation + setpoint dispatch
      3. ComplianceGuardian → regulatory audit + approval / block
    """

    def __init__(self, llm=None, verbose: bool = True):
        """
        Parameters
        ----------
        llm:     An optional LLM instance (e.g. ChatOpenAI). If None, agents
                 operate with default CrewAI LLM configuration.
        verbose: Enable per-agent logging.
        """
        self.verbose = verbose
        self._llm = llm

        # Instantiate agents once; reuse across control cycles
        self.nexus_executor = build_nexus_executor(llm=llm)
        self.aquifer_forecaster = build_aquifer_forecaster(llm=llm)
        self.compliance_guardian = build_compliance_guardian(llm=llm)

        logger.info(
            "HydroTwinCrew initialised | agents: NexusExecutor, "
            "AquiferForecaster, ComplianceGuardian"
        )

    def run_control_cycle(
        self,
        action_vector: list[float],
        thermal_state: dict[str, Any],
        current_it_load_mw: float,
        season: str = "summer",
        daily_extraction_ml: float = 0.0,
        proposed_setpoints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute one full decision cycle through all three agents.

        Parameters
        ----------
        action_vector:        4-dim SAC output from Plane 3.
        thermal_state:        Current DC sensor snapshot.
        current_it_load_mw:   Current IT load in megawatts.
        season:               Season label for demand-distribution context.
        daily_extraction_ml:  Cumulative groundwater extraction today (ML).
        proposed_setpoints:   Optional pre-validated setpoints; if None,
                              extracted from action_vector after execution task.

        Returns
        -------
        dict with keys: forecast_report, execution_report, compliance_report, crew_output
        """
        logger.info(
            f"Starting control cycle | action={action_vector} | "
            f"IT load={current_it_load_mw:.2f} MW | season={season}"
        )

        # Build tasks for this cycle
        forecast_task = build_demand_forecast_task(
            agent=self.aquifer_forecaster,
            current_it_load_mw=current_it_load_mw,
            season=season,
        )
        execution_task = build_action_execution_task(
            agent=self.nexus_executor,
            action_vector=action_vector,
            thermal_state=thermal_state,
        )
        audit_task = build_compliance_audit_task(
            agent=self.compliance_guardian,
            proposed_setpoints=proposed_setpoints or {"action_vector": action_vector},
            daily_extraction_ml=daily_extraction_ml,
        )

        # Sequential crew: forecast → execute → audit
        crew = Crew(
            agents=[
                self.aquifer_forecaster,
                self.nexus_executor,
                self.compliance_guardian,
            ],
            tasks=[forecast_task, execution_task, audit_task],
            process=Process.sequential,
            verbose=self.verbose,
        )

        crew_output = crew.kickoff()

        logger.info("Control cycle complete.")
        return {
            "crew_output": str(crew_output),
            "action_vector": action_vector,
            "thermal_state": thermal_state,
            "it_load_mw": current_it_load_mw,
            "season": season,
            "daily_extraction_ml": daily_extraction_ml,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> HydroTwinCrew:
        """Create a HydroTwinCrew from the application config dict."""
        crew_cfg = config.get("crew", {})
        verbose = crew_cfg.get("verbose", True)
        return cls(verbose=verbose)


__all__ = ["HydroTwinCrew"]

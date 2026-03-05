"""
HydroTwin OS — Plane 3: Workload Migration Engine

When local conditions become extreme (compound crisis: severe heat + dirty grid
+ high water stress), the agent triggers workload migration to the geographic
region with the lowest marginal carbon emissions.

This module evaluates crisis conditions, identifies optimal migration targets,
classifies deferrable workloads, and interfaces with Kubernetes for dispatch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from hydrotwin.api_clients.electricity_maps import ElectricityMapsClient

logger = logging.getLogger(__name__)


@dataclass
class MigrationTriggerConditions:
    """Thresholds that must be met to trigger migration."""
    min_ambient_temp_c: float = 40.0
    min_grid_carbon_gco2: float = 350.0
    min_water_stress_index: float = 3.0
    require_all: bool = True  # all conditions must be met


@dataclass
class CandidateRegion:
    """A potential migration target region."""
    zone: str
    name: str
    k8s_context: str
    carbon_intensity: float = 0.0  # filled at query time
    water_stress: float = 0.0


@dataclass
class MigrationRecommendation:
    """A recommendation to migrate workloads."""
    should_migrate: bool
    reason: str
    source_conditions: dict[str, float]
    target_region: CandidateRegion | None
    workloads_to_migrate: list[str]
    estimated_carbon_savings_pct: float = 0.0


class MigrationEngine:
    """
    Evaluates compound crisis conditions and recommends/executes
    workload migration to lower-carbon regions.
    """

    # Default deferrable workload labels
    DEFAULT_DEFERRABLE = [
        "batch-ml-training",
        "video-transcoding",
        "cold-storage-replication",
        "data-pipeline-backfill",
    ]

    def __init__(
        self,
        trigger_conditions: MigrationTriggerConditions | None = None,
        candidate_regions: list[CandidateRegion] | None = None,
        deferrable_labels: list[str] | None = None,
        electricity_client: ElectricityMapsClient | None = None,
    ):
        self.trigger = trigger_conditions or MigrationTriggerConditions()
        self.candidates = candidate_regions or [
            CandidateRegion("US-NW-PACW", "Pacific Northwest", "gke-portland"),
            CandidateRegion("US-MIDA-PJM", "Mid-Atlantic", "gke-virginia"),
            CandidateRegion("CA-QC", "Quebec", "gke-montreal"),
            CandidateRegion("SE", "Sweden", "gke-stockholm"),
        ]
        self.deferrable_labels = deferrable_labels or self.DEFAULT_DEFERRABLE
        self.electricity_client = electricity_client or ElectricityMapsClient(mock_mode=True)

    def evaluate(
        self,
        ambient_temp_c: float,
        grid_carbon_gco2: float,
        water_stress_index: float,
    ) -> MigrationRecommendation:
        """
        Evaluate current conditions and decide whether to recommend migration.

        Returns a MigrationRecommendation with the best target region if
        migration is warranted.
        """
        conditions = {
            "ambient_temp_c": ambient_temp_c,
            "grid_carbon_gco2": grid_carbon_gco2,
            "water_stress_index": water_stress_index,
        }

        # Check trigger conditions
        checks = {
            "high_temp": ambient_temp_c >= self.trigger.min_ambient_temp_c,
            "dirty_grid": grid_carbon_gco2 >= self.trigger.min_grid_carbon_gco2,
            "water_stressed": water_stress_index >= self.trigger.min_water_stress_index,
        }

        if self.trigger.require_all:
            should_migrate = all(checks.values())
        else:
            should_migrate = any(checks.values())

        if not should_migrate:
            triggered = [k for k, v in checks.items() if v]
            return MigrationRecommendation(
                should_migrate=False,
                reason=f"Conditions not met for migration. Triggered: {triggered or 'none'}",
                source_conditions=conditions,
                target_region=None,
                workloads_to_migrate=[],
            )

        # Find best migration target (lowest carbon intensity)
        target = self._find_best_region(grid_carbon_gco2)

        if target is None:
            return MigrationRecommendation(
                should_migrate=False,
                reason="All candidate regions have higher carbon intensity than source.",
                source_conditions=conditions,
                target_region=None,
                workloads_to_migrate=[],
            )

        savings_pct = max(0, (grid_carbon_gco2 - target.carbon_intensity) / grid_carbon_gco2 * 100)

        return MigrationRecommendation(
            should_migrate=True,
            reason=(
                f"Compound crisis detected: temp={ambient_temp_c:.1f}°C, "
                f"carbon={grid_carbon_gco2:.0f} gCO₂/kWh, stress={water_stress_index:.1f}. "
                f"Recommending migration to {target.name} ({target.carbon_intensity:.0f} gCO₂/kWh)."
            ),
            source_conditions=conditions,
            target_region=target,
            workloads_to_migrate=self.deferrable_labels.copy(),
            estimated_carbon_savings_pct=savings_pct,
        )

    def _find_best_region(self, source_carbon: float) -> CandidateRegion | None:
        """Query carbon intensity for all candidate regions and find the best one."""
        zones = [c.zone for c in self.candidates]
        intensities = self.electricity_client.get_multi_region_intensity(zones)

        best_region = None
        best_carbon = source_carbon  # must be better than source

        for candidate, intensity_data in zip(self.candidates, intensities):
            carbon = intensity_data.get("carbon_intensity", 999)
            candidate.carbon_intensity = carbon

            if carbon < best_carbon:
                best_carbon = carbon
                best_region = candidate

        return best_region

    def dispatch_workloads(
        self,
        recommendation: MigrationRecommendation,
    ) -> dict[str, Any]:
        """
        Execute workload migration via Kubernetes.

        NOTE: In production, this would use the Kubernetes Python client
        to actually move pods. Here we simulate the dispatch and return
        a structured result.
        """
        if not recommendation.should_migrate or not recommendation.target_region:
            return {"status": "skipped", "reason": "No migration recommended"}

        target = recommendation.target_region
        workloads = recommendation.workloads_to_migrate

        logger.info(
            f"MIGRATION DISPATCH: {len(workloads)} workload types → "
            f"{target.name} ({target.k8s_context})"
        )

        # In production:
        # from kubernetes import client, config
        # config.load_kube_config(context=target.k8s_context)
        # ... reschedule pods with matching labels

        return {
            "status": "dispatched",
            "target_region": target.name,
            "target_context": target.k8s_context,
            "target_carbon_intensity": target.carbon_intensity,
            "workloads_migrated": workloads,
            "estimated_savings_pct": recommendation.estimated_carbon_savings_pct,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> MigrationEngine:
        mig_cfg = config.get("migration", {})
        trigger_cfg = mig_cfg.get("trigger_conditions", {})
        candidates_cfg = mig_cfg.get("candidate_regions", [])

        trigger = MigrationTriggerConditions(
            min_ambient_temp_c=trigger_cfg.get("min_ambient_temp_c", 40.0),
            min_grid_carbon_gco2=trigger_cfg.get("min_grid_carbon_gco2", 350.0),
            min_water_stress_index=trigger_cfg.get("min_water_stress_index", 3.0),
            require_all=trigger_cfg.get("require_all", True),
        )

        candidates = [
            CandidateRegion(
                zone=c.get("zone", ""),
                name=c.get("name", ""),
                k8s_context=c.get("k8s_context", ""),
            )
            for c in candidates_cfg
        ] if candidates_cfg else None

        return cls(
            trigger_conditions=trigger,
            candidate_regions=candidates,
            deferrable_labels=mig_cfg.get("deferrable_labels"),
            electricity_client=ElectricityMapsClient.from_config(config),
        )

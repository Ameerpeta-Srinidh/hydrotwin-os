"""
HydroTwin OS — Plane 3: Kafka Event Schemas

Pydantic models for all event payloads flowing through the Kafka event mesh.
These schemas enforce structure and enable validation at the message boundary.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def _correlation_id() -> str:
    return str(uuid4())


def _now() -> datetime:
    return datetime.utcnow()


# ─────────────────────── Sensor Events ───────────────────────

class SensorReading(BaseModel):
    """A cleaned sensor reading from the ingestion pipeline."""
    sensor_id: str
    metric_name: str  # e.g., "inlet_temp", "flow_rate", "power_draw"
    value: float
    unit: str
    timestamp: datetime = Field(default_factory=_now)
    facility_id: str = "HydroTwin-DC-01"
    source_plane: int = 2
    correlation_id: str = Field(default_factory=_correlation_id)


# ─────────────────────── RL Action Events ───────────────────────

class RLAction(BaseModel):
    """An action taken by the Nexus RL agent."""
    action_id: str = Field(default_factory=_correlation_id)
    timestamp: datetime = Field(default_factory=_now)

    # Action vector
    cooling_mode_mix: float          # 0 = evaporative, 1 = chiller
    supply_air_temp_setpoint: float  # °C
    fan_speed_pct: float             # 0.2–1.0
    economizer_damper: float         # 0–1

    # State at time of action
    inlet_temp_c: float
    ambient_temp_c: float
    it_load_kw: float
    grid_carbon_intensity: float
    water_stress_index: float

    # Metrics at time of action
    wue: float
    pue: float
    carbon_intensity: float
    thermal_satisfaction: float

    # Reward
    reward: float
    reward_weights: dict[str, float]  # {alpha, beta, gamma, delta}

    # Metadata
    facility_id: str = "HydroTwin-DC-01"
    source_plane: int = 3
    correlation_id: str = Field(default_factory=_correlation_id)
    scenario: str = ""


# ─────────────────────── Forecast Events ───────────────────────

class Forecast(BaseModel):
    """An IT load or weather forecast from the forecasting subsystem."""
    forecast_id: str = Field(default_factory=_correlation_id)
    timestamp: datetime = Field(default_factory=_now)
    forecast_type: str  # "it_load", "ambient_temp", "grid_carbon"
    horizon_hours: int
    values: list[float]
    lower_bound: list[float] = []
    upper_bound: list[float] = []
    model: str  # "prophet", "lstm", "ensemble"
    source_plane: int = 3
    correlation_id: str = Field(default_factory=_correlation_id)


# ─────────────────────── Layout Events ───────────────────────

class LayoutUpdate(BaseModel):
    """A structural layout update from Plane 1 (physics twin)."""
    update_id: str = Field(default_factory=_correlation_id)
    timestamp: datetime = Field(default_factory=_now)
    update_type: str  # "rack_moved", "pipe_added", "cooling_unit_replaced"
    affected_assets: list[str]  # asset IDs
    parameters: dict[str, Any] = {}
    source_plane: int = 1
    correlation_id: str = Field(default_factory=_correlation_id)


# ─────────────────────── Migration Events ───────────────────────

class MigrationEvent(BaseModel):
    """A workload migration event triggered by the Nexus agent."""
    migration_id: str = Field(default_factory=_correlation_id)
    timestamp: datetime = Field(default_factory=_now)

    # Trigger conditions
    ambient_temp_c: float
    grid_carbon_intensity: float
    water_stress_index: float

    # Migration details
    target_region: str
    target_zone: str
    target_carbon_intensity: float
    workloads_migrated: list[str]
    estimated_savings_pct: float

    # Status
    status: str = "dispatched"  # dispatched, completed, failed, rolled_back
    source_plane: int = 3
    correlation_id: str = Field(default_factory=_correlation_id)


# ─────────────────────── Anomaly Events ───────────────────────

class AnomalyAlert(BaseModel):
    """An anomaly alert from Plane 2 (vision/detection)."""
    alert_id: str = Field(default_factory=_correlation_id)
    timestamp: datetime = Field(default_factory=_now)
    anomaly_type: str  # "leak", "hotspot", "vibration", "flow_deviation"
    severity: str  # "info", "warning", "critical"
    location: str  # physical location identifier
    confidence: float  # 0–1
    details: dict[str, Any] = {}
    source_plane: int = 2
    correlation_id: str = Field(default_factory=_correlation_id)


# ─────────────────────── Physics Twin Events ───────────────────────

class PhysicsRecompute(BaseModel):
    """Triggered when the physics twin needs to recompute thermal state."""
    recompute_id: str = Field(default_factory=_correlation_id)
    timestamp: datetime = Field(default_factory=_now)
    trigger_reason: str  # "layout_change", "anomaly_detected", "calibration", "scheduled"
    affected_zones: list[str] = []
    priority: str = "normal"  # "normal", "high", "critical"
    source_plane: int = 1
    correlation_id: str = Field(default_factory=_correlation_id)


class TwinStateSnapshot(BaseModel):
    """A serialized snapshot of the digital twin simulation state."""
    snapshot_id: str = Field(default_factory=_correlation_id)
    timestamp: datetime = Field(default_factory=_now)
    facility_id: str = "HydroTwin-DC-01"

    # Summary metrics
    avg_temp_c: float
    max_temp_c: float
    min_temp_c: float
    total_it_kw: float
    total_cooling_kw: float
    pue: float
    wue: float

    # Alerts
    num_hotspots: int = 0
    hotspot_ids: list[str] = []

    # Metadata
    num_nodes: int = 0
    simulation_mode: str = "steady_state"
    source_plane: int = 1
    correlation_id: str = Field(default_factory=_correlation_id)


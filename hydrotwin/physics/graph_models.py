"""
HydroTwin OS — Plane 1: Asset Graph Data Models

Pydantic models for every physical asset type in the data center.
These form the nodes and edges of the asset graph that the GNN
operates on and the Digital Twin simulates.

Asset Types:
    Rack            — Server rack with IT load and thermal properties
    Pipe            — Cooling water pipe with flow/pressure properties
    Pump            — Water pump with flow rate and power draw
    CRAH            — Computer Room Air Handler (cooling unit)
    Sensor          — Temperature, flow, or pressure sensor
    CoolingTower    — Evaporative cooling tower
    Column          — Structural building column

Edge Types:
    thermal         — Heat transfer path between assets
    hydraulic       — Water/coolant flow connection
    electrical      — Power supply connection
    structural      — Physical structural relationship
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


# ─────────────────────── Enums ───────────────────────

class AssetType(str, Enum):
    RACK = "rack"
    PIPE = "pipe"
    PUMP = "pump"
    CRAH = "crah"
    SENSOR = "sensor"
    COOLING_TOWER = "cooling_tower"
    COLUMN = "column"
    WALL = "wall"
    ZONE = "zone"


class EdgeType(str, Enum):
    THERMAL = "thermal"
    HYDRAULIC = "hydraulic"
    ELECTRICAL = "electrical"
    STRUCTURAL = "structural"
    PROXIMITY = "proximity"


class SensorType(str, Enum):
    TEMPERATURE = "temperature"
    FLOW_RATE = "flow_rate"
    PRESSURE = "pressure"
    HUMIDITY = "humidity"
    POWER = "power"


# ─────────────────────── Position ───────────────────────

class Position3D(BaseModel):
    """3D position in the data center coordinate system (meters)."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def distance_to(self, other: Position3D) -> float:
        return ((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2) ** 0.5


# ─────────────────────── Base Asset Node ───────────────────────

def _asset_id() -> str:
    return str(uuid4())[:8]


class AssetNode(BaseModel):
    """Base class for all physical asset nodes in the data center graph."""
    id: str = Field(default_factory=_asset_id)
    asset_type: AssetType
    name: str = ""
    position: Position3D = Field(default_factory=Position3D)
    zone: str = "default"
    metadata: dict[str, Any] = {}

    # Thermal state
    temperature_c: float = 22.0  # current temperature


# ─────────────────────── Specialized Nodes ───────────────────────

class RackNode(AssetNode):
    """Server rack — heat source in the data center."""
    asset_type: AssetType = AssetType.RACK

    # Capacity
    max_load_kw: float = 20.0          # max IT power draw
    current_load_kw: float = 10.0      # current IT power draw
    num_servers: int = 42              # typical U-space count

    # Thermal
    inlet_temp_c: float = 22.0
    outlet_temp_c: float = 35.0
    delta_t_c: float = 13.0            # typical server ΔT

    # Associated sensors
    inlet_sensor_id: str = ""
    outlet_sensor_id: str = ""


class PipeNode(AssetNode):
    """Cooling water pipe — heat transport path."""
    asset_type: AssetType = AssetType.PIPE

    # Physical
    diameter_mm: float = 150.0          # inner diameter
    length_m: float = 10.0
    material: str = "copper"

    # Flow
    flow_rate_lpm: float = 0.0         # liters per minute
    flow_direction: int = 1             # 1 = forward, -1 = reverse
    max_flow_lpm: float = 500.0

    # Thermal
    thermal_conductivity_w_mk: float = 385.0  # copper default
    inlet_temp_c: float = 7.0
    outlet_temp_c: float = 12.0

    # Pressure
    pressure_drop_kpa: float = 0.0

    # Endpoints
    source_asset_id: str = ""
    target_asset_id: str = ""


class PumpNode(AssetNode):
    """Water pump — drives coolant flow."""
    asset_type: AssetType = AssetType.PUMP

    # Performance
    max_flow_lpm: float = 1000.0
    current_flow_lpm: float = 500.0
    speed_pct: float = 0.5              # 0–1 speed fraction
    power_draw_kw: float = 15.0
    max_power_kw: float = 30.0
    efficiency: float = 0.85

    # State
    is_running: bool = True


class CRAHNode(AssetNode):
    """Computer Room Air Handler — primary cooling unit."""
    asset_type: AssetType = AssetType.CRAH

    # Capacity
    cooling_capacity_kw: float = 200.0
    current_cooling_kw: float = 100.0

    # Air temperatures
    supply_air_temp_c: float = 15.0     # cold air output
    return_air_temp_c: float = 30.0     # hot air intake
    airflow_m3s: float = 10.0           # cubic meters per second

    # Water side
    chilled_water_inlet_c: float = 7.0
    chilled_water_outlet_c: float = 12.0

    # Fan
    fan_speed_pct: float = 0.6
    fan_power_kw: float = 5.0


class SensorNode(AssetNode):
    """Physical sensor measuring temperature, flow, or pressure."""
    asset_type: AssetType = AssetType.SENSOR

    sensor_type: SensorType = SensorType.TEMPERATURE
    current_value: float = 22.0
    unit: str = "celsius"
    calibration_offset: float = 0.0
    accuracy: float = 0.5              # measurement accuracy (±)
    attached_to: str = ""              # asset ID this sensor monitors


class CoolingTowerNode(AssetNode):
    """Evaporative cooling tower — rejects heat to atmosphere using water."""
    asset_type: AssetType = AssetType.COOLING_TOWER

    # Capacity
    max_cooling_kw: float = 500.0
    current_cooling_kw: float = 250.0

    # Water
    water_flow_lpm: float = 200.0
    evaporation_rate_lpm: float = 30.0
    basin_temp_c: float = 30.0

    # Air
    fan_speed_pct: float = 0.7
    approach_temp_c: float = 5.0        # how close to wet-bulb we get


class ColumnNode(AssetNode):
    """Structural building column — constraint for layout optimization."""
    asset_type: AssetType = AssetType.COLUMN

    # Structural
    width_mm: float = 300.0
    depth_mm: float = 300.0
    is_load_bearing: bool = True


class WallNode(AssetNode):
    """Wall segment — boundary for thermal zones."""
    asset_type: AssetType = AssetType.WALL

    length_m: float = 10.0
    height_m: float = 3.0
    thickness_mm: float = 200.0
    insulation_r_value: float = 2.0     # thermal resistance


class ZoneNode(AssetNode):
    """Logical thermal zone grouping multiple assets."""
    asset_type: AssetType = AssetType.ZONE

    area_m2: float = 100.0
    height_m: float = 3.0
    volume_m3: float = 300.0
    target_temp_c: float = 22.0
    asset_ids: list[str] = []           # assets within this zone


# ─────────────────────── Asset Edge ───────────────────────

class AssetEdge(BaseModel):
    """Edge connecting two assets in the data center graph."""
    id: str = Field(default_factory=_asset_id)
    source_id: str
    target_id: str
    edge_type: EdgeType

    # Physical properties
    distance_m: float = 0.0
    thermal_conductivity: float = 1.0   # W/(m·K), effective for this connection
    hydraulic_resistance: float = 0.0   # resistance to flow

    # State
    heat_flux_w: float = 0.0            # current heat transfer rate
    flow_rate_lpm: float = 0.0          # current flow through this edge

    metadata: dict[str, Any] = {}


# ─────────────────────── Type Registry ───────────────────────

ASSET_TYPE_MAP: dict[AssetType, type[AssetNode]] = {
    AssetType.RACK: RackNode,
    AssetType.PIPE: PipeNode,
    AssetType.PUMP: PumpNode,
    AssetType.CRAH: CRAHNode,
    AssetType.SENSOR: SensorNode,
    AssetType.COOLING_TOWER: CoolingTowerNode,
    AssetType.COLUMN: ColumnNode,
    AssetType.WALL: WallNode,
    AssetType.ZONE: ZoneNode,
}

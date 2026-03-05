"""
HydroTwin OS — Plane 1: DXF Floor Plan Ingestion

Parses architectural DXF files to automatically build the data center
asset graph. Extracts walls, rooms, and zones from DXF layers, then
detects rack rows, places sensors, and generates pipe topology.

Supports:
    - DXF layer-based element classification
    - Rectangular entity detection for racks
    - Polyline walls and room boundaries
    - Auto-placement of sensors at rack inlets/outlets
    - Proximity-based pipe topology generation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from hydrotwin.physics.asset_graph import AssetGraph
from hydrotwin.physics.graph_models import (
    AssetEdge, AssetType, CRAHNode, ColumnNode, EdgeType,
    PipeNode, Position3D, PumpNode, RackNode, SensorNode,
    SensorType, WallNode, ZoneNode,
)

logger = logging.getLogger(__name__)


# ─────────────────────── Layer Conventions ───────────────────────

DEFAULT_LAYER_MAP = {
    "RACK": ["RACKS", "IT_RACKS", "SERVER_RACKS", "EQUIPMENT"],
    "WALL": ["WALLS", "A-WALL", "WALL", "PARTITION"],
    "CRAH": ["CRAH", "COOLING", "AHU", "AIR_HANDLER"],
    "COLUMN": ["COLUMN", "COLUMNS", "STRUCTURAL", "S-COLS"],
    "PIPE": ["PIPE", "PIPING", "P-PIPE", "PLUMBING"],
    "ZONE": ["ZONE", "ZONES", "ROOMS", "AREAS"],
}


class DXFIngestion:
    """
    Ingests a DXF architectural drawing and produces an AssetGraph.

    Layer conventions map DXF layers to asset types. The parser extracts
    geometric entities from each layer and converts them to graph nodes.
    """

    def __init__(
        self,
        layer_map: dict[str, list[str]] | None = None,
        rack_width_m: float = 0.6,
        rack_depth_m: float = 1.2,
        proximity_threshold_m: float = 5.0,
    ):
        self.layer_map = layer_map or DEFAULT_LAYER_MAP
        self.rack_width = rack_width_m
        self.rack_depth = rack_depth_m
        self.proximity_threshold = proximity_threshold_m

        # Inverted map for quick lookup
        self._layer_to_type: dict[str, str] = {}
        for asset_key, layer_names in self.layer_map.items():
            for ln in layer_names:
                self._layer_to_type[ln.upper()] = asset_key

    def ingest(self, dxf_path: str | Path, facility_id: str = "HydroTwin-DC-01") -> AssetGraph:
        """
        Parse a DXF file and build an AssetGraph.

        Args:
            dxf_path: Path to the DXF file
            facility_id: Facility identifier for the graph

        Returns:
            Populated AssetGraph
        """
        try:
            import ezdxf
        except ImportError:
            logger.warning("ezdxf not installed. Generating synthetic graph instead.")
            return AssetGraph.create_synthetic(facility_id=facility_id)

        path = Path(dxf_path)
        if not path.exists():
            raise FileNotFoundError(f"DXF file not found: {path}")

        logger.info(f"Ingesting DXF: {path}")
        doc = ezdxf.readfile(str(path))
        msp = doc.modelspace()

        graph = AssetGraph(facility_id=facility_id)

        # ── Phase 1: Extract entities by layer ──
        racks_placed = 0
        walls_placed = 0
        columns_placed = 0

        for entity in msp:
            layer = entity.dxf.layer.upper() if hasattr(entity.dxf, "layer") else ""
            asset_key = self._layer_to_type.get(layer, "")

            if asset_key == "RACK":
                nodes = self._extract_racks(entity)
                for n in nodes:
                    graph.add_node(n)
                    racks_placed += 1

            elif asset_key == "WALL":
                nodes = self._extract_walls(entity)
                for n in nodes:
                    graph.add_node(n)
                    walls_placed += 1

            elif asset_key == "COLUMN":
                nodes = self._extract_columns(entity)
                for n in nodes:
                    graph.add_node(n)
                    columns_placed += 1

            elif asset_key == "CRAH":
                nodes = self._extract_crahs(entity)
                for n in nodes:
                    graph.add_node(n)

            elif asset_key == "PIPE":
                nodes = self._extract_pipes(entity)
                for n in nodes:
                    graph.add_node(n)

        # ── Phase 2: Place sensors at racks ──
        self._place_sensors(graph)

        # ── Phase 3: Auto-assign zones from spatial clustering ──
        self._assign_zones(graph)

        # ── Phase 4: Auto-connect by proximity ──
        graph.auto_connect_by_proximity(
            max_distance_m=self.proximity_threshold,
            edge_type=EdgeType.THERMAL,
        )

        logger.info(
            f"DXF ingestion complete: {graph.num_nodes} nodes, {graph.num_edges} edges | "
            f"Racks: {racks_placed}, Walls: {walls_placed}, Columns: {columns_placed}"
        )

        return graph

    def ingest_synthetic(self, facility_id: str = "HydroTwin-DC-01", **kwargs) -> AssetGraph:
        """Generate a synthetic graph when no DXF file is available."""
        return AssetGraph.create_synthetic(facility_id=facility_id, **kwargs)

    # ──────────────────────────── Entity Extractors ────────────────────────────

    def _extract_racks(self, entity) -> list[RackNode]:
        """Extract rack nodes from DXF INSERT/RECTANGLE entities."""
        racks = []
        try:
            if entity.dxftype() in ("INSERT", "POINT"):
                pos = entity.dxf.insert if hasattr(entity.dxf, "insert") else (0, 0, 0)
                racks.append(RackNode(
                    name=f"Rack-{len(racks)+1}",
                    position=Position3D(x=pos[0] / 1000.0, y=pos[1] / 1000.0, z=0),  # mm → m
                ))
            elif entity.dxftype() == "LWPOLYLINE":
                points = list(entity.get_points())
                if len(points) >= 4:
                    cx = np.mean([p[0] for p in points]) / 1000.0
                    cy = np.mean([p[1] for p in points]) / 1000.0
                    racks.append(RackNode(
                        name=f"Rack-poly",
                        position=Position3D(x=cx, y=cy, z=0),
                    ))
            elif entity.dxftype() == "LINE":
                sx, sy = entity.dxf.start.x / 1000.0, entity.dxf.start.y / 1000.0
                ex, ey = entity.dxf.end.x / 1000.0, entity.dxf.end.y / 1000.0
                cx, cy = (sx + ex) / 2, (sy + ey) / 2
                racks.append(RackNode(
                    name=f"Rack-line",
                    position=Position3D(x=cx, y=cy, z=0),
                ))
        except Exception as e:
            logger.debug(f"Skipping rack entity: {e}")

        return racks

    def _extract_walls(self, entity) -> list[WallNode]:
        """Extract wall nodes from DXF LINE/POLYLINE entities."""
        walls = []
        try:
            if entity.dxftype() == "LINE":
                sx, sy = entity.dxf.start.x / 1000.0, entity.dxf.start.y / 1000.0
                ex, ey = entity.dxf.end.x / 1000.0, entity.dxf.end.y / 1000.0
                length = ((ex - sx)**2 + (ey - sy)**2) ** 0.5
                walls.append(WallNode(
                    name=f"Wall-line",
                    position=Position3D(x=(sx + ex) / 2, y=(sy + ey) / 2, z=0),
                    length_m=length,
                ))
            elif entity.dxftype() == "LWPOLYLINE":
                points = list(entity.get_points())
                for i in range(len(points) - 1):
                    p1, p2 = points[i], points[i + 1]
                    sx, sy = p1[0] / 1000.0, p1[1] / 1000.0
                    ex, ey = p2[0] / 1000.0, p2[1] / 1000.0
                    length = ((ex - sx)**2 + (ey - sy)**2) ** 0.5
                    if length > 0.5:  # filter noise
                        walls.append(WallNode(
                            name=f"Wall-poly-{i}",
                            position=Position3D(x=(sx + ex) / 2, y=(sy + ey) / 2, z=0),
                            length_m=length,
                        ))
        except Exception as e:
            logger.debug(f"Skipping wall entity: {e}")

        return walls

    def _extract_columns(self, entity) -> list[ColumnNode]:
        """Extract column nodes from DXF entities."""
        columns = []
        try:
            if entity.dxftype() in ("INSERT", "POINT", "CIRCLE"):
                if entity.dxftype() == "CIRCLE":
                    pos = (entity.dxf.center.x, entity.dxf.center.y, 0)
                else:
                    pos = entity.dxf.insert if hasattr(entity.dxf, "insert") else (0, 0, 0)
                columns.append(ColumnNode(
                    name=f"Column",
                    position=Position3D(x=pos[0] / 1000.0, y=pos[1] / 1000.0, z=0),
                ))
        except Exception as e:
            logger.debug(f"Skipping column entity: {e}")

        return columns

    def _extract_crahs(self, entity) -> list[CRAHNode]:
        """Extract CRAH nodes from DXF entities."""
        crahs = []
        try:
            if entity.dxftype() in ("INSERT", "POINT"):
                pos = entity.dxf.insert if hasattr(entity.dxf, "insert") else (0, 0, 0)
                crahs.append(CRAHNode(
                    name=f"CRAH",
                    position=Position3D(x=pos[0] / 1000.0, y=pos[1] / 1000.0, z=0),
                ))
        except Exception as e:
            logger.debug(f"Skipping CRAH entity: {e}")

        return crahs

    def _extract_pipes(self, entity) -> list[PipeNode]:
        """Extract pipe nodes from DXF LINE entities."""
        pipes = []
        try:
            if entity.dxftype() == "LINE":
                sx, sy = entity.dxf.start.x / 1000.0, entity.dxf.start.y / 1000.0
                ex, ey = entity.dxf.end.x / 1000.0, entity.dxf.end.y / 1000.0
                length = ((ex - sx)**2 + (ey - sy)**2) ** 0.5
                pipes.append(PipeNode(
                    name=f"Pipe",
                    position=Position3D(x=(sx + ex) / 2, y=(sy + ey) / 2, z=0),
                    length_m=length,
                ))
        except Exception as e:
            logger.debug(f"Skipping pipe entity: {e}")

        return pipes

    # ──────────────────────────── Post-processing ────────────────────────────

    def _place_sensors(self, graph: AssetGraph) -> None:
        """Place inlet/outlet temperature sensors on every rack."""
        racks = graph.nodes_by_type(AssetType.RACK)
        for rack in racks:
            inlet = SensorNode(
                name=f"Sensor-{rack.name}-inlet",
                position=Position3D(
                    x=rack.position.x,
                    y=rack.position.y - 0.3,
                    z=0.5,
                ),
                zone=rack.zone,
                sensor_type=SensorType.TEMPERATURE,
                current_value=getattr(rack, "inlet_temp_c", 22.0),
                attached_to=rack.id,
            )
            outlet = SensorNode(
                name=f"Sensor-{rack.name}-outlet",
                position=Position3D(
                    x=rack.position.x,
                    y=rack.position.y + 0.3,
                    z=2.0,
                ),
                zone=rack.zone,
                sensor_type=SensorType.TEMPERATURE,
                current_value=getattr(rack, "outlet_temp_c", 35.0),
                attached_to=rack.id,
            )
            graph.add_node(inlet)
            graph.add_node(outlet)
            graph.update_node(rack.id, inlet_sensor_id=inlet.id, outlet_sensor_id=outlet.id)

    def _assign_zones(self, graph: AssetGraph, zone_size_m: float = 10.0) -> None:
        """
        Assign zones to nodes based on spatial grid.
        Divides the floor into grid cells and assigns zone names.
        """
        for node in graph.nodes.values():
            zx = int(node.position.x // zone_size_m)
            zy = int(node.position.y // zone_size_m)
            zone_name = f"zone-{zx}-{zy}"
            graph.update_node(node.id, zone=zone_name)

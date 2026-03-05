"""
HydroTwin OS — Plane 1: Asset Graph

In-memory graph structure representing the physical data center topology.
Nodes are physical assets (racks, pipes, pumps, CRAHs, sensors, columns).
Edges are physical connections (thermal, hydraulic, electrical, structural).

The graph supports:
    - Node/edge CRUD operations
    - Adjacency and connectivity queries
    - Subgraph extraction by zone or asset type
    - Serialization to JSON and Neo4j-compatible formats
    - Factory methods for building from floor plans or synthetic layouts
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from typing import Any, Iterator

from hydrotwin.physics.graph_models import (
    AssetNode, AssetEdge, AssetType, EdgeType, Position3D,
    RackNode, PipeNode, PumpNode, CRAHNode, SensorNode,
    CoolingTowerNode, ColumnNode, WallNode, ZoneNode,
    ASSET_TYPE_MAP,
)

logger = logging.getLogger(__name__)


class AssetGraph:
    """
    In-memory graph of data center physical assets.

    Undirected for thermal connections, directed for hydraulic flow.
    Supports efficient adjacency queries and subgraph extraction.
    """

    def __init__(self, facility_id: str = "HydroTwin-DC-01"):
        self.facility_id = facility_id
        self._nodes: dict[str, AssetNode] = {}
        self._edges: dict[str, AssetEdge] = {}
        self._adjacency: dict[str, list[str]] = defaultdict(list)  # node_id → [edge_ids]
        self._type_index: dict[AssetType, set[str]] = defaultdict(set)  # type → {node_ids}
        self._zone_index: dict[str, set[str]] = defaultdict(set)  # zone → {node_ids}

    # ──────────────────────────── Node Operations ────────────────────────────

    def add_node(self, node: AssetNode) -> str:
        """Add an asset node. Returns the node ID."""
        if node.id in self._nodes:
            raise ValueError(f"Node {node.id} already exists")

        self._nodes[node.id] = node
        self._type_index[node.asset_type].add(node.id)
        self._zone_index[node.zone].add(node.id)
        return node.id

    def remove_node(self, node_id: str) -> AssetNode | None:
        """Remove a node and all its connected edges."""
        node = self._nodes.pop(node_id, None)
        if node is None:
            return None

        self._type_index[node.asset_type].discard(node_id)
        self._zone_index[node.zone].discard(node_id)

        # Remove connected edges
        edge_ids = list(self._adjacency.get(node_id, []))
        for edge_id in edge_ids:
            self.remove_edge(edge_id)

        del self._adjacency[node_id]
        return node

    def get_node(self, node_id: str) -> AssetNode | None:
        return self._nodes.get(node_id)

    def update_node(self, node_id: str, **updates) -> bool:
        """Update specific fields of a node."""
        node = self._nodes.get(node_id)
        if node is None:
            return False

        for key, val in updates.items():
            if hasattr(node, key):
                setattr(node, key, val)
        return True

    @property
    def nodes(self) -> dict[str, AssetNode]:
        return self._nodes

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    # ──────────────────────────── Edge Operations ────────────────────────────

    def add_edge(self, edge: AssetEdge) -> str:
        """Add an edge between two existing nodes. Returns edge ID."""
        if edge.source_id not in self._nodes:
            raise ValueError(f"Source node {edge.source_id} not found")
        if edge.target_id not in self._nodes:
            raise ValueError(f"Target node {edge.target_id} not found")

        self._edges[edge.id] = edge
        self._adjacency[edge.source_id].append(edge.id)
        self._adjacency[edge.target_id].append(edge.id)
        return edge.id

    def remove_edge(self, edge_id: str) -> AssetEdge | None:
        """Remove an edge."""
        edge = self._edges.pop(edge_id, None)
        if edge is None:
            return None

        if edge_id in self._adjacency.get(edge.source_id, []):
            self._adjacency[edge.source_id].remove(edge_id)
        if edge_id in self._adjacency.get(edge.target_id, []):
            self._adjacency[edge.target_id].remove(edge_id)
        return edge

    def get_edge(self, edge_id: str) -> AssetEdge | None:
        return self._edges.get(edge_id)

    @property
    def edges(self) -> dict[str, AssetEdge]:
        return self._edges

    @property
    def num_edges(self) -> int:
        return len(self._edges)

    # ──────────────────────────── Queries ────────────────────────────

    def neighbors(self, node_id: str) -> list[AssetNode]:
        """Get all neighbor nodes connected to the given node."""
        result = []
        for edge_id in self._adjacency.get(node_id, []):
            edge = self._edges.get(edge_id)
            if edge is None:
                continue
            other_id = edge.target_id if edge.source_id == node_id else edge.source_id
            other = self._nodes.get(other_id)
            if other:
                result.append(other)
        return result

    def edges_of(self, node_id: str) -> list[AssetEdge]:
        """Get all edges connected to a node."""
        return [
            self._edges[eid]
            for eid in self._adjacency.get(node_id, [])
            if eid in self._edges
        ]

    def nodes_by_type(self, asset_type: AssetType) -> list[AssetNode]:
        """Get all nodes of a specific asset type."""
        return [self._nodes[nid] for nid in self._type_index.get(asset_type, set()) if nid in self._nodes]

    def nodes_in_zone(self, zone: str) -> list[AssetNode]:
        """Get all nodes in a specific zone."""
        return [self._nodes[nid] for nid in self._zone_index.get(zone, set()) if nid in self._nodes]

    def edges_by_type(self, edge_type: EdgeType) -> list[AssetEdge]:
        """Get all edges of a specific type."""
        return [e for e in self._edges.values() if e.edge_type == edge_type]

    def connected_components(self) -> list[set[str]]:
        """Find connected components in the graph using BFS."""
        visited: set[str] = set()
        components: list[set[str]] = []

        for node_id in self._nodes:
            if node_id in visited:
                continue
            component: set[str] = set()
            queue = [node_id]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                for neighbor in self.neighbors(current):
                    if neighbor.id not in visited:
                        queue.append(neighbor.id)
            components.append(component)

        return components

    def shortest_path(self, source_id: str, target_id: str) -> list[str] | None:
        """BFS shortest path by node count."""
        if source_id not in self._nodes or target_id not in self._nodes:
            return None

        visited: set[str] = {source_id}
        queue: list[list[str]] = [[source_id]]

        while queue:
            path = queue.pop(0)
            current = path[-1]

            if current == target_id:
                return path

            for neighbor in self.neighbors(current):
                if neighbor.id not in visited:
                    visited.add(neighbor.id)
                    queue.append(path + [neighbor.id])

        return None  # no path found

    def subgraph(self, node_ids: set[str]) -> AssetGraph:
        """Extract a subgraph containing only the specified nodes and their connecting edges."""
        sub = AssetGraph(facility_id=self.facility_id)
        for nid in node_ids:
            node = self._nodes.get(nid)
            if node:
                sub.add_node(node.model_copy())

        for edge in self._edges.values():
            if edge.source_id in node_ids and edge.target_id in node_ids:
                sub.add_edge(edge.model_copy())

        return sub

    # ──────────────────────────── Auto-Connect ────────────────────────────

    def auto_connect_by_proximity(
        self,
        max_distance_m: float = 5.0,
        edge_type: EdgeType = EdgeType.THERMAL,
        thermal_conductivity: float = 1.0,
    ) -> int:
        """
        Automatically create edges between nodes within a maximum distance.
        Returns the number of edges created.
        """
        node_list = list(self._nodes.values())
        count = 0

        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                a, b = node_list[i], node_list[j]
                dist = a.position.distance_to(b.position)

                if dist <= max_distance_m and dist > 0:
                    edge = AssetEdge(
                        source_id=a.id,
                        target_id=b.id,
                        edge_type=edge_type,
                        distance_m=dist,
                        thermal_conductivity=thermal_conductivity / max(dist, 0.1),
                    )
                    self.add_edge(edge)
                    count += 1

        logger.info(f"Auto-connected {count} edges within {max_distance_m}m")
        return count

    # ──────────────────────────── Serialization ────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize the graph to a JSON-compatible dictionary."""
        return {
            "facility_id": self.facility_id,
            "nodes": {nid: node.model_dump() for nid, node in self._nodes.items()},
            "edges": {eid: edge.model_dump() for eid, edge in self._edges.items()},
            "stats": {
                "num_nodes": self.num_nodes,
                "num_edges": self.num_edges,
                "node_types": {t.value: len(ids) for t, ids in self._type_index.items() if ids},
                "zones": list(self._zone_index.keys()),
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_neo4j_statements(self) -> list[str]:
        """Generate Cypher CREATE statements for Neo4j import."""
        statements = []

        for node in self._nodes.values():
            label = node.asset_type.value.upper()
            props = json.dumps({k: v for k, v in node.model_dump().items()
                               if k not in ("asset_type", "metadata", "position")}, default=str)
            statements.append(
                f"CREATE (n:{label} {{id: '{node.id}', name: '{node.name}', "
                f"x: {node.position.x}, y: {node.position.y}, z: {node.position.z}, "
                f"zone: '{node.zone}', temperature_c: {node.temperature_c}}})"
            )

        for edge in self._edges.values():
            rel_type = edge.edge_type.value.upper()
            statements.append(
                f"MATCH (a {{id: '{edge.source_id}'}}), (b {{id: '{edge.target_id}'}}) "
                f"CREATE (a)-[:{rel_type} {{distance_m: {edge.distance_m}, "
                f"thermal_conductivity: {edge.thermal_conductivity}}}]->(b)"
            )

        return statements

    # ──────────────────────────── Factory Methods ────────────────────────────

    @classmethod
    def create_synthetic(
        cls,
        num_racks: int = 40,
        num_crahs: int = 4,
        rows: int = 4,
        facility_id: str = "HydroTwin-DC-01",
    ) -> AssetGraph:
        """
        Create a synthetic data center layout for testing and training.

        Layout: hot-aisle/cold-aisle arrangement with CRAHs at the perimeter.
        """
        graph = cls(facility_id=facility_id)

        rack_spacing_m = 2.0
        row_spacing_m = 4.0
        racks_per_row = num_racks // rows

        # ── Place racks ──
        rack_ids: list[str] = []
        for row in range(rows):
            for col in range(racks_per_row):
                rack = RackNode(
                    name=f"Rack-R{row+1}C{col+1}",
                    position=Position3D(
                        x=col * rack_spacing_m,
                        y=row * row_spacing_m,
                        z=0.0,
                    ),
                    zone=f"zone-{row // 2 + 1}",
                    current_load_kw=8.0 + (row * col % 7) * 1.5,
                    inlet_temp_c=20.0 + row * 0.5,
                )
                graph.add_node(rack)
                rack_ids.append(rack.id)

        # ── Place CRAHs at perimeter ──
        floor_width = racks_per_row * rack_spacing_m
        floor_depth = rows * row_spacing_m
        crah_ids: list[str] = []
        crah_positions = [
            Position3D(x=-3, y=floor_depth * 0.25, z=0),
            Position3D(x=-3, y=floor_depth * 0.75, z=0),
            Position3D(x=floor_width + 3, y=floor_depth * 0.25, z=0),
            Position3D(x=floor_width + 3, y=floor_depth * 0.75, z=0),
        ]
        for i in range(min(num_crahs, len(crah_positions))):
            crah = CRAHNode(
                name=f"CRAH-{i+1}",
                position=crah_positions[i],
                zone=f"zone-{i // 2 + 1}",
                cooling_capacity_kw=200.0,
                current_cooling_kw=120.0,
            )
            graph.add_node(crah)
            crah_ids.append(crah.id)

        # ── Place pumps ──
        pump_ids: list[str] = []
        for i in range(2):
            pump = PumpNode(
                name=f"Pump-{i+1}",
                position=Position3D(x=-5, y=floor_depth * (0.3 + i * 0.4), z=0),
                zone="mechanical",
                current_flow_lpm=600.0,
            )
            graph.add_node(pump)
            pump_ids.append(pump.id)

        # ── Place cooling tower ──
        tower = CoolingTowerNode(
            name="CoolingTower-1",
            position=Position3D(x=-8, y=floor_depth * 0.5, z=0),
            zone="exterior",
        )
        graph.add_node(tower)

        # ── Place sensors on racks ──
        for rid in rack_ids:
            rack = graph.get_node(rid)
            if rack and isinstance(rack, RackNode):
                inlet_sensor = SensorNode(
                    name=f"Sensor-{rack.name}-inlet",
                    position=Position3D(x=rack.position.x, y=rack.position.y - 0.3, z=0.5),
                    zone=rack.zone,
                    current_value=rack.inlet_temp_c,
                    attached_to=rid,
                )
                outlet_sensor = SensorNode(
                    name=f"Sensor-{rack.name}-outlet",
                    position=Position3D(x=rack.position.x, y=rack.position.y + 0.3, z=2.0),
                    zone=rack.zone,
                    current_value=rack.outlet_temp_c,
                    sensor_type="temperature",
                    attached_to=rid,
                )
                graph.add_node(inlet_sensor)
                graph.add_node(outlet_sensor)
                graph.update_node(rid, inlet_sensor_id=inlet_sensor.id, outlet_sensor_id=outlet_sensor.id)

        # ── Auto-connect by proximity ──
        graph.auto_connect_by_proximity(max_distance_m=5.0, edge_type=EdgeType.THERMAL)

        # ── Add hydraulic edges: pump → CRAH ──
        for pid in pump_ids:
            for cid in crah_ids:
                pump = graph.get_node(pid)
                crah = graph.get_node(cid)
                if pump and crah:
                    dist = pump.position.distance_to(crah.position)
                    graph.add_edge(AssetEdge(
                        source_id=pid, target_id=cid,
                        edge_type=EdgeType.HYDRAULIC,
                        distance_m=dist,
                        flow_rate_lpm=300.0,
                    ))

        # ── Add hydraulic edges: CRAH → tower ──
        for cid in crah_ids:
            crah = graph.get_node(cid)
            if crah:
                dist = crah.position.distance_to(tower.position)
                graph.add_edge(AssetEdge(
                    source_id=cid, target_id=tower.id,
                    edge_type=EdgeType.HYDRAULIC,
                    distance_m=dist,
                    flow_rate_lpm=200.0,
                ))

        logger.info(
            f"Synthetic graph created: {graph.num_nodes} nodes, "
            f"{graph.num_edges} edges, {len(rack_ids)} racks, "
            f"{len(crah_ids)} CRAHs"
        )
        return graph

    def summary(self) -> dict[str, Any]:
        """Return a summary of the graph contents."""
        return {
            "facility_id": self.facility_id,
            "total_nodes": self.num_nodes,
            "total_edges": self.num_edges,
            "by_type": {t.value: len(ids) for t, ids in self._type_index.items() if ids},
            "zones": {z: len(ids) for z, ids in self._zone_index.items() if ids},
            "connected_components": len(self.connected_components()),
        }

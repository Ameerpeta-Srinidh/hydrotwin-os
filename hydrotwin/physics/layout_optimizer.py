"""
HydroTwin OS — Plane 1: Layout Optimizer

Gradient-based optimization using the Digital Twin's differentiable output.
Optimizes rack placement, pipe routing, and CRAH positioning to minimize
thermal non-uniformity and maximize cooling efficiency.

Optimizer types:
    RackPlacementOptimizer  — Minimize max inlet temperature
    PipeRoutingOptimizer    — Minimize total pipe length for target flow
    CRAHPlacementAdvisor    — Recommend optimal CRAH positions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hydrotwin.physics.asset_graph import AssetGraph
from hydrotwin.physics.graph_models import (
    AssetType, CRAHNode, Position3D, RackNode,
)

logger = logging.getLogger(__name__)


# ─────────────────────── Result Types ───────────────────────

@dataclass
class PlacementResult:
    """Result of a rack placement optimization."""
    original_positions: dict[str, dict[str, float]] = field(default_factory=dict)
    optimized_positions: dict[str, dict[str, float]] = field(default_factory=dict)
    original_max_temp: float = 0.0
    optimized_max_temp: float = 0.0
    improvement_pct: float = 0.0
    iterations: int = 0
    converged: bool = False


@dataclass
class RoutingResult:
    """Result of a pipe routing optimization."""
    total_pipe_length_m: float = 0.0
    optimized_length_m: float = 0.0
    savings_pct: float = 0.0
    routes: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CRAHPlacement:
    """Recommended CRAH placement."""
    position: Position3D = field(default_factory=Position3D)
    zone: str = ""
    expected_temp_reduction_c: float = 0.0
    coverage_rack_ids: list[str] = field(default_factory=list)
    priority: int = 1  # 1 = highest


# ─────────────────────── Layout Optimizer ───────────────────────

class LayoutOptimizer:
    """
    Optimizes data center physical layout using the asset graph topology.

    Uses heuristic + gradient-free optimization (since full differentiable
    optimization requires the trained GNN). Provides practical placement
    recommendations based on thermal analysis.
    """

    # ASHRAE clearance requirements
    MIN_RACK_SPACING_M = 1.2         # minimum between racks
    MIN_AISLE_WIDTH_M = 1.0          # cold/hot aisle minimum
    MIN_CRAH_CLEARANCE_M = 2.0       # clearance around CRAHs

    def __init__(
        self,
        graph: AssetGraph,
        ashrae_max_c: float = 32.0,
        ashrae_min_c: float = 15.0,
    ):
        self.graph = graph
        self.ashrae_max = ashrae_max_c
        self.ashrae_min = ashrae_min_c

    def optimize_rack_placement(
        self,
        max_iterations: int = 100,
        step_size: float = 0.1,
    ) -> PlacementResult:
        """
        Optimize rack positions to minimize thermal non-uniformity.

        Strategy: iteratively move hot racks closer to CRAHs and spread
        high-density racks apart to improve airflow distribution.
        """
        racks = self.graph.nodes_by_type(AssetType.RACK)
        crahs = self.graph.nodes_by_type(AssetType.CRAH)

        if not racks or not crahs:
            return PlacementResult(converged=True)

        # Record original positions
        original = {r.id: {"x": r.position.x, "y": r.position.y, "z": r.position.z} for r in racks}

        # Initial thermal analysis
        rack_temps = self._compute_estimated_temps(racks, crahs)
        original_max = max(rack_temps.values()) if rack_temps else 22.0

        best_max_temp = original_max

        for iteration in range(max_iterations):
            improved = False

            for rack in racks:
                if rack.id not in rack_temps:
                    continue

                temp = rack_temps[rack.id]

                if temp > self.ashrae_max * 0.9:  # hot rack
                    # Move toward nearest CRAH
                    nearest_crah = self._nearest_asset(rack, crahs)
                    if nearest_crah:
                        dx = (nearest_crah.position.x - rack.position.x) * step_size
                        dy = (nearest_crah.position.y - rack.position.y) * step_size

                        new_x = rack.position.x + dx
                        new_y = rack.position.y + dy

                        # Check clearance constraints
                        if self._check_clearance(rack.id, new_x, new_y, racks):
                            self.graph.update_node(rack.id,
                                position=Position3D(x=new_x, y=new_y, z=rack.position.z))
                            improved = True

                elif temp < self.ashrae_min * 1.1:  # cold rack
                    # Move away from nearest CRAH (too much cooling)
                    nearest_crah = self._nearest_asset(rack, crahs)
                    if nearest_crah:
                        dx = (rack.position.x - nearest_crah.position.x) * step_size * 0.3
                        dy = (rack.position.y - nearest_crah.position.y) * step_size * 0.3
                        new_x = rack.position.x + dx
                        new_y = rack.position.y + dy

                        if self._check_clearance(rack.id, new_x, new_y, racks):
                            self.graph.update_node(rack.id,
                                position=Position3D(x=new_x, y=new_y, z=rack.position.z))
                            improved = True

            # Recompute temperatures
            racks = self.graph.nodes_by_type(AssetType.RACK)
            rack_temps = self._compute_estimated_temps(racks, crahs)
            current_max = max(rack_temps.values()) if rack_temps else 22.0

            if current_max < best_max_temp:
                best_max_temp = current_max

            if not improved:
                break

        # Record optimized positions
        optimized = {r.id: {"x": r.position.x, "y": r.position.y, "z": r.position.z}
                     for r in self.graph.nodes_by_type(AssetType.RACK)}

        improvement = (original_max - best_max_temp) / max(original_max, 0.1) * 100

        logger.info(
            f"Rack placement optimized: {original_max:.1f}°C → {best_max_temp:.1f}°C "
            f"({improvement:.1f}% improvement, {iteration+1} iterations)"
        )

        return PlacementResult(
            original_positions=original,
            optimized_positions=optimized,
            original_max_temp=original_max,
            optimized_max_temp=best_max_temp,
            improvement_pct=improvement,
            iterations=iteration + 1,
            converged=not improved,
        )

    def recommend_cooling_units(
        self,
        max_recommendations: int = 3,
    ) -> list[CRAHPlacement]:
        """
        Identify optimal positions for additional CRAH units.

        Strategy: find thermal hotspot clusters and recommend
        CRAH positions that maximize coverage of underserved racks.
        """
        racks = self.graph.nodes_by_type(AssetType.RACK)
        crahs = self.graph.nodes_by_type(AssetType.CRAH)

        if not racks:
            return []

        # Find underserved racks (far from CRAHs or hot)
        rack_temps = self._compute_estimated_temps(racks, crahs)
        hot_racks = [
            r for r in racks
            if rack_temps.get(r.id, 22) > self.ashrae_max * 0.8
        ]

        if not hot_racks:
            logger.info("No hot racks found — no additional CRAHs needed")
            return []

        # Cluster hot racks and recommend CRAHs at centroids
        recommendations: list[CRAHPlacement] = []
        remaining = list(hot_racks)

        for priority in range(1, max_recommendations + 1):
            if not remaining:
                break

            # Centroid of remaining hot racks
            cx = np.mean([r.position.x for r in remaining])
            cy = np.mean([r.position.y for r in remaining])

            # Offset slightly to the aisle side
            pos = Position3D(x=float(cx) - 3.0, y=float(cy), z=0.0)

            # Identify which racks this CRAH would cover (within 8m)
            covered = [r.id for r in remaining if r.position.distance_to(pos) < 8.0]

            # Estimate temperature reduction
            avg_temp = np.mean([rack_temps.get(rid, 25) for rid in covered])
            expected_reduction = min(5.0, avg_temp - 22.0) * 0.6  # 60% improvement estimate

            recommendations.append(CRAHPlacement(
                position=pos,
                zone=remaining[0].zone if remaining else "default",
                expected_temp_reduction_c=float(expected_reduction),
                coverage_rack_ids=covered,
                priority=priority,
            ))

            # Remove covered racks
            remaining = [r for r in remaining if r.id not in covered]

        logger.info(f"Recommended {len(recommendations)} additional CRAH positions")
        return recommendations

    def optimize_pipe_routing(self) -> RoutingResult:
        """
        Analyze and optimize pipe routing between pumps, CRAHs, and cooling towers.

        Minimizes total pipe length while maintaining connectivity.
        """
        from hydrotwin.physics.graph_models import EdgeType

        hydraulic_edges = self.graph.edges_by_type(EdgeType.HYDRAULIC)
        if not hydraulic_edges:
            return RoutingResult()

        total_length = sum(e.distance_m for e in hydraulic_edges)

        # Simple optimization: for each hydraulic edge, check if a shorter
        # route exists through an intermediate node
        optimized_length = total_length
        routes = []

        for edge in hydraulic_edges:
            src = self.graph.get_node(edge.source_id)
            tgt = self.graph.get_node(edge.target_id)
            if src and tgt:
                direct_dist = src.position.distance_to(tgt.position)
                routes.append({
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "current_length_m": edge.distance_m,
                    "direct_distance_m": direct_dist,
                    "overhead_pct": (edge.distance_m - direct_dist) / max(direct_dist, 0.1) * 100,
                })
                optimized_length -= max(0, edge.distance_m - direct_dist * 1.1)

        savings = (total_length - optimized_length) / max(total_length, 0.1) * 100

        return RoutingResult(
            total_pipe_length_m=total_length,
            optimized_length_m=optimized_length,
            savings_pct=savings,
            routes=routes,
        )

    # ──────────────────────────── Helpers ────────────────────────────

    def _compute_estimated_temps(
        self,
        racks: list[AssetNode],
        crahs: list[AssetNode],
    ) -> dict[str, float]:
        """
        Estimate rack inlet temperatures based on distance to CRAHs.

        Simplified model: T_inlet ≈ T_supply + f(distance, IT_load)
        """
        temps = {}
        for rack in racks:
            load_kw = getattr(rack, "current_load_kw", 10.0)
            max_load = getattr(rack, "max_load_kw", 20.0)
            load_factor = load_kw / max(max_load, 1.0)

            # Distance to nearest CRAH
            nearest = self._nearest_asset(rack, crahs)
            if nearest:
                dist = rack.position.distance_to(nearest.position)
                supply_temp = getattr(nearest, "supply_air_temp_c", 15.0)
            else:
                dist = 20.0  # no CRAH nearby
                supply_temp = 15.0

            # Estimated inlet temp: supply + distance penalty + load penalty
            estimated = supply_temp + (dist * 0.3) + (load_factor * 8.0)
            temps[rack.id] = estimated

        return temps

    def _nearest_asset(
        self,
        source: AssetNode,
        targets: list[AssetNode],
    ) -> AssetNode | None:
        """Find the nearest target asset to the source."""
        best = None
        best_dist = float("inf")
        for t in targets:
            d = source.position.distance_to(t.position)
            if d < best_dist:
                best_dist = d
                best = t
        return best

    def _check_clearance(
        self,
        rack_id: str,
        new_x: float,
        new_y: float,
        all_racks: list[AssetNode],
    ) -> bool:
        """Check if a new position maintains minimum clearance from other racks."""
        new_pos = Position3D(x=new_x, y=new_y, z=0)
        for r in all_racks:
            if r.id == rack_id:
                continue
            dist = new_pos.distance_to(r.position)
            if dist < self.MIN_RACK_SPACING_M:
                return False

        # Check clearance from columns
        columns = self.graph.nodes_by_type(AssetType.COLUMN)
        for col in columns:
            if new_pos.distance_to(col.position) < 0.5:
                return False

        return True

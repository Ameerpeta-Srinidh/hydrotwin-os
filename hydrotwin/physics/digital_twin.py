"""
HydroTwin OS — Plane 1: Digital Twin Engine

The unified simulation engine that combines the GNN thermal model with
PINN physics constraints to create a differentiable digital twin of the
data center. Provides interfaces for:

    simulate()      — Forward pass: given layout + conditions → thermal state
    what_if()       — Hypothetical scenario evaluation
    calibrate()     — Update model to match real sensor data
    get_metrics()   — Compute WUE, PUE, cooling efficiency
    train()         — Train the GNN+PINN on synthetic or real data
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from hydrotwin.physics.asset_graph import AssetGraph
from hydrotwin.physics.graph_models import (
    AssetEdge, AssetNode, AssetType, CRAHNode, RackNode,
)
from hydrotwin.physics.thermal_gnn import ThermalGNN, graph_to_tensors
from hydrotwin.physics.physics_loss import PhysicsLoss, PhysicsLossWeights

logger = logging.getLogger(__name__)


# ─────────────────────── Twin State ───────────────────────

@dataclass
class TwinState:
    """Snapshot of the digital twin simulation results."""

    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Per-node results
    node_temperatures: dict[str, float] = field(default_factory=dict)
    node_ids: list[str] = field(default_factory=list)

    # Facility-level metrics
    total_it_load_kw: float = 0.0
    total_cooling_kw: float = 0.0
    avg_inlet_temp_c: float = 0.0
    max_inlet_temp_c: float = 0.0
    min_inlet_temp_c: float = 0.0
    wue: float = 0.0
    pue: float = 0.0

    # Alerts
    hotspots: list[str] = field(default_factory=list)
    cold_spots: list[str] = field(default_factory=list)

    # Metadata
    simulation_mode: str = "steady_state"  # or "transient"
    converged: bool = True

    def summary(self) -> dict[str, Any]:
        return {
            "avg_temp": round(self.avg_inlet_temp_c, 1),
            "max_temp": round(self.max_inlet_temp_c, 1),
            "min_temp": round(self.min_inlet_temp_c, 1),
            "it_load_kw": round(self.total_it_load_kw, 1),
            "cooling_kw": round(self.total_cooling_kw, 1),
            "wue": round(self.wue, 3),
            "pue": round(self.pue, 3),
            "hotspots": len(self.hotspots),
            "num_nodes": len(self.node_temperatures),
        }


# ─────────────────────── Digital Twin Engine ───────────────────────

class DigitalTwin:
    """
    Differentiable Digital Twin of a data center.

    Combines a Graph Neural Network for thermal simulation with
    Physics-Informed loss functions for physically grounded predictions.
    """

    def __init__(
        self,
        graph: AssetGraph | None = None,
        hidden_dim: int = 64,
        num_gnn_layers: int = 3,
        heads: int = 4,
        pinn_weights: PhysicsLossWeights | None = None,
        device: str = "auto",
        ashrae_max_c: float = 32.0,
        ashrae_min_c: float = 15.0,
    ):
        self.graph = graph or AssetGraph.create_synthetic()
        self.ashrae_max = ashrae_max_c
        self.ashrae_min = ashrae_min_c

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Build GNN
        self.gnn = ThermalGNN(
            node_feature_dim=13,
            edge_feature_dim=7,
            hidden_dim=hidden_dim,
            num_layers=num_gnn_layers,
            heads=heads,
        ).to(self.device)

        # Physics loss
        self.physics_loss = PhysicsLoss(pinn_weights)

        # Internal state
        self._last_state: TwinState | None = None
        self._trained = False

        logger.info(
            f"DigitalTwin initialized | "
            f"Graph: {self.graph.num_nodes} nodes, {self.graph.num_edges} edges | "
            f"Device: {self.device}"
        )

    # ──────────────────────────── Simulation ────────────────────────────

    def simulate(
        self,
        boundary_conditions: dict[str, Any] | None = None,
    ) -> TwinState:
        """
        Run a forward simulation on the current graph state.

        Args:
            boundary_conditions: Optional overrides for ambient temp, IT loads, etc.

        Returns:
            TwinState with temperatures, metrics, and hotspot alerts.
        """
        # Apply boundary conditions
        if boundary_conditions:
            self._apply_boundary_conditions(boundary_conditions)

        # Convert graph to tensors
        node_list = list(self.graph.nodes.values())
        edge_list = list(self.graph.edges.values())
        tensors = graph_to_tensors(node_list, edge_list)

        x = tensors["x"].to(self.device)
        edge_index = tensors["edge_index"].to(self.device)
        edge_attr = tensors["edge_attr"].to(self.device)

        # Forward pass
        self.gnn.eval()
        with torch.no_grad():
            output = self.gnn(x, edge_index, edge_attr)

        predicted_temps = output["node_temps"].cpu().numpy()
        global_metrics = output["global_metrics"].cpu().numpy()

        # Build TwinState
        state = self._build_state(node_list, predicted_temps, global_metrics)
        self._last_state = state

        return state

    def what_if(
        self,
        modification: dict[str, Any],
    ) -> TwinState:
        """
        Evaluate a hypothetical scenario without permanently modifying the graph.

        Args:
            modification: Dict describing the change, e.g.:
                {"type": "remove_node", "node_id": "pump-1"}
                {"type": "update_node", "node_id": "rack-5", "current_load_kw": 20.0}
                {"type": "add_crah", "position": {"x": 10, "y": 5, "z": 0}}

        Returns:
            TwinState for the hypothetical scenario.
        """
        import copy

        # Deep copy the graph
        original_dict = self.graph.to_dict()
        temp_graph = AssetGraph(facility_id=self.graph.facility_id)

        # Rebuild from dict (simplified — just copy nodes and edges)
        for nid, node_data in original_dict["nodes"].items():
            node = self._reconstruct_node(node_data)
            if node:
                temp_graph.add_node(node)

        for eid, edge_data in original_dict["edges"].items():
            try:
                edge = AssetEdge(**edge_data)
                temp_graph.add_edge(edge)
            except Exception:
                pass

        # Apply modification to temp graph
        mod_type = modification.get("type", "")

        if mod_type == "remove_node":
            node_id = modification.get("node_id", "")
            temp_graph.remove_node(node_id)
            logger.info(f"What-if: removed node {node_id}")

        elif mod_type == "update_node":
            node_id = modification.get("node_id", "")
            updates = {k: v for k, v in modification.items() if k not in ("type", "node_id")}
            temp_graph.update_node(node_id, **updates)
            logger.info(f"What-if: updated node {node_id} with {updates}")

        elif mod_type == "add_crah":
            from hydrotwin.physics.graph_models import CRAHNode, Position3D
            pos = modification.get("position", {})
            crah = CRAHNode(
                name=f"WhatIf-CRAH",
                position=Position3D(**pos),
                cooling_capacity_kw=modification.get("cooling_capacity_kw", 200.0),
            )
            temp_graph.add_node(crah)
            temp_graph.auto_connect_by_proximity(max_distance_m=8.0)
            logger.info(f"What-if: added CRAH at {pos}")

        # Save original graph, simulate with temp, restore
        original_graph = self.graph
        self.graph = temp_graph

        try:
            state = self.simulate()
            state.simulation_mode = f"what_if:{mod_type}"
        finally:
            self.graph = original_graph

        return state

    def calibrate(self, sensor_readings: dict[str, float]) -> int:
        """
        Update the graph's node temperatures to match real sensor readings.

        Args:
            sensor_readings: Dict of {sensor_id: measured_value}

        Returns:
            Number of nodes calibrated.
        """
        calibrated = 0
        sensors = self.graph.nodes_by_type(AssetType.SENSOR)

        for sensor in sensors:
            if sensor.id in sensor_readings:
                # Update sensor's value
                new_val = sensor_readings[sensor.id]
                self.graph.update_node(sensor.id, current_value=new_val, temperature_c=new_val)

                # Also update the asset it's attached to
                if hasattr(sensor, "attached_to") and sensor.attached_to:
                    self.graph.update_node(sensor.attached_to, temperature_c=new_val)

                calibrated += 1

        if calibrated > 0:
            logger.info(f"Calibrated {calibrated} nodes from sensor readings")

        return calibrated

    # ──────────────────────────── Training ────────────────────────────

    def train(
        self,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        num_synthetic_samples: int = 50,
    ) -> dict[str, list[float]]:
        """
        Train the GNN+PINN on synthetic thermal data.

        Generates synthetic training data by perturbing the graph state
        and computing analytical temperature solutions as targets.
        """
        logger.info(f"Starting training: {epochs} epochs, {num_synthetic_samples} samples")

        optimizer = optim.Adam(self.gnn.parameters(), lr=learning_rate)
        history: dict[str, list[float]] = {"total": [], "data": [], "energy": [], "fourier": []}

        self.gnn.train()

        for epoch in range(epochs):
            epoch_losses: dict[str, float] = {"total": 0, "data": 0, "energy": 0, "fourier": 0}

            for _ in range(num_synthetic_samples):
                # Generate a synthetic sample by perturbing IT loads
                self._perturb_graph_state()

                node_list = list(self.graph.nodes.values())
                edge_list = list(self.graph.edges.values())
                tensors = graph_to_tensors(node_list, edge_list)

                x = tensors["x"].to(self.device)
                edge_index = tensors["edge_index"].to(self.device)
                edge_attr = tensors["edge_attr"].to(self.device)
                y = tensors["y"].to(self.device)

                # Forward
                output = self.gnn(x, edge_index, edge_attr)
                predicted = output["node_temps"]

                # Physics-informed loss
                losses = self.physics_loss(predicted, y, edge_index, edge_attr, x)

                # Backward
                optimizer.zero_grad()
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.gnn.parameters(), max_norm=1.0)
                optimizer.step()

                for key in epoch_losses:
                    if key in losses:
                        epoch_losses[key] += losses[key].item()

            # Average over samples
            for key in epoch_losses:
                epoch_losses[key] /= num_synthetic_samples
                history[key].append(epoch_losses[key])

            # Anneal physics weights
            self.physics_loss.anneal_weights(epoch, epochs)

            if (epoch + 1) % 20 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Total: {epoch_losses['total']:.4f} | "
                    f"Data: {epoch_losses['data']:.4f} | "
                    f"Energy: {epoch_losses['energy']:.4f}"
                )

        self._trained = True
        logger.info("Training complete.")
        return history

    # ──────────────────────────── Metrics ────────────────────────────

    def get_metrics(self) -> dict[str, float]:
        """Compute facility-level metrics from the current graph state."""
        racks = self.graph.nodes_by_type(AssetType.RACK)
        crahs = self.graph.nodes_by_type(AssetType.CRAH)

        total_it_kw = sum(getattr(r, "current_load_kw", 0) for r in racks)
        total_cooling_kw = sum(getattr(c, "current_cooling_kw", 0) for c in crahs)
        fan_power = sum(getattr(c, "fan_power_kw", 0) for c in crahs)

        facility_power = total_it_kw + total_cooling_kw + fan_power
        pue = facility_power / max(total_it_kw, 1.0)

        # Estimate water consumption (evaporative cooling)
        towers = self.graph.nodes_by_type(AssetType.COOLING_TOWER)
        water_lpm = sum(getattr(t, "evaporation_rate_lpm", 0) for t in towers)
        water_per_hour = water_lpm * 60
        it_energy_kwh = total_it_kw  # per hour
        wue = water_per_hour / max(it_energy_kwh, 1.0)

        temps = [getattr(r, "inlet_temp_c", 22) for r in racks]

        return {
            "total_it_kw": total_it_kw,
            "total_cooling_kw": total_cooling_kw,
            "facility_power_kw": facility_power,
            "pue": pue,
            "wue": wue,
            "avg_inlet_temp": np.mean(temps) if temps else 22.0,
            "max_inlet_temp": max(temps) if temps else 22.0,
            "min_inlet_temp": min(temps) if temps else 22.0,
            "num_racks": len(racks),
            "num_crahs": len(crahs),
        }

    # ──────────────────────────── Internal Helpers ────────────────────────────

    def _apply_boundary_conditions(self, conditions: dict[str, Any]) -> None:
        """Apply boundary conditions to graph nodes."""
        ambient = conditions.get("ambient_temp_c")
        if ambient is not None:
            for node in self.graph.nodes_by_type(AssetType.COOLING_TOWER):
                self.graph.update_node(node.id, temperature_c=ambient)

        it_load_factor = conditions.get("it_load_factor")
        if it_load_factor is not None:
            for rack in self.graph.nodes_by_type(AssetType.RACK):
                current = getattr(rack, "current_load_kw", 10)
                self.graph.update_node(rack.id, current_load_kw=current * it_load_factor)

    def _build_state(
        self,
        node_list: list[AssetNode],
        predicted_temps: np.ndarray,
        global_metrics: np.ndarray,
    ) -> TwinState:
        """Build a TwinState from simulation outputs."""
        node_temps = {}
        rack_temps = []
        node_ids = []

        for i, node in enumerate(node_list):
            temp = float(predicted_temps[i]) if i < len(predicted_temps) else node.temperature_c
            node_temps[node.id] = temp
            node_ids.append(node.id)
            if node.asset_type == AssetType.RACK:
                rack_temps.append(temp)

        hotspots = [nid for nid, t in node_temps.items() if t > self.ashrae_max]
        cold_spots = [nid for nid, t in node_temps.items() if t < self.ashrae_min]

        metrics = self.get_metrics()

        return TwinState(
            node_temperatures=node_temps,
            node_ids=node_ids,
            total_it_load_kw=metrics["total_it_kw"],
            total_cooling_kw=metrics["total_cooling_kw"],
            avg_inlet_temp_c=np.mean(rack_temps) if rack_temps else 22.0,
            max_inlet_temp_c=max(rack_temps) if rack_temps else 22.0,
            min_inlet_temp_c=min(rack_temps) if rack_temps else 22.0,
            wue=metrics["wue"],
            pue=metrics["pue"],
            hotspots=hotspots,
            cold_spots=cold_spots,
        )

    def _perturb_graph_state(self) -> None:
        """Perturb the graph state for synthetic training data generation."""
        rng = np.random.default_rng()

        for rack in self.graph.nodes_by_type(AssetType.RACK):
            max_load = getattr(rack, "max_load_kw", 20.0)
            new_load = rng.uniform(3.0, max_load)
            new_inlet = 18.0 + new_load * 0.5 + rng.normal(0, 1)
            self.graph.update_node(
                rack.id,
                current_load_kw=new_load,
                inlet_temp_c=new_inlet,
                outlet_temp_c=new_inlet + 10 + new_load * 0.3,
                temperature_c=new_inlet,
            )

        for crah in self.graph.nodes_by_type(AssetType.CRAH):
            cooling = getattr(crah, "cooling_capacity_kw", 200)
            new_cooling = rng.uniform(50, cooling)
            supply = rng.uniform(12, 18)
            self.graph.update_node(
                crah.id,
                current_cooling_kw=new_cooling,
                supply_air_temp_c=supply,
                temperature_c=supply,
            )

    def _reconstruct_node(self, data: dict[str, Any]) -> AssetNode | None:
        """Reconstruct a typed node from dict data."""
        from hydrotwin.physics.graph_models import ASSET_TYPE_MAP, Position3D
        asset_type_str = data.get("asset_type", "rack")
        try:
            asset_type = AssetType(asset_type_str)
        except ValueError:
            return None

        node_cls = ASSET_TYPE_MAP.get(asset_type, AssetNode)

        # Handle position
        pos_data = data.get("position", {})
        if isinstance(pos_data, dict):
            data["position"] = Position3D(**pos_data)

        try:
            return node_cls(**{k: v for k, v in data.items() if k != "asset_type" or node_cls == AssetNode})
        except Exception:
            return None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DigitalTwin:
        """Create a DigitalTwin from config."""
        twin_cfg = config.get("physics_twin", {})
        gnn_cfg = twin_cfg.get("gnn", {})
        pinn_cfg = twin_cfg.get("pinn_loss_weights", {})
        facility_cfg = config.get("facility", {}).get("thermal", {})

        pinn_weights = PhysicsLossWeights(
            energy=pinn_cfg.get("energy", 1.0),
            fourier=pinn_cfg.get("fourier", 0.5),
            mass=pinn_cfg.get("mass", 0.5),
            newton=pinn_cfg.get("newton", 0.3),
        )

        return cls(
            hidden_dim=gnn_cfg.get("hidden_dim", 64),
            num_gnn_layers=gnn_cfg.get("num_layers", 3),
            heads=gnn_cfg.get("heads", 4),
            pinn_weights=pinn_weights,
            ashrae_max_c=facility_cfg.get("server_inlet_max_c", 32.0),
            ashrae_min_c=facility_cfg.get("server_inlet_min_c", 15.0),
        )

"""
Tests for Plane 1: Asset Graph, GNN, PINN, Digital Twin, and Layout Optimizer.
"""

import numpy as np
import pytest
import torch

from hydrotwin.physics.graph_models import (
    AssetNode, AssetType, AssetEdge, EdgeType, Position3D,
    RackNode, PipeNode, PumpNode, CRAHNode, SensorNode,
    CoolingTowerNode, ColumnNode, WallNode, ZoneNode,
)
from hydrotwin.physics.asset_graph import AssetGraph
from hydrotwin.physics.thermal_gnn import (
    ThermalGNN, nodes_to_features, edges_to_index_and_features, graph_to_tensors,
    EdgeConditionedConv,
)
from hydrotwin.physics.physics_loss import PhysicsLoss, PhysicsLossWeights
from hydrotwin.physics.digital_twin import DigitalTwin, TwinState
from hydrotwin.physics.layout_optimizer import LayoutOptimizer, PlacementResult


# ═══════════════════════════════════════════════════════════
#  Phase 1: Asset Graph Tests
# ═══════════════════════════════════════════════════════════

class TestGraphModels:
    """Tests for Pydantic asset node/edge models."""

    def test_position_distance(self):
        a = Position3D(x=0, y=0, z=0)
        b = Position3D(x=3, y=4, z=0)
        assert abs(a.distance_to(b) - 5.0) < 1e-6

    def test_rack_node_defaults(self):
        rack = RackNode(name="R1")
        assert rack.asset_type == AssetType.RACK
        assert rack.max_load_kw == 20.0
        assert rack.num_servers == 42

    def test_pipe_node(self):
        pipe = PipeNode(name="P1", diameter_mm=200, length_m=15)
        assert pipe.thermal_conductivity_w_mk == 385.0

    def test_crah_node(self):
        crah = CRAHNode(name="C1", cooling_capacity_kw=250)
        assert crah.supply_air_temp_c == 15.0

    def test_sensor_node(self):
        from hydrotwin.physics.graph_models import SensorType
        s = SensorNode(name="S1", sensor_type=SensorType.FLOW_RATE)
        assert s.sensor_type == SensorType.FLOW_RATE

    def test_asset_edge(self):
        e = AssetEdge(source_id="a", target_id="b", edge_type=EdgeType.THERMAL, distance_m=3.0)
        assert e.thermal_conductivity == 1.0


class TestAssetGraph:
    """Tests for the in-memory asset graph."""

    def _make_graph(self) -> AssetGraph:
        g = AssetGraph()
        r1 = RackNode(id="r1", name="R1", position=Position3D(x=0, y=0))
        r2 = RackNode(id="r2", name="R2", position=Position3D(x=2, y=0))
        c1 = CRAHNode(id="c1", name="C1", position=Position3D(x=1, y=3))
        g.add_node(r1)
        g.add_node(r2)
        g.add_node(c1)
        g.add_edge(AssetEdge(id="e1", source_id="r1", target_id="r2", edge_type=EdgeType.THERMAL, distance_m=2.0))
        g.add_edge(AssetEdge(id="e2", source_id="r1", target_id="c1", edge_type=EdgeType.THERMAL, distance_m=3.2))
        return g

    def test_add_and_count(self):
        g = self._make_graph()
        assert g.num_nodes == 3
        assert g.num_edges == 2

    def test_get_node(self):
        g = self._make_graph()
        r1 = g.get_node("r1")
        assert r1 is not None
        assert r1.name == "R1"

    def test_remove_node_removes_edges(self):
        g = self._make_graph()
        g.remove_node("r1")
        assert g.num_nodes == 2
        assert g.num_edges == 0  # both edges connected to r1

    def test_neighbors(self):
        g = self._make_graph()
        neighbors = g.neighbors("r1")
        ids = {n.id for n in neighbors}
        assert "r2" in ids
        assert "c1" in ids

    def test_nodes_by_type(self):
        g = self._make_graph()
        racks = g.nodes_by_type(AssetType.RACK)
        assert len(racks) == 2

    def test_edges_by_type(self):
        g = self._make_graph()
        thermal = g.edges_by_type(EdgeType.THERMAL)
        assert len(thermal) == 2

    def test_connected_components(self):
        g = self._make_graph()
        comps = g.connected_components()
        assert len(comps) == 1  # all connected

    def test_shortest_path(self):
        g = self._make_graph()
        path = g.shortest_path("r2", "c1")
        assert path is not None
        assert path[0] == "r2"
        assert path[-1] == "c1"

    def test_subgraph(self):
        g = self._make_graph()
        sub = g.subgraph({"r1", "r2"})
        assert sub.num_nodes == 2
        assert sub.num_edges == 1  # only the r1-r2 edge

    def test_serialization_roundtrip(self):
        g = self._make_graph()
        d = g.to_dict()
        assert d["stats"]["num_nodes"] == 3
        assert "r1" in d["nodes"]

    def test_neo4j_statements(self):
        g = self._make_graph()
        stmts = g.to_neo4j_statements()
        assert len(stmts) == 5  # 3 nodes + 2 edges

    def test_auto_connect_by_proximity(self):
        g = AssetGraph()
        g.add_node(RackNode(id="a", position=Position3D(x=0, y=0)))
        g.add_node(RackNode(id="b", position=Position3D(x=1, y=0)))
        g.add_node(RackNode(id="c", position=Position3D(x=100, y=0)))
        count = g.auto_connect_by_proximity(max_distance_m=5.0)
        assert count == 1  # only a-b are within 5m

    def test_create_synthetic(self):
        g = AssetGraph.create_synthetic(num_racks=8, num_crahs=2, rows=2)
        assert g.num_nodes > 8  # racks + CRAHs + pumps + sensors + tower
        assert g.num_edges > 0
        racks = g.nodes_by_type(AssetType.RACK)
        assert len(racks) == 8


# ═══════════════════════════════════════════════════════════
#  Phase 2: GNN Tests
# ═══════════════════════════════════════════════════════════

class TestThermalGNN:
    """Tests for the GNN thermal model."""

    def _make_graph_tensors(self):
        graph = AssetGraph.create_synthetic(num_racks=8, num_crahs=2, rows=2)
        nodes = list(graph.nodes.values())
        edges = list(graph.edges.values())
        return graph_to_tensors(nodes, edges)

    def test_node_feature_shape(self):
        nodes = [
            RackNode(id="r1", name="R1"),
            CRAHNode(id="c1", name="C1"),
        ]
        feats = nodes_to_features(nodes)
        assert feats.shape == (2, 13)

    def test_edge_feature_shape(self):
        edges = [
            AssetEdge(source_id="r1", target_id="c1", edge_type=EdgeType.THERMAL, distance_m=3.0),
        ]
        idx, attr = edges_to_index_and_features(edges, {"r1": 0, "c1": 1})
        assert idx.shape == (2, 2)   # bidirectional
        assert attr.shape == (2, 7)

    def test_graph_to_tensors(self):
        tensors = self._make_graph_tensors()
        assert "x" in tensors
        assert "edge_index" in tensors
        assert "y" in tensors
        assert tensors["x"].shape[1] == 13

    def test_gnn_forward_pass(self):
        tensors = self._make_graph_tensors()
        model = ThermalGNN(node_feature_dim=13, edge_feature_dim=7, hidden_dim=32, num_layers=2)
        output = model(tensors["x"], tensors["edge_index"], tensors["edge_attr"])

        assert "node_temps" in output
        assert "node_embeddings" in output
        assert "global_metrics" in output
        assert output["node_temps"].shape[0] == tensors["num_nodes"]
        assert output["global_metrics"].shape[0] == 4

    def test_gnn_no_edges(self):
        """GNN should handle graphs with no edges."""
        nodes = [RackNode(id="r1"), RackNode(id="r2")]
        x = nodes_to_features(nodes)
        empty_index = torch.zeros(2, 0, dtype=torch.long)
        empty_attr = torch.zeros(0, 7)
        model = ThermalGNN(hidden_dim=16, num_layers=1)
        output = model(x, empty_index, empty_attr)
        assert output["node_temps"].shape[0] == 2

    def test_predict_temperatures(self):
        tensors = self._make_graph_tensors()
        model = ThermalGNN(hidden_dim=16, num_layers=1)
        temps = model.predict_temperatures(tensors["x"], tensors["edge_index"], tensors["edge_attr"])
        assert isinstance(temps, np.ndarray)
        assert len(temps) == tensors["num_nodes"]


# ═══════════════════════════════════════════════════════════
#  Phase 3: PINN Loss Tests
# ═══════════════════════════════════════════════════════════

class TestPhysicsLoss:
    """Tests for physics-informed loss functions."""

    def _make_data(self):
        graph = AssetGraph.create_synthetic(num_racks=4, num_crahs=2, rows=2)
        nodes = list(graph.nodes.values())
        edges = list(graph.edges.values())
        tensors = graph_to_tensors(nodes, edges)
        predicted = torch.randn(tensors["num_nodes"], requires_grad=True)
        return tensors, predicted

    def test_loss_computes_without_error(self):
        tensors, predicted = self._make_data()
        loss_fn = PhysicsLoss()
        losses = loss_fn(
            predicted, tensors["y"],
            tensors["edge_index"], tensors["edge_attr"], tensors["x"],
        )
        assert "total" in losses
        assert losses["total"].requires_grad

    def test_all_loss_terms_present(self):
        tensors, predicted = self._make_data()
        loss_fn = PhysicsLoss()
        losses = loss_fn(predicted, tensors["y"], tensors["edge_index"], tensors["edge_attr"], tensors["x"])
        for key in ["data", "energy", "fourier", "mass", "newton"]:
            assert key in losses

    def test_perfect_prediction_low_data_loss(self):
        tensors, _ = self._make_data()
        perfect = tensors["y"].clone()
        loss_fn = PhysicsLoss()
        losses = loss_fn(perfect, tensors["y"], tensors["edge_index"], tensors["edge_attr"], tensors["x"])
        assert losses["data"].item() < 1e-6

    def test_anneal_decreases_weights(self):
        loss_fn = PhysicsLoss(PhysicsLossWeights(energy=1.0))
        initial = loss_fn.weights.energy
        loss_fn.anneal_weights(50, 100)
        assert loss_fn.weights.energy < initial

    def test_from_config(self):
        config = {"physics_twin": {"pinn_loss_weights": {"energy": 2.0, "fourier": 0.8}}}
        loss_fn = PhysicsLoss.from_config(config)
        assert loss_fn.weights.energy == 2.0
        assert loss_fn.weights.fourier == 0.8


# ═══════════════════════════════════════════════════════════
#  Phase 4: Digital Twin Tests
# ═══════════════════════════════════════════════════════════

class TestDigitalTwin:
    """Tests for the unified simulation engine."""

    def _make_twin(self) -> DigitalTwin:
        graph = AssetGraph.create_synthetic(num_racks=8, num_crahs=2, rows=2)
        return DigitalTwin(graph=graph, hidden_dim=16, num_gnn_layers=1, device="cpu")

    def test_simulate_returns_state(self):
        twin = self._make_twin()
        state = twin.simulate()
        assert isinstance(state, TwinState)
        assert len(state.node_temperatures) > 0
        assert state.total_it_load_kw > 0

    def test_what_if_remove_node(self):
        twin = self._make_twin()
        racks = twin.graph.nodes_by_type(AssetType.RACK)
        if racks:
            state = twin.what_if({"type": "remove_node", "node_id": racks[0].id})
            assert isinstance(state, TwinState)
            assert "what_if" in state.simulation_mode

    def test_what_if_update_node(self):
        twin = self._make_twin()
        racks = twin.graph.nodes_by_type(AssetType.RACK)
        if racks:
            state = twin.what_if({
                "type": "update_node",
                "node_id": racks[0].id,
                "current_load_kw": 20.0,
            })
            assert isinstance(state, TwinState)

    def test_calibrate(self):
        twin = self._make_twin()
        sensors = twin.graph.nodes_by_type(AssetType.SENSOR)
        if sensors:
            readings = {sensors[0].id: 25.0}
            count = twin.calibrate(readings)
            assert count == 1

    def test_get_metrics(self):
        twin = self._make_twin()
        metrics = twin.get_metrics()
        assert "pue" in metrics
        assert "wue" in metrics
        assert metrics["pue"] >= 1.0
        assert metrics["num_racks"] == 8

    def test_state_summary(self):
        twin = self._make_twin()
        state = twin.simulate()
        summary = state.summary()
        assert "avg_temp" in summary
        assert "pue" in summary

    def test_train_short(self):
        """Quick training sanity check (1 epoch, 2 samples)."""
        twin = self._make_twin()
        history = twin.train(epochs=1, num_synthetic_samples=2)
        assert "total" in history
        assert len(history["total"]) == 1


# ═══════════════════════════════════════════════════════════
#  Phase 5: Layout Optimizer Tests
# ═══════════════════════════════════════════════════════════

class TestLayoutOptimizer:
    """Tests for the layout optimizer."""

    def _make_optimizer(self) -> LayoutOptimizer:
        graph = AssetGraph.create_synthetic(num_racks=8, num_crahs=2, rows=2)
        return LayoutOptimizer(graph=graph)

    def test_rack_placement_produces_result(self):
        opt = self._make_optimizer()
        result = opt.optimize_rack_placement(max_iterations=5)
        assert isinstance(result, PlacementResult)
        assert result.iterations > 0

    def test_recommend_cooling_units(self):
        opt = self._make_optimizer()
        recommendations = opt.recommend_cooling_units(max_recommendations=2)
        # May or may not find hot racks depending on synthetic data
        assert isinstance(recommendations, list)

    def test_pipe_routing(self):
        opt = self._make_optimizer()
        result = opt.optimize_pipe_routing()
        assert result.total_pipe_length_m >= 0


# ═══════════════════════════════════════════════════════════
#  Phase 7: New Event Schemas
# ═══════════════════════════════════════════════════════════

class TestPhysicsEventSchemas:
    """Tests for Plane 1 event schemas."""

    def test_physics_recompute(self):
        from hydrotwin.events.schemas import PhysicsRecompute
        event = PhysicsRecompute(trigger_reason="layout_change", affected_zones=["zone-1"])
        assert event.source_plane == 1
        assert event.priority == "normal"

    def test_twin_state_snapshot(self):
        from hydrotwin.events.schemas import TwinStateSnapshot
        event = TwinStateSnapshot(
            avg_temp_c=23.5, max_temp_c=28.0, min_temp_c=18.0,
            total_it_kw=4000, total_cooling_kw=1200,
            pue=1.3, wue=0.8, num_hotspots=2, hotspot_ids=["r1", "r2"],
            num_nodes=120,
        )
        assert event.source_plane == 1
        assert event.num_hotspots == 2

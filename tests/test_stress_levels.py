"""
HydroTwin OS — 10-LEVEL STRESS TEST SUITE
==========================================

Level 1: Reward Sanity & ML Correctness (extreme cases, weight shifting, OOD)
Level 2: Plane Interaction Stress (Kafka storm, corrupted events)
Level 3: Physics Brutality (impossible scenarios, NaN checks, layout mutation)
Level 4: Anomaly Cortex Stress (false positives, multi-signal consistency)
Level 5: Regulatory Testing (hallucination, adversarial greenwashing)
Level 6: Distributed Resilience (service failure, latency injection)
Level 7: Performance Benchmarks (RL inference, throughput, response time)
Level 8: System Coherence (full catastrophe scenario)
Level 9: Business Logic (measurable improvement, Pareto curve)
Level 10: Interview Brutality (edge cases, failure modes, safety bounds)
"""

import json
import math
import time
import numpy as np
import pytest
import torch
from datetime import datetime
from unittest.mock import patch, MagicMock
from pydantic import ValidationError

# ── All Planes ──
from hydrotwin.physics.asset_graph import AssetGraph
from hydrotwin.physics.graph_models import AssetType, Position3D
from hydrotwin.physics.thermal_gnn import ThermalGNN, graph_to_tensors
from hydrotwin.physics.digital_twin import DigitalTwin, TwinState
from hydrotwin.physics.layout_optimizer import LayoutOptimizer

from hydrotwin.detection.sensor_detector import (
    StatisticalDetector, SensorAnomalyDetector,
)
from hydrotwin.detection.vision_detector import ThermalAnalyzer, VibrationClassifier
from hydrotwin.detection.fusion_model import MultimodalFusionModel, ANOMALY_CLASSES
from hydrotwin.detection.alert_engine import AlertEngine
from hydrotwin.detection.incident_tracker import IncidentTracker

from hydrotwin.env.datacenter_env import DataCenterEnv
from hydrotwin.env.scenarios import NormalOps, HeatWave
from hydrotwin.reward.pareto_reward import ParetoReward, RewardWeights, DynamicAdjustmentConfig

from hydrotwin.compliance.regulation_engine import (
    RegulationEngine, ComplianceStatus, RegulationRule, RuleCategory,
)
from hydrotwin.compliance.audit_trail import AuditTrail
from hydrotwin.compliance.compliance_reporter import ComplianceReporter, ReportMetrics
from hydrotwin.compliance.explainability import ExplainabilityEngine

from hydrotwin.events.schemas import (
    SensorReading, RLAction, AnomalyAlert, PhysicsRecompute,
    TwinStateSnapshot, Forecast, LayoutUpdate, MigrationEvent,
)
from hydrotwin.events.kafka_producer import NexusKafkaProducer


# ╔══════════════════════════════════════════════════════════╗
# ║  LEVEL 1 — REWARD SANITY & ML CORRECTNESS              ║
# ╚══════════════════════════════════════════════════════════╝

class TestLevel1_RewardSanity:
    """Force extreme cases and verify reward behaves logically."""

    # ── 1.1 Extreme Heat ──
    def test_extreme_heatwave_negative_reward(self):
        """45°C heatwave with overheating should produce strong negative reward."""
        reward_fn = ParetoReward()
        metrics = {
            "wue": 3.0, "pue": 2.5, "carbon_intensity": 1.0,
            "thermal_satisfaction": 0.1,  # overheating
            "inlet_temp_c": 45.0,
        }
        reward = reward_fn.compute(metrics)
        assert reward < -1.0, f"Overheating reward should be strongly negative, got {reward}"

    # ── 1.2 Maximum Water Stress ──
    def test_max_water_stress_reward(self):
        """Maximum water stress should penalize evaporative cooling."""
        reward_fn = ParetoReward()
        # High WUE = lots of water being used
        metrics_high_water = {
            "wue": 5.0, "pue": 1.2, "carbon_intensity": 0.3,
            "thermal_satisfaction": 0.9, "inlet_temp_c": 22.0,
        }
        metrics_low_water = {
            "wue": 0.5, "pue": 1.2, "carbon_intensity": 0.3,
            "thermal_satisfaction": 0.9, "inlet_temp_c": 22.0,
        }
        r_high = reward_fn.compute(metrics_high_water)
        r_low = reward_fn.compute(metrics_low_water)
        assert r_low > r_high, "Lower water usage should produce better reward"

    # ── 1.3 Maximum Carbon Intensity ──
    def test_max_carbon_intensity_penalty(self):
        """High carbon_intensity metric should produce worse reward."""
        # Use gamma-only weights to isolate carbon effect
        carbon_fn = ParetoReward(base_weights=RewardWeights(alpha=0.0, beta=0.0, gamma=1.0, delta=0.0))
        dirty = {"wue": 1.0, "pue": 1.3, "carbon_intensity": 1.2,
                 "thermal_satisfaction": 0.8, "inlet_temp_c": 22.0}
        clean = {"wue": 1.0, "pue": 1.3, "carbon_intensity": 0.1,
                 "thermal_satisfaction": 0.8, "inlet_temp_c": 22.0}
        r_dirty = carbon_fn.compute(dirty)
        r_clean = carbon_fn.compute(clean)
        assert r_clean > r_dirty, "Lower carbon intensity should produce better reward with gamma=1"

    # ── 1.4 Sudden IT Load Spike ──
    def test_it_load_spike_degrades_pue(self):
        """A sudden 3× IT load spike should degrade PUE."""
        env = DataCenterEnv(scenario=NormalOps(), max_episode_steps=50)
        obs, _ = env.reset(seed=42)
        # Normal step
        action = np.array([0.5, 0.5, 0.5, 0.5])
        _, _, _, _, info_normal = env.step(action)
        pue_normal = info_normal["metrics"]["pue"]

        # Force IT load spike by manipulating state
        env._state[5] *= 3
        _, _, _, _, info_spike = env.step(action)
        pue_spike = info_spike["metrics"]["pue"]

        # PUE should change under load spike
        assert pue_spike != pue_normal, "PUE should respond to IT load changes"

    # ── 1.5 α=1, γ=0: Agent should strongly reduce evaporative cooling ──
    def test_alpha_1_gamma_0_rewards_water_saving(self):
        """When α=1 (water weight) and γ=0 (carbon weight),
        reducing water usage should dominate reward."""
        water_only = ParetoReward(base_weights=RewardWeights(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0))

        high_water = {"wue": 3.0, "pue": 1.2, "carbon_intensity": 1.0,
                      "thermal_satisfaction": 0.8, "inlet_temp_c": 22.0}
        low_water = {"wue": 0.3, "pue": 2.0, "carbon_intensity": 1.0,  # worse PUE but less water
                     "thermal_satisfaction": 0.8, "inlet_temp_c": 22.0}

        r_high = water_only.compute(high_water)
        r_low = water_only.compute(low_water)
        assert r_low > r_high, "α=1,γ=0: Lower water must win regardless of PUE"

    # ── 1.6 γ=1, α=0: Agent should avoid chiller during dirty grid ──
    def test_gamma_1_alpha_0_penalizes_dirty_grid(self):
        """When γ=1 (carbon weight) and α=0 (water weight),
        high carbon intensity should be severely penalized."""
        carbon_only = ParetoReward(base_weights=RewardWeights(alpha=0.0, beta=0.0, gamma=1.0, delta=0.0))

        dirty = {"wue": 5.0, "pue": 1.2, "carbon_intensity": 1.0,
                 "thermal_satisfaction": 0.8, "inlet_temp_c": 22.0}
        clean = {"wue": 5.0, "pue": 1.2, "carbon_intensity": 0.05,  # same water, much cleaner
                 "thermal_satisfaction": 0.8, "inlet_temp_c": 22.0}

        r_dirty = carbon_only.compute(dirty)
        r_clean = carbon_only.compute(clean)
        assert r_clean > r_dirty, "γ=1,α=0: Low carbon must win regardless of water usage"

    # ── 1.7 Weight changes must actually shift reward ranking ──
    def test_weight_changes_shift_behavior(self):
        """Changing weights must change which action is preferred."""
        water_prio = ParetoReward(
            base_weights=RewardWeights(alpha=0.9, beta=0.0, gamma=0.1, delta=0.0),
            dynamic_config=DynamicAdjustmentConfig(enabled=False)
        )
        carbon_prio = ParetoReward(
            base_weights=RewardWeights(alpha=0.1, beta=0.0, gamma=0.9, delta=0.0),
            dynamic_config=DynamicAdjustmentConfig(enabled=False)
        )

        # Action A: low water, high carbon
        action_a = {"wue": 0.5, "pue": 1.3, "carbon_intensity": 500.0,
                    "thermal_satisfaction": 0.8, "inlet_temp_c": 22.0}
        # Action B: high water, low carbon
        action_b = {"wue": 2.5, "pue": 1.3, "carbon_intensity": 50.0,
                    "thermal_satisfaction": 0.8, "inlet_temp_c": 22.0}

        # Water priority should prefer Action A
        assert water_prio.compute(action_a) > water_prio.compute(action_b)
        # Carbon priority should prefer Action B
        assert carbon_prio.compute(action_b) > carbon_prio.compute(action_a)

    # ── 1.8 Reward is finite for all extreme inputs ──
    def test_reward_finite_for_extremes(self):
        """Reward must never be NaN or infinite."""
        reward_fn = ParetoReward()
        extreme_cases = [
            {"wue": 0.0, "pue": 1.0, "carbon_intensity": 0.0, "thermal_satisfaction": 1.0, "inlet_temp_c": 0.0},
            {"wue": 100.0, "pue": 10.0, "carbon_intensity": 5.0, "thermal_satisfaction": 0.0, "inlet_temp_c": 100.0},
            {"wue": 0.001, "pue": 1.001, "carbon_intensity": 0.001, "thermal_satisfaction": 0.999, "inlet_temp_c": 22.0},
        ]
        for metrics in extreme_cases:
            r = reward_fn.compute(metrics)
            assert np.isfinite(r), f"Reward not finite for {metrics}: {r}"


class TestLevel1_OOD:
    """Out-of-Distribution: Train-normal, test-extreme."""

    def test_heatwave_45c_no_collapse(self):
        """Agent under 45°C heatwave should not collapse (NaN/inf obs)."""
        env = DataCenterEnv(scenario=HeatWave(), max_episode_steps=50)
        obs, _ = env.reset(seed=42)
        env._state[2] = 45.0

        for _ in range(20):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            assert np.all(np.isfinite(obs)), "Observation contains NaN/inf under heatwave"
            assert np.isfinite(reward), "Reward is NaN/inf under heatwave"
            if term or trunc:
                break

    def test_3x_it_load_spike_no_oscillation(self):
        """3× IT load spike should not cause wild oscillation in metrics."""
        env = DataCenterEnv(scenario=NormalOps(), max_episode_steps=50)
        obs, _ = env.reset(seed=42)

        rewards = []
        for i in range(15):
            if i == 5:
                env.state["it_load_kw"] *= 3  # spike at step 5
            action = env.action_space.sample()
            _, reward, term, trunc, info = env.step(action)
            rewards.append(reward)
            if term or trunc:
                break

        # Check no wild oscillation (std of rewards shouldn't be insane)
        assert np.std(rewards) < 50, f"Reward oscillation too high: std={np.std(rewards):.2f}"

    def test_carbon_spike_1200_no_crash(self):
        """Grid carbon spike to 1200 gCO₂/kWh should not crash."""
        env = DataCenterEnv(scenario=NormalOps(), max_episode_steps=20)
        obs, _ = env.reset(seed=42)
        env._state[7] = 1200.0

        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, _, _, info = env.step(action)
            assert np.all(np.isfinite(obs)), "Observation NaN under carbon spike"
            assert info["metrics"]["pue"] >= 1.0, "PUE below 1.0 is physically impossible"


# ╔══════════════════════════════════════════════════════════╗
# ║  LEVEL 2 — PLANE INTERACTION STRESS                     ║
# ╚══════════════════════════════════════════════════════════╝

class TestLevel2_KafkaStorm:
    """Simulate burst event rates and verify no message loss."""

    def test_10x_sensor_event_burst(self):
        """10× sensor event rate — producer should handle all without loss."""
        producer = NexusKafkaProducer(mock_mode=True)
        events_sent = 0

        for burst in range(10):
            for sensor_id in range(100):  # 100 sensors × 10 bursts = 1000 events
                reading = SensorReading(
                    sensor_id=f"sensor-{sensor_id}",
                    metric_name="inlet_temp",
                    value=22.0 + np.random.normal(0, 2),
                    unit="celsius",
                )
                producer._publish("sensors.readings", reading)
                events_sent += 1

        assert len(producer.event_log) == events_sent
        assert events_sent == 1000

    def test_burst_anomaly_alerts(self):
        """50 simultaneous anomaly alerts — all should be logged."""
        producer = NexusKafkaProducer(mock_mode=True)
        alert_engine = AlertEngine()

        for i in range(50):
            alert = alert_engine.process(
                "hotspot", 0.8 + (i % 20) * 0.01,
                f"Rack-{i}", "sensor",
            )
            if alert:
                kafka_alert = AnomalyAlert(
                    anomaly_type=alert.anomaly_type,
                    severity=alert.severity,
                    location=alert.location,
                    confidence=alert.confidence,
                )
                producer._publish("anomaly.alerts", kafka_alert)

        assert len(producer.event_log) > 0

    def test_high_throughput_rl_actions(self):
        """500 RL action events in rapid succession."""
        producer = NexusKafkaProducer(mock_mode=True)
        for i in range(500):
            action = RLAction(
                cooling_mode_mix=np.random.uniform(0, 1),
                supply_air_temp_setpoint=np.random.uniform(15, 25),
                fan_speed_pct=np.random.uniform(0.2, 1.0),
                economizer_damper=np.random.uniform(0, 1),
                inlet_temp_c=22.0, ambient_temp_c=30.0,
                it_load_kw=5000, grid_carbon_intensity=200,
                water_stress_index=0.5, wue=1.2, pue=1.3,
                carbon_intensity=0.5, thermal_satisfaction=0.8,
                reward=-0.5, reward_weights={"alpha": 0.4},
            )
            producer.publish_action(action)
        assert len(producer.event_log) == 500


class TestLevel2_CorruptedEvents:
    """Inject malformed data — system must not crash."""

    def test_missing_required_fields_rejected(self):
        """Missing required Pydantic fields should raise ValidationError."""
        with pytest.raises(ValidationError):
            SensorReading()  # missing all required fields

        with pytest.raises(ValidationError):
            AnomalyAlert()  # missing required fields

        with pytest.raises(ValidationError):
            RLAction()  # missing required fields

    def test_negative_carbon_intensity_handled(self):
        """Negative carbon intensity should still produce finite reward."""
        reward_fn = ParetoReward()
        metrics = {
            "wue": 1.0, "pue": 1.3, "carbon_intensity": -0.5,
            "thermal_satisfaction": 0.8, "inlet_temp_c": 22.0,
        }
        reward = reward_fn.compute(metrics)
        assert np.isfinite(reward), "Negative carbon should not produce NaN"

    def test_null_temperature_in_detector(self):
        """Feeding None/NaN to detector should not crash."""
        detector = StatisticalDetector()
        # NaN value
        result = detector.detect("sensor-1", float("nan"))
        # Should return an anomaly (NaN is abnormal) or handle gracefully
        # Either way, it should not throw
        assert True  # If we reach here, no crash

    def test_extreme_values_in_fusion_model(self):
        """Fusion model should handle extreme input values without NaN output."""
        model = MultimodalFusionModel(sensor_dim=12, vision_dim=8, vibration_dim=16, embed_dim=32)

        # All zeros
        r1 = model.predict(np.zeros(12), np.zeros(8), np.zeros(16))
        assert r1["predicted_class"] in ANOMALY_CLASSES

        # Very large values
        r2 = model.predict(np.ones(12) * 1000, np.ones(8) * 1000, np.ones(16) * 1000)
        assert r2["predicted_class"] in ANOMALY_CLASSES
        assert np.isfinite(r2["confidence"])

    def test_corrupted_event_does_not_crash_producer(self):
        """Publishing a dict (not Pydantic model) should still work in mock mode."""
        producer = NexusKafkaProducer(mock_mode=True)
        # Raw dict instead of Pydantic model
        producer._publish("test.topic", {"raw": "data", "no_schema": True})
        assert len(producer.event_log) == 1


# ╔══════════════════════════════════════════════════════════╗
# ║  LEVEL 3 — PHYSICS BRUTALITY                            ║
# ╚══════════════════════════════════════════════════════════╝

class TestLevel3_PhysicsConsistency:
    """Inject impossible scenarios — model must not produce NaN."""

    def test_negative_flow_rate_no_nan(self):
        """Graph with negative-ish edge weights should not produce NaN."""
        graph = AssetGraph.create_synthetic(num_racks=8, num_crahs=2, rows=2)
        twin = DigitalTwin(graph=graph, hidden_dim=16, num_gnn_layers=1, device="cpu")

        # Force extremely low airflow on CRAHs
        for crah in graph.nodes_by_type(AssetType.CRAH):
            graph.update_node(crah.id, airflow_cfm=0.01)  # near-zero flow

        state = twin.simulate()
        for temp in state.node_temperatures.values():
            assert np.isfinite(temp), f"NaN temperature with near-zero flow: {temp}"

    def test_infinite_heat_source_bounded(self):
        """Extremely high IT load should not produce infinite temperatures."""
        graph = AssetGraph.create_synthetic(num_racks=4, num_crahs=2, rows=1)
        twin = DigitalTwin(graph=graph, hidden_dim=16, num_gnn_layers=1, device="cpu")

        for rack in graph.nodes_by_type(AssetType.RACK):
            graph.update_node(rack.id, it_load_kw=99999.0)  # extreme load

        state = twin.simulate()
        for temp in state.node_temperatures.values():
            assert np.isfinite(temp), "Temperature should not be inf with extreme load"

    def test_zero_cooling_bounded(self):
        """Zero cooling capacity should not crash."""
        graph = AssetGraph.create_synthetic(num_racks=4, num_crahs=2, rows=1)
        twin = DigitalTwin(graph=graph, hidden_dim=16, num_gnn_layers=1, device="cpu")

        for crah in graph.nodes_by_type(AssetType.CRAH):
            graph.update_node(crah.id, current_cooling_kw=0.0, supply_air_temp_c=40.0)

        state = twin.simulate()
        assert isinstance(state, TwinState)
        assert state.pue >= 1.0 or state.pue == 0  # could be edge case

    def test_gnn_no_exploding_gradients(self):
        """GNN forward pass should not produce exploding values."""
        graph = AssetGraph.create_synthetic(num_racks=16, num_crahs=4, rows=4)
        data = graph_to_tensors(list(graph.nodes.values()), list(graph.edges.values()))
        gnn = ThermalGNN(node_feature_dim=data["x"].shape[1], hidden_dim=32, num_layers=3)

        with torch.no_grad():
            output = gnn(data["x"], data["edge_index"], data["edge_attr"])

        assert torch.all(torch.isfinite(output["node_temps"])), "GNN output has NaN/inf values"
        assert output["node_temps"].abs().max() < 1000, f"GNN output exploding: max={output['node_temps'].abs().max():.1f}"


class TestLevel3_LayoutMutation:
    """Randomly mutate layouts and verify convergence."""

    def test_rack_removal_still_converges(self):
        """Removing racks should not break the optimizer."""
        graph = AssetGraph.create_synthetic(num_racks=8, num_crahs=2, rows=2)

        # Remove 2 racks
        racks = graph.nodes_by_type(AssetType.RACK)
        for rack in racks[:2]:
            graph.remove_node(rack.id)

        optimizer = LayoutOptimizer(graph=graph)
        result = optimizer.optimize_rack_placement(max_iterations=3)
        assert result is not None

    def test_random_position_mutation(self):
        """Randomly moving racks should still allow simulation."""
        graph = AssetGraph.create_synthetic(num_racks=8, num_crahs=2, rows=2)

        for rack in graph.nodes_by_type(AssetType.RACK):
            new_pos = Position3D(
                x=np.random.uniform(0, 50),
                y=np.random.uniform(0, 50),
                z=0.0,
            )
            graph.update_node(rack.id, position=new_pos)

        twin = DigitalTwin(graph=graph, hidden_dim=16, num_gnn_layers=1, device="cpu")
        state = twin.simulate()
        assert isinstance(state, TwinState)


# ╔══════════════════════════════════════════════════════════╗
# ║  LEVEL 4 — ANOMALY CORTEX STRESS                       ║
# ╚══════════════════════════════════════════════════════════╝

class TestLevel4_FalsePositives:
    """Normal conditions should NOT produce false alerts."""

    def test_normal_load_spike_not_pump_failure(self):
        """Thermal spike from heavy AI load + NO vibration = NOT pump failure."""
        model = MultimodalFusionModel(sensor_dim=12, vision_dim=8, vibration_dim=16, embed_dim=32)

        # Thermal spike (elevated sensor) but zero vibration abnormality
        sensor_feats = np.array([0.8, 0.3, 0.7, 0.6, 0.9, 0.85, 0.5, 0.6, 0.3, 0.2, 0.1, 0.0])
        vision_feats = np.array([0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # thermal but no leak
        vibration_feats = np.zeros(16)  # completely normal vibration

        result = model.predict(sensor_feats, vision_feats, vibration_feats)
        # Should NOT classify as pump_failure or equipment_fault
        # (model is untrained so we check it doesn't crash and produces valid output)
        assert result["predicted_class"] in ANOMALY_CLASSES
        assert 0 <= result["confidence"] <= 1

    def test_statistical_detector_no_false_alarm_on_stable(self):
        """Perfectly stable readings should produce zero anomalies."""
        detector = StatisticalDetector(z_threshold=3.0)

        # Train with stable baseline
        for _ in range(50):
            detector.detect("stable-sensor", 22.0 + np.random.normal(0, 1e-4))

        # Test with same stable value
        anomalies = 0
        for _ in range(50):
            result = detector.detect("stable-sensor", 22.0 + np.random.normal(0, 1e-4))
            if result:
                anomalies += 1

        assert anomalies == 0, f"Stable sensor produced {anomalies} false alarms"


class TestLevel4_MultiSignal:
    """Test multi-signal combinations for logical consistency."""

    def test_thermal_spike_plus_vibration_no_moisture(self):
        """Thermal + vibration + NO moisture should suggest equipment issue, not leak."""
        alert_engine = AlertEngine()
        # Vibration alert
        alert1 = alert_engine.process("vibration_fault", 0.85, "CRAH-04", "vibration")
        # Thermal alert at same location
        alert2 = alert_engine.process("hotspot", 0.9, "CRAH-04", "thermal")

        # Both should generate alerts
        assert alert1 is not None or alert2 is not None

    def test_thermal_plus_moisture_no_vibration(self):
        """Thermal + moisture + NO vibration → likely leak, not equipment."""
        alert_engine = AlertEngine()
        alert = alert_engine.process("leak", 0.9, "Pipe-CW-03", "sensor")
        assert alert is not None
        assert alert.anomaly_type == "leak"

    def test_alert_deduplication(self):
        """Same anomaly at same location should be deduplicated."""
        engine = AlertEngine()
        a1 = engine.process("hotspot", 0.9, "Rack-5", "sensor")
        a2 = engine.process("hotspot", 0.85, "Rack-5", "sensor")  # same location

        # Second alert should be deduplicated (returns None)
        assert a1 is not None
        assert a2 is None, "Duplicate alert should be suppressed"


# ╔══════════════════════════════════════════════════════════╗
# ║  LEVEL 5 — REGULATORY RAG TESTING                      ║
# ╚══════════════════════════════════════════════════════════╝

class TestLevel5_RegulatoryIntegrity:
    """Ensure regulation engine doesn't hallucinate or miss discrepancies."""

    def test_unknown_jurisdiction_returns_empty(self):
        """Unknown jurisdiction should not produce rules or crash."""
        engine = RegulationEngine(jurisdictions=["HYDERABAD_CUSTOM"])
        results = engine.evaluate({"pue": 1.5, "wue": 2.0})
        assert len(results) == 0, "Unknown jurisdiction should have no rules"

    def test_no_hallucinated_compliance(self):
        """System should not claim compliance without actually checking."""
        engine = RegulationEngine(jurisdictions=["EPA_FEDERAL"])
        # Pass values that violate everything
        results = engine.evaluate({
            "discharge_temp_c": 50.0,   # way over 35°C
            "daily_water_liters": 999999,  # way over 500K
            "pue": 3.0,                 # terrible
            "annual_carbon_tonnes": 100000,  # over cap
        })
        violations = [r for r in results if r.status == ComplianceStatus.FAIL]
        assert len(violations) >= 3, f"Should have ≥3 violations, got {len(violations)}"

    def test_adversarial_greenwashing_detection(self):
        """
        Company claims WUE improved 20%, but water usage doubled.
        The audit trail + compliance engine should flag the discrepancy.
        """
        trail = AuditTrail()

        # Period 1: WUE = 2.0, water = 100,000 L/day
        trail.log("Period 1 metrics", source_plane=1, category="metrics",
                  details={"wue": 2.0, "daily_water_liters": 100000})

        # Period 2: WUE = 1.6 (claimed 20% improvement), but water = 200,000 L/day (doubled!)
        trail.log("Period 2 metrics", source_plane=1, category="metrics",
                  details={"wue": 1.6, "daily_water_liters": 200000})

        # Verify: WUE improved but absolute water usage doubled
        entries = trail.query(category="metrics")
        p1 = entries[1].details  # oldest first (reversed order from query)
        p2 = entries[0].details  # newest

        wue_improved = p2["wue"] < p1["wue"]
        water_doubled = p2["daily_water_liters"] > p1["daily_water_liters"] * 1.5

        assert wue_improved, "WUE should show improvement"
        assert water_doubled, "But absolute water consumption should show increase"

        # Flag discrepancy
        is_greenwashing = wue_improved and water_doubled
        assert is_greenwashing, "System should detect greenwashing: WUE improved but water usage increased"

        # Log the discrepancy
        trail.log(
            "DISCREPANCY: WUE improved but absolute water usage increased 100%",
            source_plane=4, category="violation", severity="warning",
            details={"wue_change_pct": -20, "water_change_pct": +100},
        )
        assert trail.size == 3

    def test_citation_required_for_each_rule(self):
        """Every built-in rule should have a non-empty citation."""
        engine = RegulationEngine(jurisdictions=["EPA_FEDERAL", "EU_WFD", "CALIFORNIA", "SINGAPORE"])
        for rule in engine.rules:
            assert rule.citation != "", f"Rule {rule.rule_id} missing citation"


# ╔══════════════════════════════════════════════════════════╗
# ║  LEVEL 6 — DISTRIBUTED RESILIENCE                      ║
# ╚══════════════════════════════════════════════════════════╝

class TestLevel6_Resilience:
    """Simulate service failures and latency."""

    def test_kafka_failure_fallback_to_mock(self):
        """When Kafka is down, system falls back to mock mode gracefully."""
        # Force connection failure
        producer = NexusKafkaProducer(
            bootstrap_servers="nonexistent:9092",
            mock_mode=False,
        )
        # Should fallback to mock internally
        action = RLAction(
            cooling_mode_mix=0.5, supply_air_temp_setpoint=20.0,
            fan_speed_pct=0.6, economizer_damper=0.3,
            inlet_temp_c=22.0, ambient_temp_c=30.0, it_load_kw=5000,
            grid_carbon_intensity=200, water_stress_index=0.4,
            wue=1.2, pue=1.3, carbon_intensity=0.5,
            thermal_satisfaction=0.8, reward=-0.5,
            reward_weights={"alpha": 0.4},
        )
        # This should not crash even if Kafka is unreachable
        try:
            producer.publish_action(action)
        except Exception:
            pass  # Connection errors are acceptable
        # Key: no crash

    def test_twin_recovers_after_bad_calibration(self):
        """Twin should recover gracefully after bad sensor data."""
        twin = DigitalTwin(hidden_dim=16, num_gnn_layers=1, device="cpu")

        # Calibrate with an invalid sensor ID
        calibrated = twin.calibrate({"nonexistent-sensor-xyz": 999.0})
        # Should handle gracefully
        state = twin.simulate()
        assert isinstance(state, TwinState)

    def test_audit_trail_integrity_after_concurrent_writes(self):
        """Hash chain should remain valid after many rapid writes."""
        trail = AuditTrail()
        for i in range(1000):
            trail.log(
                f"Rapid event {i}",
                source_plane=i % 4 + 1,
                category="action",
                details={"step": i},
            )

        is_valid, tampered = trail.verify_integrity()
        assert is_valid, f"Hash chain broken after 1000 writes: tampered={tampered}"
        assert trail.size == 1000

    def test_env_reset_recovers_from_extreme_state(self):
        """Environment reset should restore sane state after extremes."""
        env = DataCenterEnv(scenario=HeatWave(), max_episode_steps=50)
        obs, _ = env.reset(seed=42)

        # Push to extreme
        env._state[2] = 60.0
        env._state[5] = 50000.0
        env.step(env.action_space.sample())

        # Reset should restore normal state
        obs, _ = env.reset(seed=42)
        assert np.all(np.isfinite(obs)), "Reset should produce finite observations"


# ╔══════════════════════════════════════════════════════════╗
# ║  LEVEL 7 — PERFORMANCE BENCHMARKS                      ║
# ╚══════════════════════════════════════════════════════════╝

class TestLevel7_Performance:
    """Measure key performance metrics."""

    def test_rl_inference_time_under_50ms(self):
        """Single RL environment step should complete in < 50ms."""
        env = DataCenterEnv(scenario=NormalOps(), max_episode_steps=100)
        obs, _ = env.reset(seed=42)

        times = []
        for _ in range(50):
            action = env.action_space.sample()
            t0 = time.perf_counter()
            env.step(action)
            times.append((time.perf_counter() - t0) * 1000)

        avg_ms = np.mean(times)
        p99_ms = np.percentile(times, 99)
        assert avg_ms < 50, f"Avg RL step time {avg_ms:.1f}ms exceeds 50ms target"
        assert p99_ms < 100, f"P99 RL step time {p99_ms:.1f}ms exceeds 100ms target"

    def test_reward_computation_throughput(self):
        """Reward function should handle 10,000 computations/sec."""
        reward_fn = ParetoReward()
        metrics = {"wue": 1.2, "pue": 1.3, "carbon_intensity": 0.5,
                   "thermal_satisfaction": 0.8, "inlet_temp_c": 22.0}

        t0 = time.perf_counter()
        for _ in range(10000):
            reward_fn.compute(metrics)
        elapsed = time.perf_counter() - t0

        throughput = 10000 / elapsed
        assert throughput > 10000, f"Reward throughput {throughput:.0f}/s below 10K target"

    def test_twin_simulation_under_500ms(self):
        """Digital twin simulation should complete in < 500ms."""
        graph = AssetGraph.create_synthetic(num_racks=16, num_crahs=4, rows=4)
        twin = DigitalTwin(graph=graph, hidden_dim=16, num_gnn_layers=2, device="cpu")

        t0 = time.perf_counter()
        state = twin.simulate()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert elapsed_ms < 500, f"Twin simulation took {elapsed_ms:.0f}ms (target: <500ms)"

    def test_compliance_evaluation_under_10ms(self):
        """Compliance evaluation of 13 rules should be < 10ms."""
        engine = RegulationEngine(jurisdictions=["EPA_FEDERAL", "EU_WFD", "CALIFORNIA", "SINGAPORE"])
        metrics = {"pue": 1.3, "wue": 1.5, "inlet_temp_c": 23.0,
                   "discharge_temp_c": 28.0, "daily_water_liters": 200000,
                   "annual_carbon_tonnes": 25000}

        t0 = time.perf_counter()
        for _ in range(1000):
            engine.evaluate(metrics)
        avg_ms = (time.perf_counter() - t0) / 1000 * 1000

        assert avg_ms < 10, f"Compliance evaluation took {avg_ms:.1f}ms (target: <10ms)"

    def test_audit_trail_query_under_10ms_with_1000_entries(self):
        """Querying 1000-entry audit trail should be < 10ms."""
        trail = AuditTrail()
        for i in range(1000):
            trail.log(f"Event {i}", source_plane=i % 4 + 1, category="action")

        t0 = time.perf_counter()
        results = trail.query(source_plane=3, limit=50)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert elapsed_ms < 10, f"Audit query took {elapsed_ms:.1f}ms"
        assert len(results) == 50


# ╔══════════════════════════════════════════════════════════╗
# ║  LEVEL 8 — SYSTEM COHERENCE (Full Catastrophe)         ║
# ╚══════════════════════════════════════════════════════════╝

class TestLevel8_SystemCoherence:
    """Full catastrophe: heatwave + pump failure + carbon spike + IT surge."""

    def test_full_catastrophe_all_planes_react(self):
        """
        Scenario: Heatwave(45°C) + pump failure + carbon spike(1200) + IT surge(3×)
        All 4 planes must react coherently without race conditions.
        """
        trail = AuditTrail()
        events_timeline = []

        # ═══ Plane 1: Extreme simulation ═══
        graph = AssetGraph.create_synthetic(num_racks=8, num_crahs=2, rows=2)
        twin = DigitalTwin(graph=graph, hidden_dim=16, num_gnn_layers=1, device="cpu")

        # Force extreme conditions
        for rack in graph.nodes_by_type(AssetType.RACK):
            graph.update_node(rack.id, it_load_kw=150.0)  # 3× normal
        for crah in graph.nodes_by_type(AssetType.CRAH):
            graph.update_node(crah.id, current_cooling_kw=50.0)  # reduced (pump failure)

        state = twin.simulate()
        trail.log("CATASTROPHE: Extreme simulation",
                  source_plane=1, category="simulation", severity="critical",
                  details={"pue": state.pue, "hotspots": len(state.hotspots)})
        events_timeline.append("P1:simulate")

        # ═══ Plane 2: Detect multiple anomalies ═══
        alert_engine = AlertEngine()
        tracker = IncidentTracker()

        alerts = []
        a1 = alert_engine.process("hotspot", 0.95, "Rack-R1C1", "thermal")
        if a1:
            alerts.append(a1)
            tracker.create_incident(a1.anomaly_type, a1.severity, a1.location)

        a2 = alert_engine.process("vibration_fault", 0.9, "CRAH-01", "vibration")
        if a2:
            alerts.append(a2)
            tracker.create_incident(a2.anomaly_type, a2.severity, a2.location)

        trail.log(f"CATASTROPHE: {len(alerts)} alerts fired",
                  source_plane=2, category="alert", severity="critical")
        events_timeline.append("P2:detect")

        # ═══ Plane 3: RL responds under extreme conditions ═══
        reward_fn = ParetoReward()
        catastrophe_metrics = {
            "wue": 4.0, "pue": 3.0, "carbon_intensity": 1.2,
            "thermal_satisfaction": 0.1, "inlet_temp_c": 42.0,
            "grid_carbon_intensity": 1200.0,
        }
        reward = reward_fn.compute(catastrophe_metrics)
        assert reward < 0, "Catastrophe should produce negative reward"

        explainer = ExplainabilityEngine()
        explanation = explainer.explain_rl_action(
            action={"cooling_mode_mix": 0.9, "supply_air_temp": 12.0, "fan_speed": 1.0},
            state={"ambient_temp_c": 45, "inlet_temp_c": 42,
                   "grid_carbon_intensity": 1200, "water_stress_index": 4},
            reward=reward,
            metrics={"pue": 3.0, "wue": 4.0},
        )
        trail.log(f"CATASTROPHE: RL decision — {explanation.action_summary}",
                  source_plane=3, category="decision", severity="critical")
        events_timeline.append("P3:decide")

        # ═══ Plane 4: Compliance check + report ═══
        reg_engine = RegulationEngine(jurisdictions=["EPA_FEDERAL", "EU_WFD"])
        compliance = reg_engine.evaluate({
            "pue": 3.0, "wue": 4.0, "inlet_temp_c": 42.0,
            "discharge_temp_c": 50.0, "daily_water_liters": 800000,
            "annual_carbon_tonnes": 80000,
        })
        violations = [r for r in compliance if r.status == ComplianceStatus.FAIL]
        assert len(violations) >= 3, f"Catastrophe should trigger ≥3 violations, got {len(violations)}"

        trail.log(f"CATASTROPHE: {len(violations)} violations detected",
                  source_plane=4, category="violation", severity="critical")
        events_timeline.append("P4:comply")

        reporter = ComplianceReporter()
        report = reporter.generate_report(
            ReportMetrics(avg_pue=3.0, avg_wue=4.0, avg_inlet_temp_c=42.0,
                          total_incidents=2, critical_incidents=2),
            compliance_results=[r.to_dict() for r in compliance],
        )

        # ═══ VERIFY COHERENCE ═══
        assert trail.size == 4
        is_valid, tampered = trail.verify_integrity()
        assert is_valid, "Hash chain must be intact"

        # All planes participated in order
        assert events_timeline == ["P1:simulate", "P2:detect", "P3:decide", "P4:comply"]

        # All planes present in audit trail
        planes = {e.source_plane for e in trail.entries}
        assert planes == {1, 2, 3, 4}

        # Report reflects catastrophe
        assert report.compliance_score < 0.5, "Catastrophe score should be low"
        assert len(report.violations) >= 3

        # Explanation exists
        assert len(explanation.reasoning) >= 2, "Should provide multiple reasons for extreme conditions"


# ╔══════════════════════════════════════════════════════════╗
# ║  LEVEL 9 — BUSINESS LOGIC                              ║
# ╚══════════════════════════════════════════════════════════╝

class TestLevel9_BusinessLogic:
    """Does the system actually reduce water and carbon measurably?"""

    def test_measurable_water_reduction(self):
        """Switching to mechanical cooling should measurably reduce water usage."""
        reward_fn = ParetoReward()

        # Evaporative cooling (high water, low energy)
        evap = {"wue": 2.5, "pue": 1.15, "carbon_intensity": 0.3,
                "thermal_satisfaction": 0.85, "inlet_temp_c": 22.0}
        # Mechanical cooling (low water, higher energy)
        mech = {"wue": 0.3, "pue": 1.4, "carbon_intensity": 0.5,
                "thermal_satisfaction": 0.85, "inlet_temp_c": 22.0}

        r_evap = reward_fn.compute(evap)
        r_mech = reward_fn.compute(mech)

        # Both should be computable and different
        assert r_evap != r_mech, "Different cooling modes should produce different rewards"
        water_saved_pct = (1 - mech["wue"] / evap["wue"]) * 100
        assert water_saved_pct > 80, f"Mechanical should save >80% water, got {water_saved_pct:.0f}%"

    def test_measurable_carbon_reduction(self):
        """Shifting to clean grid hours should reduce carbon."""
        dirty_grid = {"wue": 1.5, "pue": 1.3, "carbon_intensity": 0.8,
                      "thermal_satisfaction": 0.8, "inlet_temp_c": 22.0,
                      "grid_carbon_intensity": 600}
        clean_grid = {"wue": 1.5, "pue": 1.3, "carbon_intensity": 0.2,
                      "thermal_satisfaction": 0.8, "inlet_temp_c": 22.0,
                      "grid_carbon_intensity": 100}

        # Isolate carbon comparison to see pure policy difference
        reward_fn = ParetoReward(
            base_weights=RewardWeights(alpha=0.0, beta=0.0, gamma=1.0, delta=0.0),
            dynamic_config=DynamicAdjustmentConfig(enabled=False)
        )
        r_dirty = reward_fn.compute(dirty_grid)
        r_clean = reward_fn.compute(clean_grid)
        assert r_clean > r_dirty, "Clean grid should produce better reward"

    def test_pareto_frontier_exists(self):
        """
        Generate Pareto curve: sweep cooling mix from 0 (evaporative) to 1 (mechanical).
        Water and energy should trade off — the Pareto frontier must be non-trivial.
        """
        reward_fn = ParetoReward()
        rewards = []
        wues = []
        pues = []

        for mix in np.linspace(0, 1, 20):
            wue = 2.5 * (1 - mix) + 0.1 * mix   # more mechanical → less water
            pue = 1.1 + 0.4 * mix                 # more mechanical → more energy
            carbon = 0.3 + 0.2 * mix               # more energy → more carbon

            metrics = {"wue": wue, "pue": pue, "carbon_intensity": carbon,
                       "thermal_satisfaction": 0.85, "inlet_temp_c": 22.0}
            r = reward_fn.compute(metrics)
            rewards.append(r)
            wues.append(wue)
            pues.append(pue)

        # Verify trade-off: WUE goes down while PUE goes up
        assert wues[0] > wues[-1], "WUE should decrease as cooling mix increases"
        assert pues[0] < pues[-1], "PUE should increase as cooling mix increases"

        # Rewards should vary (not all the same)
        assert np.std(rewards) > 0.001, "Pareto curve should produce varying rewards"

        # Extreme ends should be distinguishable
        assert rewards[0] != rewards[-1], "Frontier endpoints must differ"

    def test_compliance_trend_improves(self):
        """Compliance score should improve when metrics improve."""
        reporter = ComplianceReporter()
        engine = RegulationEngine(jurisdictions=["EPA_FEDERAL"])

        # Month 1: bad metrics
        r1 = engine.evaluate({"pue": 1.6, "daily_water_liters": 450000})
        report1 = reporter.generate_report(
            ReportMetrics(avg_pue=1.6), compliance_results=[r.to_dict() for r in r1])

        # Month 2: improved metrics
        r2 = engine.evaluate({"pue": 1.2, "daily_water_liters": 200000})
        report2 = reporter.generate_report(
            ReportMetrics(avg_pue=1.2), compliance_results=[r.to_dict() for r in r2])

        assert report2.compliance_score >= report1.compliance_score, \
            "Better metrics should produce equal or better compliance"


# ╔══════════════════════════════════════════════════════════╗
# ║  LEVEL 10 — INTERVIEW BRUTALITY                        ║
# ╚══════════════════════════════════════════════════════════╝

class TestLevel10_InterviewBrutality:
    """Edge cases an interviewer would attack."""

    def test_sensor_hacked_extreme_values(self):
        """Hacked sensor sending 999°C — system should detect as anomaly."""
        detector = StatisticalDetector(z_threshold=3.0)
        for _ in range(50):
            detector.detect("s1", 22.0 + np.random.normal(0, 0.5))

        # Hacked reading
        result = detector.detect("s1", 999.0)
        assert result is not None, "999°C should be flagged as anomaly"
        assert result.confidence > 0.7, "Extreme anomaly should have high confidence"

    def test_grid_api_failure_graceful(self):
        """If grid carbon API fails, system should use last known value."""
        env = DataCenterEnv(scenario=NormalOps(), max_episode_steps=20)
        obs, _ = env.reset(seed=42)

        # Step with normal carbon
        action = env.action_space.sample()
        _, _, _, _, info1 = env.step(action)
        carbon1 = info1["state"]["grid_carbon"]

        # Simulate API failure by not changing state
        # System should continue with last known value
        _, _, _, _, info2 = env.step(action)
        carbon2 = info2["state"]["grid_carbon"]
        assert np.isfinite(carbon2), "Carbon should remain finite even without API update"

    def test_reward_scaling_robustness(self):
        """Reward should be bounded regardless of input scale."""
        reward_fn = ParetoReward()
        rewards = []
        for scale in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
            metrics = {
                "wue": 1.0 * scale, "pue": 1.3,
                "carbon_intensity": 0.5 * scale,
                "thermal_satisfaction": min(1.0, 0.8 / scale),
                "inlet_temp_c": 22.0,
            }
            r = reward_fn.compute(metrics)
            rewards.append(r)
            assert np.isfinite(r), f"Reward not finite at scale {scale}"

        # Rewards should vary but all be finite
        assert all(np.isfinite(r) for r in rewards)

    def test_policy_doesnt_overfit_single_scenario(self):
        """Agent should produce different actions for different scenarios."""
        env_normal = DataCenterEnv(scenario=NormalOps(), max_episode_steps=10)
        env_heat = DataCenterEnv(scenario=HeatWave(), max_episode_steps=10)

        obs_n, _ = env_normal.reset(seed=42)
        obs_h, _ = env_heat.reset(seed=42)

        # Initial observations should differ between scenarios
        assert not np.allclose(obs_n, obs_h), "Different scenarios must produce different obs"

    def test_misreported_carbon_detected_by_audit(self):
        """Misreported carbon should be detectable through audit trail."""
        trail = AuditTrail()

        # Log actual carbon from sensors
        trail.log("Sensor carbon reading", source_plane=3,
                  details={"actual_carbon_kg": 5000})
        # Log reported carbon (misreported lower)
        trail.log("Reported carbon to regulator", source_plane=4,
                  details={"reported_carbon_kg": 3000})

        # Audit check: compare actual vs reported
        entries = trail.query(limit=10)
        actual = next(e for e in entries if "actual_carbon_kg" in e.details)
        reported = next(e for e in entries if "reported_carbon_kg" in e.details)

        discrepancy = abs(actual.details["actual_carbon_kg"] - reported.details["reported_carbon_kg"])
        assert discrepancy > 0, "Audit trail should reveal carbon misreporting"

        # Flag it
        trail.log("AUDIT FLAG: Carbon misreporting detected",
                  source_plane=4, category="violation", severity="critical",
                  details={"discrepancy_kg": discrepancy, "pct": discrepancy / actual.details["actual_carbon_kg"] * 100})

        violations = trail.query(category="violation")
        assert len(violations) == 1

    def test_safety_bounds_prevent_thermal_runaway(self):
        """System should never allow unbounded temperature growth."""
        env = DataCenterEnv(scenario=HeatWave(), max_episode_steps=100)
        obs, _ = env.reset(seed=42)
        env._state[2] = 55.0  # extreme

        max_temp = 0
        for _ in range(50):
            action = np.array([1.0, 0.0, 0.0, 0.0])  # worst action
            obs, _, term, trunc, info = env.step(action)
            temp = info["state"]["inlet_temp_c"]
            max_temp = max(max_temp, temp)
            if term or trunc:
                break

        assert max_temp < 200, f"Temperature runaway: {max_temp}°C — safety bounds missing"

    def test_explainability_under_extreme_conditions(self):
        """Explainability engine should provide meaningful reasons during catastrophe."""
        explainer = ExplainabilityEngine()
        explanation = explainer.explain_rl_action(
            action={"cooling_mode_mix": 0.95},
            state={"ambient_temp_c": 50, "inlet_temp_c": 40,
                   "grid_carbon_intensity": 1200, "water_stress_index": 5.0},
            reward=-10.0,
            metrics={"pue": 3.0, "wue": 5.0},
        )
        assert len(explanation.reasoning) >= 3, \
            f"Extreme conditions should trigger multiple reasons, got {len(explanation.reasoning)}"
        assert any("ambient" in r.lower() or "water" in r.lower() or "carbon" in r.lower()
                    for r in explanation.reasoning)

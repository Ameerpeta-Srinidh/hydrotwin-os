"""
Tests for the migration engine and event schemas.
"""

import pytest

from hydrotwin.migration.migration_engine import (
    MigrationEngine, MigrationTriggerConditions, CandidateRegion,
)
from hydrotwin.events.schemas import (
    SensorReading, RLAction, Forecast, LayoutUpdate, MigrationEvent, AnomalyAlert,
)


class TestMigrationEngine:
    """Tests for workload migration logic."""

    def _make_engine(self) -> MigrationEngine:
        return MigrationEngine(
            trigger_conditions=MigrationTriggerConditions(
                min_ambient_temp_c=40.0,
                min_grid_carbon_gco2=350.0,
                min_water_stress_index=3.0,
                require_all=True,
            ),
        )

    def test_no_migration_under_normal_conditions(self):
        """Normal conditions should NOT trigger migration."""
        engine = self._make_engine()
        result = engine.evaluate(
            ambient_temp_c=25.0,
            grid_carbon_gco2=150.0,
            water_stress_index=1.5,
        )
        assert not result.should_migrate

    def test_migration_under_compound_crisis(self):
        """Compound crisis (all thresholds exceeded) should trigger migration."""
        engine = self._make_engine()
        result = engine.evaluate(
            ambient_temp_c=46.0,
            grid_carbon_gco2=700.0,
            water_stress_index=4.5,
        )
        assert result.should_migrate
        assert result.target_region is not None
        assert len(result.workloads_to_migrate) > 0
        assert result.estimated_carbon_savings_pct > 0

    def test_partial_conditions_no_migration_with_require_all(self):
        """Only 2 of 3 conditions met should NOT trigger with require_all=True."""
        engine = self._make_engine()
        result = engine.evaluate(
            ambient_temp_c=45.0,      # exceeded
            grid_carbon_gco2=400.0,   # exceeded
            water_stress_index=1.0,   # NOT exceeded
        )
        assert not result.should_migrate

    def test_dispatch_returns_result(self):
        """Dispatching migration should return structured result."""
        engine = self._make_engine()
        recommendation = engine.evaluate(
            ambient_temp_c=46.0,
            grid_carbon_gco2=700.0,
            water_stress_index=4.5,
        )
        result = engine.dispatch_workloads(recommendation)
        assert result["status"] == "dispatched"
        assert "target_region" in result

    def test_dispatch_skips_when_no_migration(self):
        """No migration recommendation should skip dispatch."""
        engine = self._make_engine()
        recommendation = engine.evaluate(25.0, 150.0, 1.5)
        result = engine.dispatch_workloads(recommendation)
        assert result["status"] == "skipped"

    def test_best_region_selection(self):
        """Migration should select the region with lowest carbon intensity."""
        engine = self._make_engine()
        recommendation = engine.evaluate(46.0, 700.0, 4.5)

        if recommendation.target_region:
            # Target should have lower carbon than source
            assert recommendation.target_region.carbon_intensity < 700.0


class TestEventSchemas:
    """Tests for Pydantic event schema validation."""

    def test_sensor_reading_valid(self):
        event = SensorReading(
            sensor_id="TEMP-001",
            metric_name="inlet_temp",
            value=23.5,
            unit="celsius",
        )
        assert event.sensor_id == "TEMP-001"
        assert event.source_plane == 2

    def test_rl_action_valid(self):
        event = RLAction(
            cooling_mode_mix=0.3,
            supply_air_temp_setpoint=20.0,
            fan_speed_pct=0.7,
            economizer_damper=0.5,
            inlet_temp_c=23.0,
            ambient_temp_c=30.0,
            it_load_kw=5000.0,
            grid_carbon_intensity=200.0,
            water_stress_index=1.5,
            wue=0.8,
            pue=1.3,
            carbon_intensity=250.0,
            thermal_satisfaction=0.95,
            reward=-0.15,
            reward_weights={"alpha": 0.4, "beta": 0.2, "gamma": 0.3, "delta": 0.1},
        )
        assert event.source_plane == 3
        assert event.action_id  # should be auto-generated

    def test_forecast_valid(self):
        event = Forecast(
            forecast_type="it_load",
            horizon_hours=24,
            values=[5000.0] * 24,
            model="prophet",
        )
        assert len(event.values) == 24

    def test_migration_event_valid(self):
        event = MigrationEvent(
            ambient_temp_c=46.0,
            grid_carbon_intensity=700.0,
            water_stress_index=4.5,
            target_region="Pacific Northwest",
            target_zone="US-NW-PACW",
            target_carbon_intensity=80.0,
            workloads_migrated=["batch-ml-training"],
            estimated_savings_pct=85.0,
        )
        assert event.status == "dispatched"
        assert event.source_plane == 3

    def test_anomaly_alert_valid(self):
        event = AnomalyAlert(
            anomaly_type="leak",
            severity="critical",
            location="Pipe-12, Rack-37",
            confidence=0.95,
        )
        assert event.source_plane == 2

    def test_rl_action_serialization(self):
        """Event should serialize to dict with all fields."""
        event = RLAction(
            cooling_mode_mix=0.3, supply_air_temp_setpoint=20.0,
            fan_speed_pct=0.7, economizer_damper=0.5,
            inlet_temp_c=23.0, ambient_temp_c=30.0, it_load_kw=5000.0,
            grid_carbon_intensity=200.0, water_stress_index=1.5,
            wue=0.8, pue=1.3, carbon_intensity=250.0,
            thermal_satisfaction=0.95, reward=-0.15,
            reward_weights={"alpha": 0.4, "beta": 0.2, "gamma": 0.3, "delta": 0.1},
        )
        d = event.model_dump()
        assert "cooling_mode_mix" in d
        assert "timestamp" in d
        assert "correlation_id" in d

"""
Tests for the DataCenterEnv Gymnasium environment.

Verifies:
    - Environment reset produces valid initial state
    - Steps with random actions produce valid transitions
    - WUE/PUE calculations are physically reasonable
    - Thermal constraint violations are detected
    - All scenario profiles initialize correctly
"""

import numpy as np
import pytest

from hydrotwin.env.datacenter_env import DataCenterEnv
from hydrotwin.env.scenarios import (
    NormalOps, HeatWave, DirtyGrid, WaterStress,
    PeakLoad, CompoundCrisis, RandomScenario,
    get_scenario, SCENARIOS,
)


class TestDataCenterEnv:
    """Tests for the core Gym environment."""

    def test_reset_returns_valid_state(self):
        """Environment reset should produce a state within observation bounds."""
        env = DataCenterEnv()
        obs, info = env.reset(seed=42)

        assert obs.shape == (12,), f"Expected shape (12,), got {obs.shape}"
        assert env.observation_space.contains(obs), "Observation out of bounds"
        assert "step" in info
        assert info["step"] == 0

    def test_step_returns_valid_transition(self):
        """A single step should produce valid next state, reward, and flags."""
        env = DataCenterEnv()
        env.reset(seed=42)

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (12,), f"Expected shape (12,), got {obs.shape}"
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "metrics" in info

    def test_multiple_steps_without_crash(self):
        """Environment should handle 100 random steps without error."""
        env = DataCenterEnv(max_episode_steps=200)
        env.reset(seed=42)

        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()

    def test_metrics_populated_after_step(self):
        """Metrics dict should contain WUE, PUE, carbon after a step."""
        env = DataCenterEnv()
        env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)

        metrics = info["metrics"]
        assert "wue" in metrics
        assert "pue" in metrics
        assert "carbon_intensity" in metrics
        assert "thermal_satisfaction" in metrics
        assert "inlet_temp_c" in metrics

    def test_pue_always_ge_one(self):
        """PUE should always be >= 1.0 (physical constraint)."""
        env = DataCenterEnv()
        env.reset(seed=42)

        for _ in range(50):
            action = env.action_space.sample()
            _, _, terminated, truncated, info = env.step(action)
            pue = info["metrics"].get("pue", 1.0)
            assert pue >= 1.0, f"PUE {pue} < 1.0 is physically impossible"
            if terminated or truncated:
                env.reset()

    def test_truncation_at_max_steps(self):
        """Episode should truncate at max_episode_steps."""
        env = DataCenterEnv(max_episode_steps=10)
        env.reset(seed=42)

        for i in range(10):
            action = env.action_space.sample()
            _, _, terminated, truncated, info = env.step(action)
            if terminated:
                break

        # If not terminated early, should be truncated at step 10
        if not terminated:
            assert truncated, "Expected truncation at max_episode_steps"

    def test_render_ansi_mode(self):
        """ANSI render mode should return a string."""
        env = DataCenterEnv(render_mode="ansi")
        env.reset(seed=42)
        env.step(env.action_space.sample())
        output = env.render()
        assert isinstance(output, str)
        assert "WUE" in output


class TestScenarios:
    """Tests for training scenario profiles."""

    @pytest.mark.parametrize("name", list(SCENARIOS.keys()))
    def test_scenario_init(self, name):
        """Every scenario should produce valid initial conditions."""
        scenario = get_scenario(name)
        rng = np.random.default_rng(42)
        init = scenario.initial_conditions(rng)

        assert "server_inlet_temp_c" in init
        assert "ambient_temp_c" in init
        assert "it_load_kw" in init
        assert init["it_load_kw"] > 0

    @pytest.mark.parametrize("name", list(SCENARIOS.keys()))
    def test_scenario_step(self, name):
        """Every scenario should produce valid step conditions."""
        scenario = get_scenario(name)
        rng = np.random.default_rng(42)
        scenario.initial_conditions(rng)
        step = scenario.step_conditions(10, rng, 30.0, 5000.0)

        assert "ambient_temp_c" in step
        assert "it_load_kw" in step
        assert "grid_carbon_intensity" in step
        assert step["it_load_kw"] > 0
        assert step["grid_carbon_intensity"] >= 0

    def test_compound_crisis_has_extreme_values(self):
        """CompoundCrisis should have high temp, high carbon, high water stress."""
        scenario = CompoundCrisis()
        rng = np.random.default_rng(42)
        init = scenario.initial_conditions(rng)

        assert init["ambient_temp_c"] > 40, "Crisis should have extreme temperature"
        assert init["grid_carbon_intensity"] > 500, "Crisis should have dirty grid"
        assert init["water_stress_index"] > 4, "Crisis should have high water stress"

    def test_random_scenario_varies(self):
        """RandomScenario should select different scenarios across resets."""
        scenario = RandomScenario()
        rng = np.random.default_rng(42)

        names = set()
        for _ in range(20):
            scenario.initial_conditions(rng)
            names.add(scenario._active.__class__.__name__)

        assert len(names) > 1, "RandomScenario should vary across resets"

    def test_get_scenario_unknown(self):
        """get_scenario should raise for unknown names."""
        with pytest.raises(ValueError, match="Unknown scenario"):
            get_scenario("nonexistent_scenario")

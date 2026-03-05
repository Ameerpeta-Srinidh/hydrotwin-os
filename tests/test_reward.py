"""
Tests for the Pareto reward function.

Verifies:
    - Reward computation with known inputs
    - Weight extremes (pure water, pure carbon, pure energy modes)
    - Dynamic weight adjustment shifts α/γ correctly
    - Thermal constraint sigmoid shape
    - Config loading
"""

import pytest

from hydrotwin.reward.pareto_reward import (
    ParetoReward, RewardWeights, DynamicWeightAdjuster, DynamicAdjustmentConfig,
)


class TestRewardWeights:
    """Tests for RewardWeights dataclass."""

    def test_defaults(self):
        w = RewardWeights()
        assert w.alpha == 0.4
        assert w.beta == 0.2
        assert w.gamma == 0.3
        assert w.delta == 0.1

    def test_to_dict(self):
        w = RewardWeights(alpha=1, beta=0, gamma=0, delta=0)
        d = w.to_dict()
        assert d == {"alpha": 1, "beta": 0, "gamma": 0, "delta": 0}


class TestDynamicWeightAdjuster:
    """Tests for dynamic weight adjustment based on grid carbon."""

    def test_clean_grid_increases_alpha(self):
        """When grid is clean, α should be at the high end (save water)."""
        adj = DynamicWeightAdjuster()
        base = RewardWeights()
        result = adj.adjust(base, grid_carbon_gco2=50.0)  # Very clean

        assert result.alpha > base.alpha, "Clean grid should increase water weight"
        assert result.gamma < base.gamma, "Clean grid should decrease carbon weight"

    def test_dirty_grid_increases_gamma(self):
        """When grid is dirty, γ should be at the high end (save carbon)."""
        adj = DynamicWeightAdjuster()
        base = RewardWeights()
        result = adj.adjust(base, grid_carbon_gco2=500.0)  # Very dirty

        assert result.gamma > base.gamma, "Dirty grid should increase carbon weight"
        assert result.alpha < base.alpha, "Dirty grid should decrease water weight"

    def test_disabled_returns_base(self):
        """When disabled, should return base weights unchanged."""
        cfg = DynamicAdjustmentConfig(enabled=False)
        adj = DynamicWeightAdjuster(cfg)
        base = RewardWeights(alpha=0.5, beta=0.2, gamma=0.2, delta=0.1)
        result = adj.adjust(base, grid_carbon_gco2=500.0)

        assert result.alpha == base.alpha
        assert result.gamma == base.gamma

    def test_mid_range_grid(self):
        """At mid-range carbon, weights should be in the middle of their ranges."""
        adj = DynamicWeightAdjuster()
        base = RewardWeights()
        result = adj.adjust(base, grid_carbon_gco2=250.0)  # Mid-range

        assert 0.2 < result.alpha < 0.6
        assert 0.1 < result.gamma < 0.5


class TestParetoReward:
    """Tests for the multi-objective reward function."""

    def test_zero_metrics_reward(self):
        """Reward should be computable with zero/minimal metrics."""
        reward_fn = ParetoReward()
        metrics = {
            "wue": 0.0,
            "pue": 1.0,
            "carbon_intensity": 0.0,
            "thermal_satisfaction": 1.0,
            "inlet_temp_c": 24.0,
        }
        reward = reward_fn.compute(metrics)
        assert isinstance(reward, float)

    def test_bad_metrics_give_negative_reward(self):
        """Poor operating conditions should produce strongly negative reward."""
        reward_fn = ParetoReward()
        metrics = {
            "wue": 3.0,           # bad
            "pue": 2.5,           # bad
            "carbon_intensity": 800.0,  # bad
            "thermal_satisfaction": 0.1,
            "inlet_temp_c": 38.0,  # too hot
        }
        reward = reward_fn.compute(metrics)
        assert reward < -0.5, f"Bad conditions should give very negative reward, got {reward}"

    def test_good_metrics_give_better_reward(self):
        """Good conditions should produce a higher reward than bad conditions."""
        reward_fn = ParetoReward()
        good = {
            "wue": 0.3, "pue": 1.1, "carbon_intensity": 50,
            "thermal_satisfaction": 1.0, "inlet_temp_c": 24.0,
        }
        bad = {
            "wue": 2.0, "pue": 1.8, "carbon_intensity": 500,
            "thermal_satisfaction": 0.2, "inlet_temp_c": 35.0,
        }
        assert reward_fn.compute(good) > reward_fn.compute(bad)

    def test_pure_water_mode(self):
        """With only α set, reward should only penalize WUE."""
        reward_fn = ParetoReward(base_weights=RewardWeights(alpha=1, beta=0, gamma=0, delta=0),
                                   dynamic_config=DynamicAdjustmentConfig(enabled=False))
        high_wue = {"wue": 3.0, "pue": 1.0, "carbon_intensity": 0, "inlet_temp_c": 24.0}
        low_wue = {"wue": 0.1, "pue": 1.0, "carbon_intensity": 0, "inlet_temp_c": 24.0}

        assert reward_fn.compute(low_wue) > reward_fn.compute(high_wue)

    def test_pure_carbon_mode(self):
        """With only γ set, reward should only penalize carbon intensity."""
        reward_fn = ParetoReward(base_weights=RewardWeights(alpha=0, beta=0, gamma=1, delta=0),
                                   dynamic_config=DynamicAdjustmentConfig(enabled=False))
        high_carbon = {"wue": 0, "pue": 1.0, "carbon_intensity": 800, "inlet_temp_c": 24.0}
        low_carbon = {"wue": 0, "pue": 1.0, "carbon_intensity": 50, "inlet_temp_c": 24.0}

        assert reward_fn.compute(low_carbon) > reward_fn.compute(high_carbon)

    def test_thermal_score_at_target(self):
        """Thermal score at target temperature should be high."""
        reward_fn = ParetoReward(target_temp_c=24.0)
        score = reward_fn._thermal_score(24.0)
        assert score > 0.5, f"Score at target should be high, got {score}"

    def test_thermal_score_at_limit(self):
        """Thermal score at ASHRAE limit should be low."""
        reward_fn = ParetoReward(target_temp_c=24.0, inlet_max_c=32.0)
        score = reward_fn._thermal_score(32.0)
        assert score < 0.5, f"Score at limit should be low, got {score}"

    def test_thermal_score_beyond_limit(self):
        """Thermal score beyond ASHRAE limit should be very negative."""
        reward_fn = ParetoReward(target_temp_c=24.0, inlet_max_c=32.0)
        score = reward_fn._thermal_score(40.0)
        assert score < 0.0, f"Score beyond limit should be negative, got {score}"

    def test_from_config(self):
        """Should create ParetoReward from config dict."""
        config = {
            "weights": {"alpha": 0.5, "beta": 0.1, "gamma": 0.3, "delta": 0.1},
            "dynamic_adjustment": {"enabled": True, "clean_grid_threshold_gco2": 80},
            "thermal_constraint": {"penalty_sharpness": 3.0, "target_temp_c": 22.0},
        }
        reward_fn = ParetoReward.from_config(config)
        assert reward_fn.base_weights.alpha == 0.5
        assert reward_fn.target_temp_c == 22.0

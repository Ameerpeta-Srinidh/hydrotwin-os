"""
HydroTwin OS — Plane 3: Pareto Reward Function

The core research contribution — a multi-objective reward function that navigates
the Water-Energy-Carbon Pareto frontier:

    R(t) = −α·WUE(t) − β·PUE(t) − γ·CarbonIntensity(t) + δ·ThermalConstraintSatisfaction(t)

Weights (α, β, γ, δ) can be:
    1. Statically configured by the operator
    2. Dynamically adjusted based on real-time grid carbon intensity:
       - Clean grid → increase α (prioritize water savings, since electric cooling is clean)
       - Dirty grid → increase γ (prioritize carbon reduction, prefer evaporative cooling)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RewardWeights:
    """Operator-configurable weights for the Pareto reward function."""

    alpha: float = 0.4   # WUE weight (water conservation)
    beta: float = 0.2    # PUE weight (energy efficiency)
    gamma: float = 0.3   # Carbon intensity weight
    delta: float = 0.1   # Thermal constraint satisfaction weight

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.alpha, self.beta, self.gamma, self.delta)

    def to_dict(self) -> dict[str, float]:
        return {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "delta": self.delta}


@dataclass
class DynamicAdjustmentConfig:
    """Configuration for dynamic weight adjustment based on grid carbon."""

    enabled: bool = True
    clean_grid_threshold_gco2: float = 100.0    # Below this → grid is clean
    dirty_grid_threshold_gco2: float = 400.0    # Above this → grid is dirty
    alpha_range: tuple[float, float] = (0.2, 0.6)
    gamma_range: tuple[float, float] = (0.1, 0.5)


class DynamicWeightAdjuster:
    """
    Shifts reward weights based on real-time grid carbon intensity.

    When the grid is clean (low carbon), electric cooling is low-carbon,
    so we increase α (water weight) to promote water savings.

    When the grid is dirty (high carbon), electric cooling is carbon-heavy,
    so we increase γ (carbon weight) to promote evaporative cooling.
    """

    def __init__(self, config: DynamicAdjustmentConfig | None = None):
        self.config = config or DynamicAdjustmentConfig()

    def adjust(self, base_weights: RewardWeights, grid_carbon_gco2: float) -> RewardWeights:
        """Adjust weights based on current grid carbon intensity."""
        if not self.config.enabled:
            return base_weights

        c = self.config
        # Linear interpolation: 0.0 at clean threshold, 1.0 at dirty threshold
        if c.dirty_grid_threshold_gco2 <= c.clean_grid_threshold_gco2:
            t = 0.5
        else:
            t = (grid_carbon_gco2 - c.clean_grid_threshold_gco2) / (
                c.dirty_grid_threshold_gco2 - c.clean_grid_threshold_gco2
            )
        t = max(0.0, min(1.0, t))  # clamp to [0, 1]

        # When grid is clean (t→0): high α, low γ → save water
        # When grid is dirty (t→1): low α, high γ → save carbon, use water
        adjusted_alpha = c.alpha_range[1] - t * (c.alpha_range[1] - c.alpha_range[0])
        adjusted_gamma = c.gamma_range[0] + t * (c.gamma_range[1] - c.gamma_range[0])

        return RewardWeights(
            alpha=adjusted_alpha,
            beta=base_weights.beta,
            gamma=adjusted_gamma,
            delta=base_weights.delta,
        )


class ParetoReward:
    """
    Multi-objective reward function for the Water-Energy-Carbon Nexus.

    R(t) = −α·WUE(t) − β·PUE(t) − γ·CarbonIntensity(t) + δ·ThermalSatisfaction(t)

    The reward is designed to be negative for poor operating points and approach
    zero for optimal ones, with the thermal satisfaction term providing a positive
    bonus for maintaining safe temperatures.
    """

    # Normalization constants (rough order of magnitude for each metric)
    _WUE_NORM = 2.0          # typical bad WUE
    _PUE_NORM = 2.0          # typical bad PUE
    _CARBON_NORM = 500.0     # typical carbon intensity gCO2/kWh

    def __init__(
        self,
        base_weights: RewardWeights | None = None,
        dynamic_config: DynamicAdjustmentConfig | None = None,
        penalty_sharpness: float = 2.0,
        target_temp_c: float = 24.0,
        inlet_min_c: float = 15.0,
        inlet_max_c: float = 32.0,
    ):
        self.base_weights = base_weights or RewardWeights()
        self.adjuster = DynamicWeightAdjuster(dynamic_config)
        self.penalty_sharpness = penalty_sharpness
        self.target_temp_c = target_temp_c
        self.inlet_min_c = inlet_min_c
        self.inlet_max_c = inlet_max_c

        # Track the active (possibly adjusted) weights
        self._active_weights = self.base_weights

    def compute(self, metrics: dict[str, float]) -> float:
        """
        Compute the reward from environment metrics.

        Args:
            metrics: Dict with keys 'wue', 'pue', 'carbon_intensity',
                     'thermal_satisfaction', 'inlet_temp_c', and optionally
                     'grid_carbon_intensity' for dynamic adjustment.

        Returns:
            Scalar reward value.
        """
        wue = metrics.get("wue", 0.0)
        pue = metrics.get("pue", 1.0)
        carbon = metrics.get("carbon_intensity", 0.0)
        thermal_sat = metrics.get("thermal_satisfaction", 1.0)
        inlet_temp = metrics.get("inlet_temp_c", self.target_temp_c)

        # Dynamic weight adjustment based on grid carbon
        grid_carbon = metrics.get("grid_carbon_intensity", 200.0)
        w = self.adjuster.adjust(self.base_weights, grid_carbon)
        self._active_weights = w

        # ── Normalized cost terms ──
        wue_cost = wue / self._WUE_NORM
        pue_cost = (pue - 1.0) / (self._PUE_NORM - 1.0)  # PUE ≥ 1.0 always
        carbon_cost = carbon / self._CARBON_NORM

        # ── Thermal constraint: sigmoid penalty for boundary violations ──
        thermal_bonus = self._thermal_score(inlet_temp)

        # ── Multi-objective reward ──
        reward = (
            -w.alpha * wue_cost
            - w.beta * pue_cost
            - w.gamma * carbon_cost
            + w.delta * thermal_bonus
        )

        return float(reward)

    def _thermal_score(self, inlet_temp_c: float) -> float:
        """
        Smooth thermal constraint satisfaction score.

        Returns 1.0 when temperature is at target, drops to 0 as temperature
        approaches ASHRAE limits, and goes strongly negative beyond limits.

        Uses a double-sigmoid to penalize both overheating and overcooling.
        """
        # Distance from target as fraction of allowable range
        if inlet_temp_c >= self.target_temp_c:
            # Upper violation — overheating
            margin = self.inlet_max_c - self.target_temp_c
            x = (inlet_temp_c - self.target_temp_c) / max(margin, 1.0)
        else:
            # Lower violation — overcooling
            margin = self.target_temp_c - self.inlet_min_c
            x = (self.target_temp_c - inlet_temp_c) / max(margin, 1.0)

        # Sigmoid: 1 at x=0, ~0 at x=1, negative beyond
        score = 1.0 - 2.0 / (1.0 + math.exp(-self.penalty_sharpness * (x - 0.8)))
        return score

    @property
    def active_weights(self) -> RewardWeights:
        """Return the currently active (possibly dynamically adjusted) weights."""
        return self._active_weights

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ParetoReward:
        """Create a ParetoReward from a config dict (typically from YAML)."""
        weights_cfg = config.get("weights", {})
        dynamic_cfg = config.get("dynamic_adjustment", {})
        thermal_cfg = config.get("thermal_constraint", {})

        base_weights = RewardWeights(
            alpha=weights_cfg.get("alpha", 0.4),
            beta=weights_cfg.get("beta", 0.2),
            gamma=weights_cfg.get("gamma", 0.3),
            delta=weights_cfg.get("delta", 0.1),
        )

        dynamic_config = DynamicAdjustmentConfig(
            enabled=dynamic_cfg.get("enabled", True),
            clean_grid_threshold_gco2=dynamic_cfg.get("clean_grid_threshold_gco2", 100.0),
            dirty_grid_threshold_gco2=dynamic_cfg.get("dirty_grid_threshold_gco2", 400.0),
            alpha_range=tuple(dynamic_cfg.get("alpha_range", [0.2, 0.6])),
            gamma_range=tuple(dynamic_cfg.get("gamma_range", [0.1, 0.5])),
        )

        return cls(
            base_weights=base_weights,
            dynamic_config=dynamic_config,
            penalty_sharpness=thermal_cfg.get("penalty_sharpness", 2.0),
            target_temp_c=thermal_cfg.get("target_temp_c", 24.0),
        )

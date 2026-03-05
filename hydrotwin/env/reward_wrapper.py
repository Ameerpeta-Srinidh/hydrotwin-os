"""
HydroTwin OS — Plane 3: Reward Wrapper

A Gymnasium wrapper that injects the ParetoReward function into the environment's
step method, replacing the default 0.0 reward with the computed multi-objective reward.
This keeps the environment and reward function cleanly separated.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from typing import Any

from hydrotwin.reward.pareto_reward import ParetoReward


class ParetoRewardWrapper(gym.Wrapper):
    """
    Wraps a DataCenterEnv to inject Pareto reward computation.

    The base environment returns reward=0.0 and provides metrics in the info dict.
    This wrapper reads those metrics and computes the actual reward via ParetoReward.
    """

    def __init__(self, env: gym.Env, reward_fn: ParetoReward | None = None):
        super().__init__(env)
        self.reward_fn = reward_fn or ParetoReward()
        self._episode_rewards: list[float] = []
        self._episode_metrics: list[dict[str, float]] = []

    def reset(self, **kwargs) -> tuple[np.ndarray, dict[str, Any]]:
        self._episode_rewards = []
        self._episode_metrics = []
        return self.env.reset(**kwargs)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, _, terminated, truncated, info = self.env.step(action)

        # Compute multi-objective reward from environment metrics
        metrics = info.get("metrics", {})
        reward = self.reward_fn.compute(metrics)

        # Track episode stats
        self._episode_rewards.append(reward)
        self._episode_metrics.append(metrics)

        # Enrich info with reward details
        info["reward_breakdown"] = {
            "total": reward,
            "weights": self.reward_fn.active_weights.to_dict(),
        }

        if terminated or truncated:
            info["episode_summary"] = self._compute_episode_summary()

        return obs, reward, terminated, truncated, info

    def _compute_episode_summary(self) -> dict[str, float]:
        """Compute episode-level aggregate metrics."""
        if not self._episode_metrics:
            return {}

        keys = ["wue", "pue", "carbon_intensity", "thermal_satisfaction",
                "water_consumed_liters", "carbon_emissions_g"]
        summary: dict[str, float] = {}
        for key in keys:
            vals = [m.get(key, 0.0) for m in self._episode_metrics]
            summary[f"mean_{key}"] = float(np.mean(vals))
            summary[f"max_{key}"] = float(np.max(vals))
            summary[f"min_{key}"] = float(np.min(vals))

        summary["total_reward"] = float(np.sum(self._episode_rewards))
        summary["mean_reward"] = float(np.mean(self._episode_rewards))
        summary["total_water_liters"] = sum(m.get("water_consumed_liters", 0.0) for m in self._episode_metrics)
        summary["total_carbon_g"] = sum(m.get("carbon_emissions_g", 0.0) for m in self._episode_metrics)

        return summary

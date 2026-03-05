"""
HydroTwin OS — Plane 3: SAC Agent

Wraps Stable-Baselines3 SAC with custom configuration tailored for the
data center cooling control problem. Provides a clean interface for
training, evaluation, prediction, and checkpoint management.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.logger import configure as configure_sb3_logger

from hydrotwin.env.datacenter_env import DataCenterEnv
from hydrotwin.env.reward_wrapper import ParetoRewardWrapper
from hydrotwin.env.scenarios import RandomScenario, get_scenario
from hydrotwin.reward.pareto_reward import ParetoReward

logger = logging.getLogger(__name__)


class NexusAgent:
    """
    The Carbon Nexus RL agent — wraps SAC for Water-Energy-Carbon optimization.

    Usage:
        agent = NexusAgent.from_config(config)
        agent.train(total_timesteps=500_000)
        action = agent.predict(observation)
        agent.save("checkpoints/nexus_v1")
    """

    def __init__(
        self,
        env: gym.Env,
        eval_env: gym.Env | None = None,
        learning_rate: float = 3e-4,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        ent_coef: str | float = "auto",
        train_freq: int = 1,
        gradient_steps: int = 1,
        net_arch: list[int] | None = None,
        tensorboard_log: str | None = None,
        checkpoint_dir: str | None = None,
        eval_freq: int = 5000,
        n_eval_episodes: int = 10,
        device: str = "auto",
    ):
        self.env = env
        self.eval_env = eval_env
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

        policy_kwargs = {}
        if net_arch:
            policy_kwargs["net_arch"] = net_arch

        self.model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            ent_coef=ent_coef,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            policy_kwargs=policy_kwargs if policy_kwargs else None,
            tensorboard_log=tensorboard_log,
            device=device,
            verbose=1,
        )

        logger.info(
            f"NexusAgent initialized | LR={learning_rate} | Buffer={buffer_size} | "
            f"Arch={net_arch or 'default'} | Device={device}"
        )

    def train(self, total_timesteps: int = 500_000) -> None:
        """Train the SAC agent."""
        callbacks = self._build_callbacks()
        logger.info(f"Starting training for {total_timesteps} timesteps...")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )

        # Save final model
        final_path = self.checkpoint_dir / "nexus_final"
        self.model.save(str(final_path))
        logger.info(f"Training complete. Final model saved to {final_path}")

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Predict the optimal cooling action for a given state."""
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return action, {
            "action_names": [
                "cooling_mode_mix",
                "supply_air_temp_setpoint",
                "fan_speed_pct",
                "economizer_damper",
            ],
            "action_values": action.tolist() if isinstance(action, np.ndarray) else action,
        }

    def evaluate(self, n_episodes: int = 10) -> dict[str, float]:
        """Evaluate current policy over n episodes, return aggregate metrics."""
        eval_env = self.eval_env or self.env
        episode_rewards = []
        episode_summaries = []

        for ep in range(n_episodes):
            obs, info = eval_env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                action, _ = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated

            episode_rewards.append(episode_reward)
            if "episode_summary" in info:
                episode_summaries.append(info["episode_summary"])

        results = {
            "mean_episode_reward": float(np.mean(episode_rewards)),
            "std_episode_reward": float(np.std(episode_rewards)),
            "min_episode_reward": float(np.min(episode_rewards)),
            "max_episode_reward": float(np.max(episode_rewards)),
        }

        # Aggregate episode summaries
        if episode_summaries:
            for key in episode_summaries[0]:
                vals = [s.get(key, 0.0) for s in episode_summaries]
                results[f"eval_{key}"] = float(np.mean(vals))

        return results

    def save(self, path: str | Path) -> None:
        """Save the trained model."""
        self.model.save(str(path))
        logger.info(f"Model saved to {path}")

    def load(self, path: str | Path) -> None:
        """Load a trained model."""
        self.model = SAC.load(str(path), env=self.env)
        logger.info(f"Model loaded from {path}")

    def _build_callbacks(self) -> list[BaseCallback]:
        """Build training callbacks for checkpointing and evaluation."""
        callbacks: list[BaseCallback] = []

        # Checkpoint callback
        callbacks.append(
            CheckpointCallback(
                save_freq=self.eval_freq,
                save_path=str(self.checkpoint_dir),
                name_prefix="nexus",
            )
        )

        # Eval callback (if we have a separate eval env)
        if self.eval_env is not None:
            callbacks.append(
                EvalCallback(
                    self.eval_env,
                    best_model_save_path=str(self.checkpoint_dir / "best"),
                    log_path=str(self.checkpoint_dir / "eval_logs"),
                    eval_freq=self.eval_freq,
                    n_eval_episodes=self.n_eval_episodes,
                    deterministic=True,
                )
            )

        return callbacks

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> NexusAgent:
        """Create a NexusAgent from the full application config."""
        agent_cfg = config.get("agent", {})
        reward_cfg = config.get("reward", {})
        facility_cfg = config.get("facility", {})
        hp = agent_cfg.get("hyperparameters", {})
        net_cfg = agent_cfg.get("network", {})
        train_cfg = agent_cfg.get("training", {})

        # Build reward function
        reward_fn = ParetoReward.from_config(reward_cfg)

        # Build training environment (random scenarios)
        train_env = ParetoRewardWrapper(
            DataCenterEnv(scenario=RandomScenario(), facility_config=facility_cfg),
            reward_fn=reward_fn,
        )

        # Build eval environment (normal ops for consistent evaluation)
        eval_env = ParetoRewardWrapper(
            DataCenterEnv(scenario=get_scenario("normal_ops"), facility_config=facility_cfg),
            reward_fn=reward_fn,
        )

        return cls(
            env=train_env,
            eval_env=eval_env,
            learning_rate=hp.get("learning_rate", 3e-4),
            buffer_size=hp.get("buffer_size", 1_000_000),
            batch_size=hp.get("batch_size", 256),
            tau=hp.get("tau", 0.005),
            gamma=hp.get("gamma", 0.99),
            ent_coef=hp.get("ent_coef", "auto"),
            train_freq=hp.get("train_freq", 1),
            gradient_steps=hp.get("gradient_steps", 1),
            net_arch=net_cfg.get("net_arch", [256, 256]),
            tensorboard_log=train_cfg.get("tensorboard_log"),
            checkpoint_dir=train_cfg.get("checkpoint_dir", "checkpoints"),
            eval_freq=train_cfg.get("eval_freq", 5000),
            n_eval_episodes=train_cfg.get("n_eval_episodes", 10),
        )

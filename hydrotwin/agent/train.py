"""
HydroTwin OS — Plane 3: Training Script

Entry point for training the Carbon Nexus RL agent.
Supports scenario selection, configurable timesteps, and TensorBoard logging.

Usage:
    python -m hydrotwin.agent.train --timesteps 500000 --scenario random
    python -m hydrotwin.agent.train --timesteps 10000 --scenario heat_wave --config config/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from hydrotwin.config import load_config
from hydrotwin.agent.sac_agent import NexusAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train the Carbon Nexus Agent")
    parser.add_argument(
        "--timesteps", type=int, default=500_000,
        help="Total training timesteps (default: 500000)",
    )
    parser.add_argument(
        "--scenario", type=str, default="random",
        help="Training scenario: normal_ops, heat_wave, dirty_grid, water_stress, peak_load, compound_crisis, random",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML (default: config/default.yaml)",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training, only evaluate a loaded model",
    )
    parser.add_argument(
        "--load-model", type=str, default=None,
        help="Path to a saved model to load (for eval or continued training)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    logger.info(f"Config loaded | Scenario: {args.scenario} | Timesteps: {args.timesteps}")

    # Create agent
    agent = NexusAgent.from_config(config)

    # Load existing model if specified
    if args.load_model:
        agent.load(args.load_model)
        logger.info(f"Loaded model from {args.load_model}")

    if args.eval_only:
        # Evaluate only
        logger.info("Running evaluation...")
        results = agent.evaluate(n_episodes=20)
        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 60)
        for key, val in sorted(results.items()):
            logger.info(f"  {key:40s} = {val:10.4f}")
    else:
        # Train
        logger.info(f"Starting training for {args.timesteps} timesteps...")
        agent.train(total_timesteps=args.timesteps)

        # Evaluate after training
        logger.info("Post-training evaluation...")
        results = agent.evaluate(n_episodes=20)
        logger.info("=" * 60)
        logger.info("POST-TRAINING EVALUATION")
        logger.info("=" * 60)
        for key, val in sorted(results.items()):
            logger.info(f"  {key:40s} = {val:10.4f}")

    logger.info("Done.")


if __name__ == "__main__":
    main()

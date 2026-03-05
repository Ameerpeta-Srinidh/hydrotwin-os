"""
HydroTwin OS — Plane 3: Main Orchestrator

Entry point for the deployed Carbon Nexus Agent. Initializes all subsystems,
loads the trained RL agent, starts Kafka consumers, and launches the
FastAPI dashboard server.

Usage:
    python -m hydrotwin.main
    python -m hydrotwin.main --config config/default.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from typing import Any

import numpy as np
import uvicorn

from hydrotwin.config import load_config
from hydrotwin.agent.sac_agent import NexusAgent
from hydrotwin.api_clients.electricity_maps import ElectricityMapsClient
from hydrotwin.api_clients.noaa_weather import NOAAWeatherClient
from hydrotwin.api_clients.wri_aqueduct import WRIAqueductClient
from hydrotwin.dashboard.api import DashboardState, create_dashboard_app
from hydrotwin.env.datacenter_env import DataCenterEnv
from hydrotwin.env.reward_wrapper import ParetoRewardWrapper
from hydrotwin.env.scenarios import NormalOps
from hydrotwin.events.kafka_consumer import NexusKafkaConsumer
from hydrotwin.events.kafka_producer import NexusKafkaProducer
from hydrotwin.events.schemas import RLAction
from hydrotwin.migration.migration_engine import MigrationEngine
from hydrotwin.reward.pareto_reward import ParetoReward

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class NexusOrchestrator:
    """
    Main runtime orchestrator for Plane 3.

    Coordinates the RL agent, API clients, event system, migration engine,
    and dashboard into a single coherent runtime.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._running = False

        # ── Initialize subsystems ──
        logger.info("=" * 60)
        logger.info("  HydroTwin OS — Plane 3: Carbon Nexus Agent")
        logger.info("=" * 60)

        # Reward function
        self.reward_fn = ParetoReward.from_config(config.get("reward", {}))
        logger.info("✓ Reward function initialized")

        # Environment (for live inference)
        facility_cfg = config.get("facility", {})
        self.env = ParetoRewardWrapper(
            DataCenterEnv(scenario=NormalOps(), facility_config=facility_cfg),
            reward_fn=self.reward_fn,
        )
        logger.info("✓ Environment initialized")

        # RL Agent
        self.agent = NexusAgent.from_config(config)
        logger.info("✓ RL Agent initialized")

        # API Clients
        self.electricity_client = ElectricityMapsClient.from_config(config)
        self.weather_client = NOAAWeatherClient.from_config(config)
        self.water_client = WRIAqueductClient.from_config(config)
        logger.info("✓ API clients initialized")

        # Migration Engine
        self.migration_engine = MigrationEngine.from_config(config)
        logger.info("✓ Migration engine initialized")

        # Kafka
        self.producer = NexusKafkaProducer.from_config(config)
        self.consumer = NexusKafkaConsumer.from_config(config)
        logger.info("✓ Kafka producer/consumer initialized")

        # Dashboard
        self.dashboard_state = DashboardState()
        self.dashboard_state.status = "running"
        self.dashboard_app = create_dashboard_app(
            state=self.dashboard_state,
            cors_origins=config.get("dashboard", {}).get("cors_origins"),
        )
        logger.info("✓ Dashboard API initialized")

        logger.info("=" * 60)
        logger.info("  All subsystems online. Ready to operate.")
        logger.info("=" * 60)

    def run_control_loop(self, steps: int = 1440) -> None:
        """
        Run the RL control loop for a specified number of steps.

        In production, this would run indefinitely, receiving real sensor
        data via Kafka. Here it runs simulation steps for demonstration.
        """
        logger.info(f"Starting control loop for {steps} steps...")
        obs, info = self.env.reset()

        for step in range(steps):
            # Predict action
            action, action_info = self.agent.predict(obs)

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            metrics = info.get("metrics", {})

            # Update dashboard state
            self.dashboard_state.update(
                action=action_info,
                metrics=metrics,
                weights=self.reward_fn.active_weights.to_dict(),
            )

            # Check for migration trigger
            if step % 60 == 0:  # check every simulated hour
                recommendation = self.migration_engine.evaluate(
                    ambient_temp_c=float(obs[2]),
                    grid_carbon_gco2=float(obs[7]),
                    water_stress_index=float(obs[8]),
                )
                if recommendation.should_migrate:
                    logger.warning(f"MIGRATION TRIGGERED: {recommendation.reason}")
                    result = self.migration_engine.dispatch_workloads(recommendation)
                    logger.info(f"Migration result: {result}")

            # Publish RL action to Kafka
            if step % 10 == 0:
                action_event = RLAction(
                    cooling_mode_mix=float(action[0]),
                    supply_air_temp_setpoint=float(action[1]),
                    fan_speed_pct=float(action[2]),
                    economizer_damper=float(action[3]),
                    inlet_temp_c=metrics.get("inlet_temp_c", 0),
                    ambient_temp_c=float(obs[2]),
                    it_load_kw=float(obs[5]),
                    grid_carbon_intensity=float(obs[7]),
                    water_stress_index=float(obs[8]),
                    wue=metrics.get("wue", 0),
                    pue=metrics.get("pue", 1),
                    carbon_intensity=metrics.get("carbon_intensity", 0),
                    thermal_satisfaction=metrics.get("thermal_satisfaction", 0),
                    reward=reward,
                    reward_weights=self.reward_fn.active_weights.to_dict(),
                )
                self.producer.publish_action(action_event)

            # Log every 100 steps
            if step % 100 == 0:
                logger.info(
                    f"Step {step:>5d} | "
                    f"Inlet: {metrics.get('inlet_temp_c', 0):5.1f}°C | "
                    f"WUE: {metrics.get('wue', 0):5.3f} | "
                    f"PUE: {metrics.get('pue', 0):5.3f} | "
                    f"Carbon: {metrics.get('carbon_intensity', 0):7.1f} | "
                    f"Reward: {reward:+7.4f}"
                )

            if terminated:
                logger.warning(f"Episode terminated at step {step}. Resetting...")
                obs, info = self.env.reset()

        logger.info("Control loop complete.")

    def start_dashboard(self) -> None:
        """Start the FastAPI dashboard server."""
        dashboard_cfg = self.config.get("dashboard", {})
        host = dashboard_cfg.get("host", "0.0.0.0")
        port = dashboard_cfg.get("port", 8000)
        logger.info(f"Starting dashboard at http://{host}:{port}")
        uvicorn.run(self.dashboard_app, host=host, port=port, log_level="info")

    def shutdown(self) -> None:
        """Gracefully shutdown all subsystems."""
        logger.info("Shutting down...")
        self._running = False
        self.consumer.stop()
        self.producer.close()
        logger.info("Shutdown complete.")


def main():
    parser = argparse.ArgumentParser(description="HydroTwin OS — Plane 3 Runtime")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--steps", type=int, default=1440, help="Control loop steps (default: 1440 = 24h)")
    parser.add_argument("--dashboard-only", action="store_true", help="Only start the dashboard server")
    parser.add_argument("--load-model", type=str, default=None, help="Path to trained model")
    args = parser.parse_args()

    config = load_config(args.config)
    orchestrator = NexusOrchestrator(config)

    # Handle shutdown signals
    def signal_handler(sig, frame):
        orchestrator.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    if args.load_model:
        orchestrator.agent.load(args.load_model)

    if args.dashboard_only:
        orchestrator.start_dashboard()
    else:
        # Run control loop, then start dashboard
        orchestrator.run_control_loop(steps=args.steps)
        orchestrator.start_dashboard()


if __name__ == "__main__":
    main()

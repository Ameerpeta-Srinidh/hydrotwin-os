"""
HydroTwin OS — Plane 3: Training Scenarios

Pre-built scenario profiles that control the external conditions during RL training.
Each scenario defines initial conditions and how ambient temperature, IT load,
grid carbon intensity, and water stress evolve over an episode.

Scenarios:
    NormalOps       — Mild weather, moderate load, clean grid
    HeatWave        — Extreme ambient temps, economizer disabled
    DirtyGrid       — High carbon intensity grid, should prefer evaporative cooling
    WaterStress     — High water stress region, should prefer chillers
    PeakLoad        — Maximum IT load, thermal constraint stressed
    CompoundCrisis  — Heat wave + dirty grid + water stress (migration trigger)
    RandomScenario  — Randomly selects and blends scenarios for training diversity
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Scenario(ABC):
    """Base class for training scenarios."""

    name: str = "base"

    @abstractmethod
    def initial_conditions(self, rng: np.random.Generator) -> dict[str, float]:
        """Return initial environment state values."""
        ...

    @abstractmethod
    def step_conditions(
        self,
        step: int,
        rng: np.random.Generator,
        current_ambient: float,
        current_it_load: float,
    ) -> dict[str, float]:
        """Return evolved external conditions for this timestep."""
        ...


class NormalOps(Scenario):
    """Mild weather, moderate load, clean grid — baseline operating conditions."""

    name = "normal_ops"

    def initial_conditions(self, rng: np.random.Generator) -> dict[str, float]:
        return {
            "server_inlet_temp_c": 22.0 + rng.uniform(-1, 1),
            "ambient_temp_c": 25.0 + rng.uniform(-3, 3),
            "wet_bulb_temp_c": 18.0 + rng.uniform(-2, 2),
            "relative_humidity": 0.45 + rng.uniform(-0.1, 0.1),
            "it_load_kw": 5000.0 + rng.uniform(-500, 500),
            "grid_carbon_intensity": 150.0 + rng.uniform(-30, 30),
            "water_stress_index": 1.5 + rng.uniform(-0.5, 0.5),
        }

    def step_conditions(
        self, step: int, rng: np.random.Generator,
        current_ambient: float, current_it_load: float,
    ) -> dict[str, float]:
        hour = (step % 1440) / 60.0  # minute-resolution hour of day

        # Diurnal temperature cycle
        ambient = 25.0 + 5.0 * math.sin(2 * math.pi * (hour - 14) / 24) + rng.normal(0, 0.3)
        wet_bulb = ambient - 7.0 + rng.normal(0, 0.2)

        # IT load: business-hours pattern
        base_load = 5000.0
        daily_swing = 1500.0 * math.sin(2 * math.pi * (hour - 13) / 24)
        it_load = max(2000.0, base_load + daily_swing + rng.normal(0, 100))

        return {
            "ambient_temp_c": ambient,
            "wet_bulb_temp_c": wet_bulb,
            "relative_humidity": np.clip(0.45 + 0.1 * math.sin(hour / 3) + rng.normal(0, 0.02), 0.1, 0.95),
            "it_load_kw": it_load,
            "grid_carbon_intensity": 150.0 + 50.0 * math.sin(2 * math.pi * (hour - 18) / 24) + rng.normal(0, 10),
            "water_stress_index": 1.5,
        }


class HeatWave(Scenario):
    """Extreme ambient temperatures — economizer becomes unavailable."""

    name = "heat_wave"

    def initial_conditions(self, rng: np.random.Generator) -> dict[str, float]:
        return {
            "server_inlet_temp_c": 26.0 + rng.uniform(-1, 1),
            "ambient_temp_c": 42.0 + rng.uniform(-2, 3),
            "wet_bulb_temp_c": 28.0 + rng.uniform(-1, 2),
            "relative_humidity": 0.25 + rng.uniform(-0.05, 0.1),
            "it_load_kw": 6000.0 + rng.uniform(-500, 1000),
            "grid_carbon_intensity": 250.0 + rng.uniform(-30, 50),
            "water_stress_index": 2.5 + rng.uniform(-0.5, 1.0),
        }

    def step_conditions(
        self, step: int, rng: np.random.Generator,
        current_ambient: float, current_it_load: float,
    ) -> dict[str, float]:
        hour = (step % 1440) / 60.0
        # Intense heat with small diurnal swing
        ambient = 43.0 + 4.0 * math.sin(2 * math.pi * (hour - 15) / 24) + rng.normal(0, 0.5)
        wet_bulb = ambient - 14.0 + rng.normal(0, 0.3)  # dry heat

        return {
            "ambient_temp_c": ambient,
            "wet_bulb_temp_c": wet_bulb,
            "relative_humidity": np.clip(0.2 + rng.normal(0, 0.03), 0.05, 0.5),
            "it_load_kw": max(3000.0, current_it_load + rng.normal(0, 50)),
            "grid_carbon_intensity": 300.0 + 80.0 * math.sin(2 * math.pi * (hour - 16) / 24) + rng.normal(0, 15),
            "water_stress_index": 3.0 + rng.uniform(-0.2, 0.3),
        }


class DirtyGrid(Scenario):
    """High carbon intensity grid — agent should prefer water-based cooling to avoid carbon."""

    name = "dirty_grid"

    def initial_conditions(self, rng: np.random.Generator) -> dict[str, float]:
        return {
            "server_inlet_temp_c": 22.0 + rng.uniform(-1, 1),
            "ambient_temp_c": 20.0 + rng.uniform(-3, 3),
            "wet_bulb_temp_c": 15.0 + rng.uniform(-2, 2),
            "relative_humidity": 0.55 + rng.uniform(-0.1, 0.1),
            "it_load_kw": 5500.0 + rng.uniform(-500, 500),
            "grid_carbon_intensity": 600.0 + rng.uniform(-50, 100),
            "water_stress_index": 1.0 + rng.uniform(-0.3, 0.3),
        }

    def step_conditions(
        self, step: int, rng: np.random.Generator,
        current_ambient: float, current_it_load: float,
    ) -> dict[str, float]:
        hour = (step % 1440) / 60.0
        ambient = 20.0 + 6.0 * math.sin(2 * math.pi * (hour - 14) / 24) + rng.normal(0, 0.3)

        return {
            "ambient_temp_c": ambient,
            "wet_bulb_temp_c": ambient - 5.0 + rng.normal(0, 0.2),
            "relative_humidity": np.clip(0.55 + rng.normal(0, 0.03), 0.2, 0.9),
            "it_load_kw": max(3000.0, 5500.0 + 1000.0 * math.sin(2 * math.pi * (hour - 13) / 24) + rng.normal(0, 80)),
            "grid_carbon_intensity": 650.0 + 150.0 * math.sin(2 * math.pi * (hour - 18) / 24) + rng.normal(0, 20),
            "water_stress_index": 1.0,
        }


class WaterStress(Scenario):
    """High water stress — agent should prefer chiller-based cooling despite energy cost."""

    name = "water_stress"

    def initial_conditions(self, rng: np.random.Generator) -> dict[str, float]:
        return {
            "server_inlet_temp_c": 23.0 + rng.uniform(-1, 1),
            "ambient_temp_c": 35.0 + rng.uniform(-3, 5),
            "wet_bulb_temp_c": 22.0 + rng.uniform(-2, 3),
            "relative_humidity": 0.3 + rng.uniform(-0.05, 0.1),
            "it_load_kw": 5000.0 + rng.uniform(-500, 1000),
            "grid_carbon_intensity": 200.0 + rng.uniform(-30, 50),
            "water_stress_index": 4.0 + rng.uniform(-0.3, 0.8),
        }

    def step_conditions(
        self, step: int, rng: np.random.Generator,
        current_ambient: float, current_it_load: float,
    ) -> dict[str, float]:
        hour = (step % 1440) / 60.0
        ambient = 36.0 + 6.0 * math.sin(2 * math.pi * (hour - 15) / 24) + rng.normal(0, 0.4)

        return {
            "ambient_temp_c": ambient,
            "wet_bulb_temp_c": ambient - 12.0 + rng.normal(0, 0.3),
            "relative_humidity": np.clip(0.25 + rng.normal(0, 0.03), 0.05, 0.6),
            "it_load_kw": max(3000.0, 5000.0 + 1500.0 * math.sin(2 * math.pi * (hour - 13) / 24) + rng.normal(0, 80)),
            "grid_carbon_intensity": 200.0 + 60.0 * math.sin(2 * math.pi * (hour - 17) / 24) + rng.normal(0, 10),
            "water_stress_index": 4.2 + rng.uniform(-0.1, 0.2),
        }


class PeakLoad(Scenario):
    """Maximum IT load — thermal constraint is the primary challenge."""

    name = "peak_load"

    def initial_conditions(self, rng: np.random.Generator) -> dict[str, float]:
        return {
            "server_inlet_temp_c": 24.0 + rng.uniform(-1, 2),
            "ambient_temp_c": 30.0 + rng.uniform(-2, 5),
            "wet_bulb_temp_c": 22.0 + rng.uniform(-2, 3),
            "relative_humidity": 0.45 + rng.uniform(-0.1, 0.1),
            "it_load_kw": 9000.0 + rng.uniform(-500, 1000),
            "grid_carbon_intensity": 200.0 + rng.uniform(-40, 60),
            "water_stress_index": 2.0 + rng.uniform(-0.5, 0.5),
        }

    def step_conditions(
        self, step: int, rng: np.random.Generator,
        current_ambient: float, current_it_load: float,
    ) -> dict[str, float]:
        hour = (step % 1440) / 60.0
        ambient = 32.0 + 5.0 * math.sin(2 * math.pi * (hour - 14) / 24) + rng.normal(0, 0.4)
        # IT load stays very high with spikes
        it_load = max(7000.0, 9000.0 + 1000.0 * math.sin(2 * math.pi * (hour - 12) / 24) + rng.normal(0, 200))

        return {
            "ambient_temp_c": ambient,
            "wet_bulb_temp_c": ambient - 8.0 + rng.normal(0, 0.3),
            "relative_humidity": np.clip(0.4 + rng.normal(0, 0.03), 0.1, 0.8),
            "it_load_kw": it_load,
            "grid_carbon_intensity": 220.0 + 70.0 * math.sin(2 * math.pi * (hour - 18) / 24) + rng.normal(0, 15),
            "water_stress_index": 2.0,
        }


class CompoundCrisis(Scenario):
    """
    Heat wave + dirty grid + water stress — the worst-case scenario.
    This should trigger the workload migration subsystem.
    """

    name = "compound_crisis"

    def initial_conditions(self, rng: np.random.Generator) -> dict[str, float]:
        return {
            "server_inlet_temp_c": 28.0 + rng.uniform(-1, 2),
            "ambient_temp_c": 45.0 + rng.uniform(-2, 3),
            "wet_bulb_temp_c": 30.0 + rng.uniform(-1, 2),
            "relative_humidity": 0.3 + rng.uniform(-0.05, 0.1),
            "it_load_kw": 8000.0 + rng.uniform(-500, 1500),
            "grid_carbon_intensity": 700.0 + rng.uniform(-50, 100),
            "water_stress_index": 4.5 + rng.uniform(-0.3, 0.5),
        }

    def step_conditions(
        self, step: int, rng: np.random.Generator,
        current_ambient: float, current_it_load: float,
    ) -> dict[str, float]:
        hour = (step % 1440) / 60.0
        # Extreme heat, barely cools at night
        ambient = 46.0 + 3.0 * math.sin(2 * math.pi * (hour - 15) / 24) + rng.normal(0, 0.5)

        return {
            "ambient_temp_c": ambient,
            "wet_bulb_temp_c": ambient - 15.0 + rng.normal(0, 0.3),
            "relative_humidity": np.clip(0.2 + rng.normal(0, 0.02), 0.05, 0.4),
            "it_load_kw": max(6000.0, current_it_load + rng.normal(0, 100)),
            "grid_carbon_intensity": 750.0 + 100.0 * math.sin(2 * math.pi * (hour - 16) / 24) + rng.normal(0, 25),
            "water_stress_index": 4.5 + rng.uniform(-0.1, 0.3),
        }


class RandomScenario(Scenario):
    """Randomly selects a scenario at each episode reset for training diversity."""

    name = "random"

    _SCENARIOS = [NormalOps, HeatWave, DirtyGrid, WaterStress, PeakLoad, CompoundCrisis]

    def __init__(self):
        self._active: Scenario | None = None

    def initial_conditions(self, rng: np.random.Generator) -> dict[str, float]:
        cls = rng.choice(self._SCENARIOS)  # type: ignore
        self._active = cls()
        return self._active.initial_conditions(rng)

    def step_conditions(
        self, step: int, rng: np.random.Generator,
        current_ambient: float, current_it_load: float,
    ) -> dict[str, float]:
        assert self._active is not None, "Must call initial_conditions first"
        return self._active.step_conditions(step, rng, current_ambient, current_it_load)


# ── Scenario registry ──
SCENARIOS: dict[str, type[Scenario]] = {
    "normal_ops": NormalOps,
    "heat_wave": HeatWave,
    "dirty_grid": DirtyGrid,
    "water_stress": WaterStress,
    "peak_load": PeakLoad,
    "compound_crisis": CompoundCrisis,
    "random": RandomScenario,
}


def get_scenario(name: str) -> Scenario:
    """Get a scenario instance by name."""
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{name}'. Available: {list(SCENARIOS.keys())}")
    return SCENARIOS[name]()

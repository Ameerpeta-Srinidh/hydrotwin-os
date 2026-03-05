"""
HydroTwin OS — Plane 3: Data Center Thermal Environment

A custom Gymnasium environment that simulates the thermal dynamics of a data center
cooling system. The agent controls cooling parameters and the environment computes
the resulting Water Usage Effectiveness (WUE), Power Usage Effectiveness (PUE),
and carbon emissions based on a simplified but physically grounded thermal model.

State Space (12 continuous dims):
    0: server_inlet_temp_c       — Current server inlet temperature (°C)
    1: server_outlet_temp_c      — Current server outlet temperature (°C)
    2: ambient_temp_c            — Outside ambient dry-bulb temperature (°C)
    3: wet_bulb_temp_c           — Outside wet-bulb temperature (°C)
    4: relative_humidity         — Outside relative humidity (0–1)
    5: it_load_kw                — Current IT power draw (kW)
    6: cooling_load_kw           — Total cooling power being consumed (kW)
    7: grid_carbon_intensity     — Grid marginal carbon intensity (gCO₂/kWh)
    8: water_stress_index        — WRI Aqueduct water stress (0–5 scale)
    9: evap_tower_water_flow_lpm — Evaporative tower water consumption (L/min)
   10: chiller_power_kw          — Mechanical chiller power draw (kW)
   11: fan_speed_pct             — Current fan speed (0–1)

Action Space (4 continuous dims):
    0: cooling_mode_mix     ∈ [0, 1]   — 0 = full evaporative, 1 = full chiller
    1: supply_air_temp_sp   ∈ [15, 25] — Supply air temperature setpoint (°C)
    2: fan_speed_pct        ∈ [0.2, 1] — Fan speed as fraction of max
    3: economizer_damper    ∈ [0, 1]   — Outside air economizer fraction
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Any

from hydrotwin.env.scenarios import Scenario, NormalOps
from hydrotwin.reward.pareto_reward import ParetoReward


# ───────────────────────────── Physical Constants ─────────────────────────────

_SPECIFIC_HEAT_AIR = 1.005      # kJ/(kg·°C)
_AIR_DENSITY = 1.2              # kg/m³
_AIRFLOW_PER_FAN_PCT = 50.0     # m³/s at 100% fan speed
_EVAP_COOLING_EFF = 0.85        # evaporative pad effectiveness
_WATER_PER_KWH_EVAP = 1.8      # L/kWh — typical evaporative water consumption
_DT = 60.0                      # simulation timestep (seconds)
_STEPS_PER_HOUR = 3600 / _DT


class DataCenterEnv(gym.Env):
    """
    Gymnasium environment for data center cooling control.

    The agent's goal is to keep server inlet temperatures within ASHRAE A1
    limits while minimizing a multi-objective cost across water usage,
    energy usage, and carbon emissions.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(
        self,
        scenario: Scenario | None = None,
        max_episode_steps: int = 1440,  # 24 hours at 1-min steps
        render_mode: str | None = None,
        facility_config: dict[str, Any] | None = None,
    ):
        super().__init__()

        self.scenario = scenario or NormalOps()
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.reward_fn = ParetoReward()

        # ── Facility parameters (defaults or from config) ──
        fc = facility_config or {}
        thermal = fc.get("thermal", {})
        cooling = fc.get("cooling", {})

        self.inlet_min_c = thermal.get("server_inlet_min_c", 15.0)
        self.inlet_max_c = thermal.get("server_inlet_max_c", 32.0)
        self.thermal_mass = thermal.get("thermal_mass_kj_per_c", 5000.0)
        self.max_it_load_kw = thermal.get("max_it_load_kw", 10000.0)

        self.evap_max_flow_lpm = cooling.get("evap_tower_max_flow_lpm", 2000.0)
        self.chiller_max_kw = cooling.get("chiller_max_power_kw", 3000.0)
        self.chiller_cop = cooling.get("chiller_cop", 4.5)
        self.econ_threshold_c = cooling.get("economizer_threshold_c", 18.0)

        # ── Observation space (12 dims, all normalized to roughly [0, 1]) ──
        obs_low = np.array([
            10.0,    # server_inlet_temp_c
            15.0,    # server_outlet_temp_c
            -10.0,   # ambient_temp_c
            -15.0,   # wet_bulb_temp_c
            0.0,     # relative_humidity
            0.0,     # it_load_kw
            0.0,     # cooling_load_kw
            0.0,     # grid_carbon_intensity
            0.0,     # water_stress_index
            0.0,     # evap_tower_water_flow_lpm
            0.0,     # chiller_power_kw
            0.0,     # fan_speed_pct
        ], dtype=np.float32)

        obs_high = np.array([
            50.0,               # server_inlet_temp_c
            70.0,               # server_outlet_temp_c
            55.0,               # ambient_temp_c (extreme heat)
            45.0,               # wet_bulb_temp_c
            1.0,                # relative_humidity
            self.max_it_load_kw,
            self.chiller_max_kw + 2000.0,
            1000.0,             # grid_carbon_intensity (gCO2/kWh)
            5.0,                # water_stress_index
            self.evap_max_flow_lpm,
            self.chiller_max_kw,
            1.0,                # fan_speed_pct
        ], dtype=np.float32)

        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # ── Action space (4 continuous dims) ──
        self.action_space = spaces.Box(
            low=np.array([0.0, 15.0, 0.2, 0.0], dtype=np.float32),
            high=np.array([1.0, 25.0, 1.0, 1.0], dtype=np.float32),
        )

        # ── Internal state ──
        self._step_count = 0
        self._state = np.zeros(12, dtype=np.float32)
        self._metrics: dict[str, float] = {}

    # ──────────────────────────── Gym Interface ────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0

        # Get initial conditions from scenario
        init = self.scenario.initial_conditions(self.np_random)

        self._state = np.array([
            init["server_inlet_temp_c"],
            init["server_inlet_temp_c"] + 12.0,  # typical ΔT across servers
            init["ambient_temp_c"],
            init["wet_bulb_temp_c"],
            init["relative_humidity"],
            init["it_load_kw"],
            0.0,  # cooling_load_kw — computed on first step
            init["grid_carbon_intensity"],
            init["water_stress_index"],
            0.0,  # evap_tower_water_flow_lpm
            0.0,  # chiller_power_kw
            0.5,  # fan_speed_pct (initial)
        ], dtype=np.float32)

        self._metrics = {}
        return self._state.copy(), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1

        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        cooling_mix, supply_temp_sp, fan_speed, econ_damper = action

        # ── Unpack current state ──
        (
            inlet_temp, outlet_temp, ambient_temp, wet_bulb_temp,
            rel_humidity, it_load, _, grid_carbon, water_stress,
            _, _, _,
        ) = self._state

        # ── Evolve external conditions (scenario-driven) ──
        ext = self.scenario.step_conditions(
            self._step_count, self.np_random,
            current_ambient=ambient_temp,
            current_it_load=it_load,
        )
        ambient_temp = ext["ambient_temp_c"]
        wet_bulb_temp = ext["wet_bulb_temp_c"]
        rel_humidity = ext["relative_humidity"]
        it_load = ext["it_load_kw"]
        grid_carbon = ext["grid_carbon_intensity"]
        water_stress = ext["water_stress_index"]

        # ── Physics: Compute cooling capacity ──

        # Airflow
        airflow_m3s = _AIRFLOW_PER_FAN_PCT * fan_speed
        mass_flow = airflow_m3s * _AIR_DENSITY  # kg/s

        # Economizer (free cooling): effective if ambient < threshold
        econ_available = ambient_temp < self.econ_threshold_c
        econ_fraction = econ_damper if econ_available else 0.0

        # Mixed air temperature (return air blended with outside air)
        return_air_temp = outlet_temp
        mixed_air_temp = (1.0 - econ_fraction) * return_air_temp + econ_fraction * ambient_temp

        # Evaporative cooling power
        evap_delta_t = _EVAP_COOLING_EFF * (mixed_air_temp - wet_bulb_temp)
        evap_cooling_kw = (1.0 - cooling_mix) * mass_flow * _SPECIFIC_HEAT_AIR * max(evap_delta_t, 0.0)

        # Mechanical chiller cooling power
        chiller_cooling_kw = cooling_mix * self.chiller_max_kw * fan_speed
        chiller_power_kw = chiller_cooling_kw / self.chiller_cop

        # Total cooling
        total_cooling_kw = evap_cooling_kw + chiller_cooling_kw

        # ── Physics: Thermal energy balance ──
        q_it = it_load  # kW generated by servers
        q_removal = total_cooling_kw

        # Temperature change: dT = dt * (Q_gen - Q_removal) / thermal_mass
        delta_t = _DT * (q_it - q_removal) / self.thermal_mass
        new_inlet_temp = inlet_temp + delta_t

        # Clamp to physical limits (environment can still overheat, but bounded)
        new_inlet_temp = np.clip(new_inlet_temp, 5.0, 60.0)
        new_outlet_temp = new_inlet_temp + 10.0 + (it_load / self.max_it_load_kw) * 5.0

        # ── Compute metrics ──
        # Water consumption (L/min) — only evaporative cooling uses water
        evap_water_flow_lpm = (1.0 - cooling_mix) * (evap_cooling_kw / 60.0) * _WATER_PER_KWH_EVAP

        # Fan power (simplified: cubic law with fan speed)
        fan_power_kw = 200.0 * (fan_speed ** 3)

        # Total facility power
        facility_power_kw = it_load + chiller_power_kw + fan_power_kw

        # Cooling load (total non-IT power)
        cooling_load_kw = chiller_power_kw + fan_power_kw

        # WUE: liters per kWh of IT energy
        it_energy_kwh = it_load / _STEPS_PER_HOUR  # energy in this timestep
        water_consumed_liters = evap_water_flow_lpm * (_DT / 60.0)
        wue = water_consumed_liters / max(it_energy_kwh, 0.01)

        # PUE: total facility power / IT power
        pue = facility_power_kw / max(it_load, 1.0)

        # Carbon emissions for this timestep (gCO₂)
        facility_energy_kwh = facility_power_kw / _STEPS_PER_HOUR
        carbon_emissions_g = grid_carbon * facility_energy_kwh

        # Carbon intensity (normalized for reward)
        carbon_intensity = carbon_emissions_g / max(it_energy_kwh, 0.01)

        # Thermal constraint satisfaction
        temp_mid = (self.inlet_min_c + self.inlet_max_c) / 2.0
        temp_range = (self.inlet_max_c - self.inlet_min_c) / 2.0
        temp_deviation = abs(new_inlet_temp - temp_mid) / temp_range
        thermal_satisfaction = max(0.0, 1.0 - temp_deviation)

        # ── Update state ──
        self._state = np.array([
            new_inlet_temp,
            new_outlet_temp,
            ambient_temp,
            wet_bulb_temp,
            rel_humidity,
            it_load,
            cooling_load_kw,
            grid_carbon,
            water_stress,
            evap_water_flow_lpm,
            chiller_power_kw,
            fan_speed,
        ], dtype=np.float32)

        # ── Store metrics for reward computation & info ──
        self._metrics = {
            "wue": float(wue),
            "pue": float(pue),
            "carbon_intensity": float(carbon_intensity),
            "thermal_satisfaction": float(thermal_satisfaction),
            "inlet_temp_c": float(new_inlet_temp),
            "water_consumed_liters": float(water_consumed_liters),
            "carbon_emissions_g": float(carbon_emissions_g),
            "facility_power_kw": float(facility_power_kw),
            "chiller_power_kw": float(chiller_power_kw),
            "evap_water_flow_lpm": float(evap_water_flow_lpm),
            "cooling_mix": float(cooling_mix),
            "economizer_fraction": float(econ_fraction),
        }

        # ── Termination conditions ──
        terminated = False
        truncated = self._step_count >= self.max_episode_steps

        # Critical thermal failure — inlet temp too far out of range
        if new_inlet_temp > 45.0 or new_inlet_temp < 5.0:
            terminated = True

        # ── Compute Multi-Objective Reward ──
        reward = self.reward_fn.compute(self._metrics)

        return self._state.copy(), reward, terminated, truncated, self._get_info()

    def _get_info(self) -> dict[str, Any]:
        """Return current metrics and state metadata."""
        return {
            "step": self._step_count,
            "metrics": self._metrics.copy(),
            "state": {
                "inlet_temp_c": float(self._state[0]),
                "ambient_temp_c": float(self._state[2]),
                "it_load_kw": float(self._state[5]),
                "grid_carbon": float(self._state[7]),
                "water_stress": float(self._state[8]),
            },
        }

    @property
    def metrics(self) -> dict[str, float]:
        """Access current step metrics (for reward computation)."""
        return self._metrics

    def render(self):
        if self.render_mode == "ansi":
            m = self._metrics
            return (
                f"Step {self._step_count:>5d} | "
                f"Inlet: {self._state[0]:5.1f}°C | "
                f"WUE: {m.get('wue', 0):5.2f} | "
                f"PUE: {m.get('pue', 0):5.2f} | "
                f"Carbon: {m.get('carbon_intensity', 0):7.1f} | "
                f"Thermal: {m.get('thermal_satisfaction', 0):4.2f}"
            )
        return None

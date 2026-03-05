"""
HydroTwin OS — Plane 3: RL Inference Service

Loads trained SAC models and executes real-time inference routing.
"""

import time
import logging
import numpy as np

try:
    from stable_baselines3 import SAC
except ImportError:
    pass

logger = logging.getLogger("RLInference")

class RLInferenceService:
    def __init__(self, model_path: str = "models/sac_carbon_prio"):
        logger.info(f"Bootstrapping RL Inference Service with model {model_path}...")
        
        # Using .zip extension internally if not provided by user
        if not model_path.endswith('.zip'):
            model_path += '.zip'
            
        try:
            self.model = SAC.load(model_path)
            logger.info("SAC model loaded successfully.")
            self.active = True
        except Exception as e:
            logger.error(f"Failed to load RL model: {e}")
            self.active = False
            
    def predict(self, state_dict: dict) -> dict:
        """
        Maps a state dictionary down to the exact 12-dim continuous array
        expected by the DataCenterEnv and PyTorch model.
        """
        if not self.active:
            raise RuntimeError("RL Model is offline or uninitialized.")
            
        start_time = time.perf_counter()
        
        # Space mappings (refer to DataCenterEnv constructor)
        obs = np.array([
            state_dict.get("inlet_temp_c", 22.0),
            state_dict.get("outlet_temp_c", 35.0),
            state_dict.get("ambient_temp_c", 25.0),
            state_dict.get("wet_bulb_temp_c", 18.0),
            state_dict.get("relative_humidity", 0.5),
            state_dict.get("it_load_kw", 5000.0),
            state_dict.get("cooling_load_kw", 1000.0),
            state_dict.get("grid_carbon_intensity", 200.0),
            state_dict.get("water_stress_index", 1.0),
            state_dict.get("evap_water_flow_lpm", 10.0),
            state_dict.get("chiller_power_kw", 800.0),
            state_dict.get("fan_speed_pct", 0.6)
        ], dtype=np.float32)
        
        # Execute forward pass
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Record execution time for observability metrics
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        
        return {
            "cooling_mix": float(action[0]),
            "supply_air_temp_sp": float(action[1]),
            "fan_speed_pct": float(action[2]),
            "economizer_damper": float(action[3]),
            "latency_ms": latency_ms
        }

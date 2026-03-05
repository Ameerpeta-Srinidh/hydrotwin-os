"""
HydroTwin OS — Live Data Center Sensor Node

Runs the DataCenterEnv continuously, pushing actual PyTorch/gymnasium state
to the Kafka event mesh, and pulling actions from the RL policy to step the physics.
"""

import time
import logging
from typing import Any

from hydrotwin.env.datacenter_env import DataCenterEnv
from hydrotwin.env.scenarios import NormalOps, HeatWave
from hydrotwin.events.kafka_producer import NexusKafkaProducer
from hydrotwin.events.kafka_consumer import NexusKafkaConsumer

logger = logging.getLogger("LiveDatacenter")
logging.basicConfig(level=logging.INFO)

def run_live_datacenter():
    logger.info("Starting up Live Datacenter Physics Engine...")
    env = DataCenterEnv(scenario=NormalOps())
    obs, _ = env.reset()
    
    # Needs real Kafka
    producer = NexusKafkaProducer(mock_mode=False)
    consumer = NexusKafkaConsumer(group_id="hydrotwin-physics-node")
    consumer.subscribe(["hydrotwin.actions"])
    
    # Store latest action from RL
    latest_action = env.action_space.sample()
    
    def action_handler(topic, value):
        nonlocal latest_action
        try:
            latest_action = [
                value.get("cooling_mode_mix", 0.3),
                value.get("supply_air_temp_sp", 22.0),
                value.get("fan_speed_pct", 0.5),
                value.get("economizer_damper", 0.0)
            ]
        except Exception as e:
            logger.error(f"Failed to parse action: {e}")
            
    # Prime the pump using background thread for consuming
    import threading
    t = threading.Thread(target=consumer.consume_loop, args=(action_handler,), daemon=True)
    t.start()
    
    step_count = 0
    try:
        while True:
            # Step physics using latest RL action
            obs, reward, term, trunc, info = env.step(latest_action)
            
            # Publish state telemetry to Kafka
            metrics = info["metrics"]
            # To act as the full state, we push metrics dict
            producer._publish("hydrotwin.telemetry", metrics)
            producer._producer.flush()
            
            step_count += 1
            if step_count % 60 == 0: # Print every 60 steps (1 hour simulated)
                logger.info(f"Stepped {step_count} times. Inlet: {metrics['inlet_temp_c']:.1f}C, WUE: {metrics['wue']:.2f}")
                
            if term or trunc:
                logger.info("Episode ended. Resetting environment.")
                obs, _ = env.reset()
                
            # Run at 1 Hz real time = 1 minute simulated time
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        logger.info("Live physics engine terminating.")

if __name__ == "__main__":
    run_live_datacenter()

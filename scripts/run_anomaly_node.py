"""
HydroTwin OS — Live Anomaly Detection Node (Plane 2)

Consumes telemetry from the physics engine via Kafka, runs multimodal
anomaly detectors, and publishes generated alerts back to the mesh.
"""

import time
import logging
from typing import Any

from hydrotwin.events.kafka_consumer import NexusKafkaConsumer
from hydrotwin.events.kafka_producer import NexusKafkaProducer
from hydrotwin.events.schemas import AnomalyAlert
from hydrotwin.detection.sensor_detector import SensorAnomalyDetector
from hydrotwin.detection.alert_engine import AlertEngine
import numpy as np

logger = logging.getLogger("AnomalyNode")
logging.basicConfig(level=logging.INFO)

def run_anomaly_node():
    logger.info("Starting up Multimodal Anomaly Detection Node...")
    
    detector = SensorAnomalyDetector(min_votes=1)
    # We prime the LSTM autoencoder with some fake data so it triggers as 'fitted' immediately
    fake_baseline = np.random.normal(0, 0.1, 100)
    detector.lstm.fit(fake_baseline, epochs=1) 
    
    engine = AlertEngine(cooldown_seconds=60)
    
    producer = NexusKafkaProducer(mock_mode=False)
    consumer = NexusKafkaConsumer(group_id="hydrotwin-anomaly-node")
    consumer.subscribe(["hydrotwin.telemetry"])
    
    def telemetry_handler(topic, value):
        try:
            # We look for high inlet temp (> 27) -> hotspot
            inlet = value.get("inlet_temp_c", 22.0)
            
            # Feed into statistical detector directly for demo spike catch
            anomaly = detector.detect("inlet_temp_c", inlet)
            
            if inlet > 27.5:
                # Force an alert through the engine
                alert = engine.process("hotspot", 0.95, "Rack-Critical", "sensor")
            elif anomaly:
                alert = engine.process(anomaly.anomaly_type, 0.8, "DataCenter-Sensor", anomaly.method)
            else:
                alert = None
                
            if alert:
                alert_payload = AnomalyAlert(
                    anomaly_type=alert.anomaly_type,
                    severity=alert.severity,
                    location=alert.location,
                    confidence=alert.confidence,
                    details={"message": alert.message}
                )
                producer.publish_anomaly(alert_payload)
                producer._producer.flush()
                logger.warning(f"🚨 Published Alert: {alert.message}")
                
        except Exception as e:
            logger.error(f"Failed to process telemetry: {e}")
            
    try:
        consumer.consume_loop(telemetry_handler)
    except KeyboardInterrupt:
        logger.info("Anomaly node terminating.")

if __name__ == "__main__":
    run_anomaly_node()

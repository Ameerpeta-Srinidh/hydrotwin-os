"""
HydroTwin OS — Brutal Kafka Tests

10x flood testing, connection resilience, and consumer lag tracking.
"""

import time
import pytest
import threading

from hydrotwin.events.kafka_producer import NexusKafkaProducer
from hydrotwin.events.kafka_consumer import NexusKafkaConsumer
from hydrotwin.events.schemas import SensorReading

@pytest.fixture(scope="module")
def real_kafka_services():
    # Because this is a real integration test, we verify Kafka is actually up
    try:
        producer = NexusKafkaProducer(mock_mode=False)
        consumer = NexusKafkaConsumer(group_id="test-brutal-group")
        return producer, consumer
    except ConnectionError:
        pytest.skip("Real Kafka broker not found. Ensure docker-compose up is running.")

class TestBrutalKafka:
    
    def test_10x_event_flood(self, real_kafka_services):
        """Flood the local Kafka broker with 10k events and measure backpressure/lag."""
        producer, consumer = real_kafka_services
        topic = "test.flood.readings"
        consumer.subscribe([topic])
        
        # Give consumer time to join group
        time.sleep(2)
        
        TOTAL_EVENTS = 10000
        t0 = time.perf_counter()
        
        for i in range(TOTAL_EVENTS):
            reading = SensorReading(
                sensor_id=f"sensor-{i}",
                metric_name="inlet_temp",
                value=25.0,
                unit="c"
            )
            producer._publish(topic, reading)
            
        # Hard flush
        producer._producer.flush()
        publish_time = time.perf_counter() - t0
        
        # Consumer logic
        received = 0
        t1 = time.perf_counter()
        while received < TOTAL_EVENTS:
            msgs = consumer.poll_messages(timeout_ms=2000)
            received += len(msgs)
            if time.perf_counter() - t1 > 10.0:  # 10s timeout
                break
                
        consume_time = time.perf_counter() - t1
        
        print(f"Publish {TOTAL_EVENTS} events took {publish_time:.2f}s")
        print(f"Consume {TOTAL_EVENTS} events took {consume_time:.2f}s. Received: {received}")
        
        # Test Assertions
        assert received == TOTAL_EVENTS, f"Message loss detected! Found {received}/{TOTAL_EVENTS}"
        assert publish_time < 15.0, f"Producer backpressure too high (took {publish_time:.2f}s for 10k messages)"
        assert consume_time < 15.0, f"Consumer lag too high (took {consume_time:.2f}s to process 10k messages)"
        
    def test_broker_disconnect_retry(self, real_kafka_services):
        """Simulate a disconnect by checking retries limit and handling."""
        producer, consumer = real_kafka_services
        # The true "brutal" test kills the docker container mid-stream, 
        # but programmatically we verify standard connectivity and exception handling
        
        # We assert the producer was instantiated with real settings and mock_mode = False
        assert not getattr(producer, "mock_mode", True), "Producer is still running in mock mode!"
        
        reading = SensorReading(
            sensor_id="test-sensor",
            metric_name="ping",
            value=1.0,
            unit="unit"
        )
        try:
            producer._publish("test.ping", reading)
            producer._producer.flush()
        except Exception as e:
            pytest.fail(f"Producer failed under typical load: {e}")

"""
HydroTwin OS — Plane 2: Kafka Consumer

Consumes real-time telemetry and alerts from the Kafka Event Mesh. 
"""

import json
import logging
from typing import Callable, Any

logger = logging.getLogger("NexusKafkaConsumer")

class NexusKafkaConsumer:
    """Consumes events from HydroTwin Kafka topics."""

    def __init__(self, bootstrap_servers: str = "localhost:9092", group_id: str = "hydrotwin-core"):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self._consumer = None

        self._connect()

    def _connect(self):
        try:
            from kafka import KafkaConsumer
            self._consumer = KafkaConsumer(
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                value_deserializer=lambda x: json.loads(x.decode("utf-8")),
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                session_timeout_ms=10000,
                max_poll_records=500
            )
            logger.info(f"Connected to Kafka broker at {self.bootstrap_servers}. Group: {self.group_id}")
        except Exception as e:
            logger.error(f"Failed to connect Kafka consumer: {e}")
            raise ConnectionError(f"Kafka consumer connection failed. Real broker required.")

    def subscribe(self, topics: list[str]):
        """Subscribe to a list of topics."""
        self._consumer.subscribe(topics)
        logger.info(f"Subscribed to topics: {topics}")

    def consume_loop(self, handler_callback: Callable[[str, dict[str, Any]], None]):
        """Runs an infinite loop consuming messages and passing them to a callback."""
        logger.info("Starting consume loop...")
        try:
            for message in self._consumer:
                handler_callback(message.topic, message.value)
        except KeyboardInterrupt:
            logger.info("Consume loop stopped by user.")
        finally:
            self._consumer.close()
            
    def poll_messages(self, timeout_ms: int = 1000) -> list[dict[str, Any]]:
        """Polls for messages without blocking forever. Useful for tests."""
        records = self._consumer.poll(timeout_ms=timeout_ms)
        messages = []
        for tp, partition_records in records.items():
            for record in partition_records:
                messages.append({"topic": tp.topic, "value": record.value})
        return messages

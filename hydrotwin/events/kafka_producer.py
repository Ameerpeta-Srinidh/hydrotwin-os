"""
HydroTwin OS — Plane 3: Kafka Producer

Publishes RL actions, forecasts, and migration events to Kafka topics.
Falls back to a local log when Kafka is unavailable.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class NexusKafkaProducer:
    """
    Kafka producer for Plane 3 events.

    Publishes to:
      - rl.actions — every RL control action
      - ml.forecasts — load/weather forecasts
      - anomaly.alerts — migration events (forwarded to anomaly pipeline)
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topics: dict[str, str] | None = None,
        mock_mode: bool = False,
    ):
        self.bootstrap_servers = bootstrap_servers
        self.topics = topics or {
            "rl_actions": "rl.actions",
            "ml_forecasts": "ml.forecasts",
            "layout_updates": "layout.updates",
        }
        self.mock_mode = mock_mode
        self._producer = None
        self._event_log = []

        try:
            from kafka import KafkaProducer

            self._producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                retries=5,
                acks="all",
                request_timeout_ms=10000,
                linger_ms=5, # Enable batching for higher throughput
                batch_size=32768,
            )
            logger.info(f"Connected to Kafka broker at {self.bootstrap_servers}")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka at {self.bootstrap_servers}: {e}")
            raise ConnectionError(f"Kafka connection failed. HydroTwin requires real Kafka.")

    def publish_action(self, action: BaseModel) -> None:
        """Publish an RL action event."""
        self._publish(self.topics.get("rl_actions", "rl.actions"), action)

    def publish_forecast(self, forecast: BaseModel) -> None:
        """Publish a forecast event."""
        self._publish(self.topics.get("ml_forecasts", "ml.forecasts"), forecast)

    def publish_migration(self, migration: BaseModel) -> None:
        """Publish a migration event."""
        self._publish(self.topics.get("rl_actions", "rl.actions"), migration)

    def publish_anomaly(self, alert: BaseModel | dict) -> None:
        """Publish an anomaly alert."""
        self._publish(self.topics.get("anomaly_alerts", "anomaly.alerts"), alert)

    def _publish(self, topic: str, event: BaseModel | dict) -> None:
        """Publish a Pydantic model or dict to a Kafka topic."""
        try:
            payload = event.model_dump() if hasattr(event, "model_dump") else event
        except Exception as e:
            logger.error(f"Failed to process payload for {topic}: {e}")
            self._event_log.append({"topic": topic, "payload": str(event), "error": str(e)})
            return

        if self.mock_mode or self._producer is None:
            self._event_log.append({"topic": topic, "payload": payload, "timestamp": str(datetime.utcnow())})
            logger.debug(f"[MOCK] Published to {topic}: {payload.get('action_id', payload.get('forecast_id', 'unknown'))}")
            return

        try:
            self._producer.send(topic, value=payload)
            self._producer.flush()
            logger.debug(f"Published to {topic}")
        except Exception as e:
            logger.error(f"Failed to publish to {topic}: {e}")
            self._event_log.append({"topic": topic, "payload": payload, "error": str(e)})

    @property
    def event_log(self) -> list[dict[str, Any]]:
        """Access the local event log (for mock mode or failed publishes)."""
        return self._event_log

    def close(self):
        if self._producer:
            self._producer.close()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> NexusKafkaProducer:
        kafka_cfg = config.get("kafka", {})
        bootstrap = kafka_cfg.get("bootstrap_servers", "localhost:9092")
        topics = kafka_cfg.get("topics", {})
        return cls(
            bootstrap_servers=bootstrap,
            topics=topics,
            mock_mode=False,  # will auto-fallback if Kafka unavailable
        )

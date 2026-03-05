"""
HydroTwin OS — Plane 2: Sensor Anomaly Detector

Multi-method anomaly detection on data center sensor streams.
Combines statistical, machine learning, and deep learning approaches
for robust anomaly identification.

Detection Methods:
    1. StatisticalDetector   — Z-score + IQR on rolling windows
    2. IsolationForestDetector — Multivariate outlier detection
    3. LSTMAutoencoderDetector — Temporal pattern anomaly via reconstruction error
    4. SensorAnomalyDetector — Ensemble combining all three
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────── Data Types ───────────────────────

@dataclass
class SensorAnomaly:
    """A detected sensor anomaly."""
    sensor_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    anomaly_type: str = "unknown"      # spike, drift, flatline, multivariate, temporal
    severity: str = "info"              # info, warning, critical
    confidence: float = 0.0             # 0–1
    value: float = 0.0                  # the anomalous value
    expected_range: tuple[float, float] = (0.0, 0.0)
    method: str = ""                    # which detector flagged it
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sensor_id": self.sensor_id,
            "timestamp": str(self.timestamp),
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "confidence": self.confidence,
            "value": self.value,
            "expected_range": list(self.expected_range),
            "method": self.method,
            "details": self.details,
        }


# ─────────────────────── Statistical Detector ───────────────────────

class StatisticalDetector:
    """
    Z-score + IQR anomaly detection on rolling windows.

    Catches: sudden spikes, flatlines, out-of-range values.
    """

    def __init__(
        self,
        window_size: int = 100,
        z_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
    ):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
        self._buffers: dict[str, deque] = {}

    def detect(self, sensor_id: str, value: float) -> SensorAnomaly | None:
        """Check a single sensor reading for statistical anomalies."""
        if sensor_id not in self._buffers:
            self._buffers[sensor_id] = deque(maxlen=self.window_size)

        buf = self._buffers[sensor_id]
        buf.append(value)

        if len(buf) < 10:
            return None  # need minimum data

        arr = np.array(buf)
        mean = arr.mean()
        std = arr.std()

        # Z-score check
        if std > 1e-8:
            z_score = abs(value - mean) / std
            if z_score > self.z_threshold:
                return SensorAnomaly(
                    sensor_id=sensor_id,
                    anomaly_type="spike",
                    severity="warning" if z_score < 5 else "critical",
                    confidence=min(1.0, z_score / 10.0),
                    value=value,
                    expected_range=(mean - 2 * std, mean + 2 * std),
                    method="z_score",
                    details={"z_score": round(z_score, 2), "mean": round(mean, 2), "std": round(std, 4)},
                )

        # IQR check
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lower = q1 - self.iqr_multiplier * iqr
        upper = q3 + self.iqr_multiplier * iqr

        if value < lower or value > upper:
            return SensorAnomaly(
                sensor_id=sensor_id,
                anomaly_type="outlier",
                severity="warning",
                confidence=0.7,
                value=value,
                expected_range=(lower, upper),
                method="iqr",
                details={"q1": round(q1, 2), "q3": round(q3, 2), "iqr": round(iqr, 2)},
            )

        # Flatline check (zero variance)
        if std < 1e-8 and len(buf) >= 20:
            return SensorAnomaly(
                sensor_id=sensor_id,
                anomaly_type="flatline",
                severity="warning",
                confidence=0.8,
                value=value,
                expected_range=(mean, mean),
                method="flatline",
                details={"constant_value": round(mean, 2), "window_size": len(buf)},
            )

        return None

    def detect_batch(self, sensor_id: str, values: list[float]) -> list[SensorAnomaly]:
        """Check a batch of values and return all anomalies."""
        anomalies = []
        for v in values:
            a = self.detect(sensor_id, v)
            if a:
                anomalies.append(a)
        return anomalies


# ─────────────────────── Isolation Forest Detector ───────────────────────

class IsolationForestDetector:
    """
    Multivariate anomaly detection using Isolation Forest.

    Catches: correlated sensor anomalies (e.g., temp rises but flow doesn't).
    Requires multiple sensor readings at the same timestamp.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._model = None
        self._feature_names: list[str] = []
        self._is_fitted = False
        self._buffer: list[np.ndarray] = []
        self._min_samples = 50

    def fit(self, data: np.ndarray, feature_names: list[str] | None = None) -> None:
        """Fit the Isolation Forest on normal operating data."""
        try:
            from sklearn.ensemble import IsolationForest

            self._model = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            )
            self._model.fit(data)
            self._feature_names = feature_names or [f"sensor_{i}" for i in range(data.shape[1])]
            self._is_fitted = True
            logger.info(f"Isolation Forest fitted on {data.shape[0]} samples, {data.shape[1]} features")
        except ImportError:
            logger.warning("scikit-learn not installed. Isolation Forest unavailable.")

    def detect(self, reading: np.ndarray) -> SensorAnomaly | None:
        """
        Check a single multi-sensor reading for multivariate anomalies.

        Args:
            reading: 1D array of sensor values (one per feature)
        """
        if not self._is_fitted or self._model is None:
            self._buffer.append(reading)
            if len(self._buffer) >= self._min_samples:
                self.fit(np.array(self._buffer))
                self._buffer = []
            return None

        x = reading.reshape(1, -1)
        prediction = self._model.predict(x)
        score = self._model.score_samples(x)[0]

        if prediction[0] == -1:  # anomaly
            # Find which feature contributed most
            anomalous_features = {}
            for i, name in enumerate(self._feature_names):
                anomalous_features[name] = float(reading[i])

            return SensorAnomaly(
                sensor_id="multivariate",
                anomaly_type="multivariate",
                severity="warning" if score > -0.7 else "critical",
                confidence=min(1.0, abs(score)),
                value=float(score),
                method="isolation_forest",
                details={"anomaly_score": round(score, 4), "features": anomalous_features},
            )

        return None


# ─────────────────────── LSTM Autoencoder Detector ───────────────────────

class LSTMAutoencoderDetector:
    """
    LSTM Autoencoder for temporal pattern anomaly detection.

    Trains on normal sensor sequences. High reconstruction error = anomaly.
    Catches: gradual drifts, unusual temporal patterns, degradation.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 32,
        sequence_length: int = 30,
        threshold_percentile: float = 95.0,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.threshold_percentile = threshold_percentile
        self._model = None
        self._threshold: float = 0.0
        self._is_fitted = False
        self._buffers: dict[str, deque] = {}

        self._build_model()

    def _build_model(self):
        """Build the LSTM autoencoder."""
        import torch
        import torch.nn as nn

        class LSTMAutoencoder(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
                self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
                self.output_layer = nn.Linear(hidden_dim, input_dim)

            def forward(self, x):
                # Encode
                _, (h, c) = self.encoder(x)
                # Decode — repeat hidden state across sequence
                seq_len = x.size(1)
                decoder_input = h.squeeze(0).unsqueeze(1).repeat(1, seq_len, 1)
                decoded, _ = self.decoder(decoder_input)
                output = self.output_layer(decoded)
                return output

        self._model = LSTMAutoencoder(self.input_dim, self.hidden_dim)

    def fit(self, data: np.ndarray, epochs: int = 20, lr: float = 1e-3) -> dict[str, list[float]]:
        """
        Train on normal sensor data.

        Args:
            data: 1D array of normal sensor readings
            epochs: training epochs
            lr: learning rate
        """
        import torch
        import torch.optim as optim

        # Create sequences
        sequences = []
        for i in range(len(data) - self.sequence_length):
            seq = data[i:i + self.sequence_length]
            sequences.append(seq)

        if not sequences:
            return {"loss": []}

        X = torch.tensor(np.array(sequences), dtype=torch.float32).unsqueeze(-1)  # [N, seq, 1]

        optimizer = optim.Adam(self._model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
        history = {"loss": []}

        self._model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self._model(X)
            loss = loss_fn(output, X)
            loss.backward()
            optimizer.step()
            history["loss"].append(loss.item())

        # Compute threshold from reconstruction errors
        self._model.eval()
        with torch.no_grad():
            reconstructed = self._model(X)
            errors = torch.mean((X - reconstructed) ** 2, dim=(1, 2)).numpy()
            self._threshold = float(np.percentile(errors, self.threshold_percentile))

        self._is_fitted = True
        logger.info(f"LSTM Autoencoder trained: threshold={self._threshold:.6f}")
        return history

    def detect(self, sensor_id: str, value: float) -> SensorAnomaly | None:
        """Check a sensor reading for temporal anomalies."""
        import torch

        if sensor_id not in self._buffers:
            self._buffers[sensor_id] = deque(maxlen=self.sequence_length)

        buf = self._buffers[sensor_id]
        buf.append(value)

        if len(buf) < self.sequence_length or not self._is_fitted:
            return None

        seq = torch.tensor(list(buf), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

        self._model.eval()
        with torch.no_grad():
            reconstructed = self._model(seq)
            error = torch.mean((seq - reconstructed) ** 2).item()

        if error > self._threshold:
            return SensorAnomaly(
                sensor_id=sensor_id,
                anomaly_type="temporal",
                severity="warning" if error < self._threshold * 3 else "critical",
                confidence=min(1.0, error / (self._threshold * 5)),
                value=value,
                method="lstm_autoencoder",
                details={
                    "reconstruction_error": round(error, 6),
                    "threshold": round(self._threshold, 6),
                    "ratio": round(error / max(self._threshold, 1e-8), 2),
                },
            )

        return None


# ─────────────────────── Ensemble Detector ───────────────────────

class SensorAnomalyDetector:
    """
    Ensemble sensor anomaly detector combining statistical, Isolation Forest,
    and LSTM Autoencoder methods.

    Voting: an anomaly is flagged if at least `min_votes` methods agree.
    The highest-severity result is returned.
    """

    SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}

    def __init__(
        self,
        min_votes: int = 1,
        statistical_config: dict | None = None,
        isolation_config: dict | None = None,
        lstm_config: dict | None = None,
    ):
        self.min_votes = min_votes

        self.statistical = StatisticalDetector(**(statistical_config or {}))
        self.isolation = IsolationForestDetector(**(isolation_config or {}))
        self.lstm = LSTMAutoencoderDetector(**(lstm_config or {}))

    def detect(
        self,
        sensor_id: str,
        value: float,
        multivariate_reading: np.ndarray | None = None,
    ) -> SensorAnomaly | None:
        """
        Run all detectors on a sensor reading and return the consensus anomaly.

        Args:
            sensor_id: sensor identifier
            value: the scalar reading
            multivariate_reading: optional array of all sensor values at this timestamp
        """
        anomalies: list[SensorAnomaly] = []

        # Statistical
        a = self.statistical.detect(sensor_id, value)
        if a:
            anomalies.append(a)

        # Isolation Forest (if multivariate data available)
        if multivariate_reading is not None:
            a = self.isolation.detect(multivariate_reading)
            if a:
                anomalies.append(a)

        # LSTM Autoencoder
        a = self.lstm.detect(sensor_id, value)
        if a:
            anomalies.append(a)

        if len(anomalies) < self.min_votes:
            return None

        # Return the highest-severity anomaly
        best = max(anomalies, key=lambda x: self.SEVERITY_ORDER.get(x.severity, 0))
        best.details["votes"] = len(anomalies)
        best.details["methods"] = [a.method for a in anomalies]
        best.method = "ensemble"
        return best

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SensorAnomalyDetector:
        det_cfg = config.get("anomaly_detection", {}).get("sensor", {})
        return cls(
            min_votes=det_cfg.get("min_votes", 1),
            statistical_config=det_cfg.get("statistical", {}),
            isolation_config=det_cfg.get("isolation_forest", {}),
            lstm_config=det_cfg.get("lstm_autoencoder", {}),
        )

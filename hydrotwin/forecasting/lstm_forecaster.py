"""
HydroTwin OS — Plane 3: LSTM-based IT Load Forecaster

A PyTorch LSTM model for sequence-to-one IT load prediction.
Serves as an alternative to Prophet, enabling ensemble forecasting.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """PyTorch LSTM for time-series forecasting."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take the last timestep's output
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)


class LSTMForecaster:
    """
    LSTM-based IT load forecaster.

    Trains on windowed historical data and produces single-step or
    multi-step (autoregressive) forecasts.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        sequence_length: int = 168,  # 7 days of hourly data
        horizon_hours: int = 24,
        learning_rate: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.horizon_hours = horizon_hours
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: LSTMModel | None = None
        self._is_fitted = False
        self._mean: float = 0.0
        self._std: float = 1.0
        self._last_sequence: np.ndarray | None = None

    def fit(self, values: list | np.ndarray) -> dict[str, list[float]]:
        """
        Train the LSTM on historical IT load data.

        Args:
            values: 1D array of IT load values (kW), hourly resolution.

        Returns:
            Training history with loss per epoch.
        """
        values = np.array(values, dtype=np.float32)

        # Normalize
        self._mean = float(values.mean())
        self._std = float(values.std()) + 1e-8
        normalized = (values - self._mean) / self._std

        # Create sliding windows
        X, y = [], []
        for i in range(len(normalized) - self.sequence_length):
            X.append(normalized[i : i + self.sequence_length])
            y.append(normalized[i + self.sequence_length])

        X = np.array(X, dtype=np.float32).reshape(-1, self.sequence_length, 1)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)

        dataset = TensorDataset(
            torch.from_numpy(X).to(self.device),
            torch.from_numpy(y).to(self.device),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Build model
        self.model = LSTMModel(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        # Train
        history = {"loss": []}
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                pred = self.model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            history["loss"].append(avg_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(f"LSTM Epoch {epoch+1}/{self.epochs} | Loss: {avg_loss:.6f}")

        # Store last sequence for autoregressive prediction
        self._last_sequence = normalized[-self.sequence_length:]
        self._is_fitted = True
        logger.info(f"LSTM fitted on {len(values)} data points")

        return history

    def predict(self, periods: int | None = None) -> dict[str, Any]:
        """
        Generate autoregressive forecast.

        Args:
            periods: Number of hourly periods to forecast.

        Returns:
            Dict with 'forecast' array and metadata.
        """
        if not self._is_fitted or self.model is None or self._last_sequence is None:
            return self._fallback_forecast(periods or self.horizon_hours)

        n_periods = periods or self.horizon_hours
        self.model.eval()

        sequence = self._last_sequence.copy()
        predictions = []

        with torch.no_grad():
            for _ in range(n_periods):
                x = torch.from_numpy(
                    sequence.reshape(1, -1, 1).astype(np.float32)
                ).to(self.device)
                pred = self.model(x).cpu().numpy().flatten()[0]
                predictions.append(pred)
                sequence = np.append(sequence[1:], pred)

        # Denormalize
        forecast = np.array(predictions) * self._std + self._mean
        forecast = np.clip(forecast, 0, None)  # load can't be negative

        # Simple confidence interval (±1 std of residuals, approximated)
        uncertainty = self._std * 0.1 * np.arange(1, n_periods + 1) ** 0.5

        return {
            "forecast": forecast.tolist(),
            "lower": (forecast - uncertainty).tolist(),
            "upper": (forecast + uncertainty).tolist(),
            "model": "lstm",
        }

    def _fallback_forecast(self, periods: int) -> dict[str, Any]:
        """Fallback when model is not trained."""
        hours = np.arange(periods)
        base = 5000.0
        daily = 1200.0 * np.sin(2 * np.pi * (hours - 14) / 24)
        forecast = base + daily + np.random.normal(0, 150, periods)
        forecast = np.clip(forecast, 2000, 10000)

        return {
            "forecast": forecast.tolist(),
            "lower": (forecast - 600).tolist(),
            "upper": (forecast + 600).tolist(),
            "model": "fallback_sinusoidal",
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> LSTMForecaster:
        lstm_cfg = config.get("forecasting", {}).get("lstm", {})
        return cls(
            hidden_size=lstm_cfg.get("hidden_size", 64),
            num_layers=lstm_cfg.get("num_layers", 2),
            dropout=lstm_cfg.get("dropout", 0.2),
            sequence_length=lstm_cfg.get("sequence_length", 168),
            horizon_hours=lstm_cfg.get("horizon_hours", 24),
        )

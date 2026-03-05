"""
HydroTwin OS — Plane 3: Ensemble Forecaster

Combines Prophet and LSTM forecasts using weighted averaging.
Provides a more robust prediction by merging the strengths of
both models — Prophet's interpretable seasonality and LSTM's
ability to capture non-linear patterns.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from hydrotwin.forecasting.prophet_forecaster import ProphetForecaster
from hydrotwin.forecasting.lstm_forecaster import LSTMForecaster

logger = logging.getLogger(__name__)


class EnsembleForecaster:
    """
    Weighted ensemble combining Prophet and LSTM predictions.

    The ensemble blends forecasts with configurable weights and computes
    merged confidence intervals.
    """

    def __init__(
        self,
        prophet: ProphetForecaster | None = None,
        lstm: LSTMForecaster | None = None,
        prophet_weight: float = 0.5,
        lstm_weight: float = 0.5,
    ):
        self.prophet = prophet or ProphetForecaster()
        self.lstm = lstm or LSTMForecaster()

        # Normalize weights
        total = prophet_weight + lstm_weight
        self.prophet_weight = prophet_weight / total
        self.lstm_weight = lstm_weight / total

    def fit(
        self,
        timestamps: list | np.ndarray,
        values: list | np.ndarray,
    ) -> None:
        """Fit both models on the same historical data."""
        logger.info("Fitting ensemble forecasters...")
        self.prophet.fit(timestamps, values)
        self.lstm.fit(values)
        logger.info("Ensemble fitted.")

    def predict(self, periods: int | None = None) -> dict[str, Any]:
        """
        Generate ensemble forecast.

        Weighted average of Prophet and LSTM predictions, with merged
        confidence intervals.
        """
        prophet_pred = self.prophet.predict(periods)
        lstm_pred = self.lstm.predict(periods)

        # Align lengths (use shortest)
        n = min(len(prophet_pred["forecast"]), len(lstm_pred["forecast"]))
        p_forecast = np.array(prophet_pred["forecast"][:n])
        l_forecast = np.array(lstm_pred["forecast"][:n])

        # Weighted average
        ensemble_forecast = self.prophet_weight * p_forecast + self.lstm_weight * l_forecast

        # Merge confidence intervals
        p_lower = np.array(prophet_pred["lower"][:n])
        p_upper = np.array(prophet_pred["upper"][:n])
        l_lower = np.array(lstm_pred["lower"][:n])
        l_upper = np.array(lstm_pred["upper"][:n])

        ensemble_lower = np.minimum(p_lower, l_lower)
        ensemble_upper = np.maximum(p_upper, l_upper)

        return {
            "forecast": ensemble_forecast.tolist(),
            "lower": ensemble_lower.tolist(),
            "upper": ensemble_upper.tolist(),
            "model": "ensemble",
            "component_models": {
                "prophet": {
                    "model": prophet_pred.get("model"),
                    "weight": self.prophet_weight,
                },
                "lstm": {
                    "model": lstm_pred.get("model"),
                    "weight": self.lstm_weight,
                },
            },
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> EnsembleForecaster:
        ensemble_cfg = config.get("forecasting", {}).get("ensemble", {})
        return cls(
            prophet=ProphetForecaster.from_config(config),
            lstm=LSTMForecaster.from_config(config),
            prophet_weight=ensemble_cfg.get("prophet_weight", 0.5),
            lstm_weight=ensemble_cfg.get("lstm_weight", 0.5),
        )

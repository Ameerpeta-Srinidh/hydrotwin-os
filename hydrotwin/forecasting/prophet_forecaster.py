"""
HydroTwin OS — Plane 3: Prophet-based IT Load Forecaster

Uses Meta's Prophet for time-series forecasting of IT load with
daily and weekly seasonality. Provides a planning horizon for the RL agent.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ProphetForecaster:
    """
    Prophet-based forecaster for IT load time series.

    Captures daily patterns (high during business hours, low at night)
    and weekly patterns (weekday vs weekend), producing forecasts that
    give the RL agent a look-ahead planning horizon.
    """

    def __init__(
        self,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        daily_seasonality: bool = True,
        weekly_seasonality: bool = True,
        horizon_hours: int = 24,
    ):
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.horizon_hours = horizon_hours
        self._model = None
        self._is_fitted = False

    def fit(self, timestamps: list | np.ndarray, values: list | np.ndarray) -> None:
        """
        Fit Prophet model on historical IT load data.

        Args:
            timestamps: Array of datetime timestamps
            values: Array of IT load values (kW)
        """
        try:
            from prophet import Prophet
        except ImportError:
            logger.warning("Prophet not installed. Using fallback forecaster.")
            self._is_fitted = False
            return

        df = pd.DataFrame({"ds": pd.to_datetime(timestamps), "y": values})

        self._model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            daily_seasonality=self.daily_seasonality,
            weekly_seasonality=self.weekly_seasonality,
        )
        self._model.fit(df)
        self._is_fitted = True
        logger.info(f"Prophet fitted on {len(df)} data points")

    def predict(self, periods: int | None = None) -> dict[str, Any]:
        """
        Generate forecast for the specified horizon.

        Args:
            periods: Number of hourly periods to forecast. Defaults to horizon_hours.

        Returns:
            Dict with 'timestamps', 'forecast', 'lower', 'upper' arrays.
        """
        if not self._is_fitted or self._model is None:
            return self._fallback_forecast(periods or self.horizon_hours)

        n_periods = periods or self.horizon_hours
        future = self._model.make_future_dataframe(periods=n_periods, freq="h")
        forecast = self._model.predict(future)

        # Return only the future predictions
        future_mask = forecast.index >= (len(forecast) - n_periods)
        pred = forecast[future_mask]

        return {
            "timestamps": pred["ds"].tolist(),
            "forecast": pred["yhat"].values.tolist(),
            "lower": pred["yhat_lower"].values.tolist(),
            "upper": pred["yhat_upper"].values.tolist(),
            "model": "prophet",
        }

    def _fallback_forecast(self, periods: int) -> dict[str, Any]:
        """Simple sinusoidal fallback when Prophet is not available."""
        hours = np.arange(periods)
        base = 5000.0
        daily = 1500.0 * np.sin(2 * np.pi * (hours - 13) / 24)
        forecast = base + daily + np.random.normal(0, 100, periods)
        forecast = np.clip(forecast, 2000, 10000)

        return {
            "timestamps": list(range(periods)),
            "forecast": forecast.tolist(),
            "lower": (forecast - 500).tolist(),
            "upper": (forecast + 500).tolist(),
            "model": "fallback_sinusoidal",
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ProphetForecaster:
        prophet_cfg = config.get("forecasting", {}).get("prophet", {})
        return cls(
            changepoint_prior_scale=prophet_cfg.get("changepoint_prior_scale", 0.05),
            seasonality_prior_scale=prophet_cfg.get("seasonality_prior_scale", 10.0),
            daily_seasonality=prophet_cfg.get("daily_seasonality", True),
            weekly_seasonality=prophet_cfg.get("weekly_seasonality", True),
            horizon_hours=prophet_cfg.get("horizon_hours", 24),
        )

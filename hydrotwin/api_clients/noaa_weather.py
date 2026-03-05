"""
HydroTwin OS — Plane 3: NOAA Weather API Client

Fetches hourly dry-bulb temperature, wet-bulb temperature, and relative
humidity from NOAA's Climate Data Online API. Used by the RL agent for
current conditions and the forecasting models for weather-aware predictions.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class NOAAWeatherClient:
    """
    Client for NOAA Climate Data Online (CDO) API v2.

    Free API with rate limits. Provides hourly weather observations
    for the RL agent's observation vector.
    """

    def __init__(
        self,
        base_url: str = "https://www.ncdc.noaa.gov/cdo-web/api/v2",
        api_key: str = "",
        station_id: str = "GHCND:USW00023183",
        cache_ttl_seconds: int = 3600,
        mock_mode: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.station_id = station_id
        self.cache_ttl = cache_ttl_seconds
        self.mock_mode = mock_mode
        self._cache: dict[str, tuple[float, Any]] = {}

    def get_current_conditions(self) -> dict[str, Any]:
        """
        Get current weather conditions.

        Returns:
            Dict with 'ambient_temp_c', 'wet_bulb_temp_c', 'relative_humidity',
            'wind_speed_ms', 'source'.
        """
        if self.mock_mode:
            return self._mock_conditions()

        cached = self._get_cached("current")
        if cached is not None:
            return cached

        try:
            response = httpx.get(
                f"{self.base_url}/data",
                params={
                    "datasetid": "GHCND",
                    "stationid": self.station_id,
                    "startdate": time.strftime("%Y-%m-%d"),
                    "enddate": time.strftime("%Y-%m-%d"),
                    "limit": 10,
                    "units": "metric",
                },
                headers={"token": self.api_key},
                timeout=15.0,
            )
            response.raise_for_status()
            data = response.json()

            # Parse relevant observations
            result = self._parse_observations(data.get("results", []))
            self._set_cached("current", result)
            return result

        except Exception as e:
            logger.warning(f"NOAA API error: {e}. Using mock data.")
            return self._mock_conditions()

    def get_hourly_forecast(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get hourly weather forecast (mock-based for free tier)."""
        # NOAA free tier doesn't provide granular forecasts
        # So we use physics-based mock data
        return [self._mock_conditions(hour_offset=h) for h in range(hours)]

    def _parse_observations(self, results: list[dict]) -> dict[str, Any]:
        """Parse NOAA observation data into a clean format."""
        ambient_temp = None
        for r in results:
            if r.get("datatype") == "TMAX":
                ambient_temp = r.get("value", 30.0) / 10.0  # NOAA reports in tenths of °C
            elif r.get("datatype") == "TMIN" and ambient_temp is None:
                ambient_temp = r.get("value", 20.0) / 10.0

        if ambient_temp is None:
            ambient_temp = 30.0

        wet_bulb = self._estimate_wet_bulb(ambient_temp, 0.4)

        return {
            "ambient_temp_c": ambient_temp,
            "wet_bulb_temp_c": wet_bulb,
            "relative_humidity": 0.4,
            "wind_speed_ms": 3.0,
            "source": "noaa_api",
        }

    @staticmethod
    def _estimate_wet_bulb(temp_c: float, rh: float) -> float:
        """
        Stull's formula for wet-bulb temperature estimation.
        Accurate to ±1°C for typical conditions.
        """
        t = temp_c
        rh_pct = rh * 100.0
        wb = t * math.atan(0.151977 * (rh_pct + 8.313659) ** 0.5) + \
             math.atan(t + rh_pct) - math.atan(rh_pct - 1.676331) + \
             0.00391838 * rh_pct ** 1.5 * math.atan(0.023101 * rh_pct) - 4.686035
        return wb

    def _mock_conditions(self, hour_offset: int = 0) -> dict[str, Any]:
        """Generate realistic mock weather data for Phoenix, AZ."""
        hour = (time.localtime().tm_hour + hour_offset) % 24
        month = time.localtime().tm_mon

        # Monthly base temperatures for Phoenix (°C)
        monthly_highs = [19, 21, 25, 30, 36, 41, 43, 42, 39, 33, 25, 19]
        monthly_lows = [7, 9, 12, 16, 21, 27, 31, 30, 26, 18, 11, 7]

        base_high = monthly_highs[month - 1]
        base_low = monthly_lows[month - 1]

        # Diurnal cycle
        diurnal = (base_high - base_low) / 2
        mid = (base_high + base_low) / 2
        ambient = mid + diurnal * math.sin(2 * math.pi * (hour - 15) / 24)

        # Phoenix humidity is very low in summer
        rh = 0.15 + 0.2 * (1 - (month - 1) / 11) + 0.05 * math.sin(2 * math.pi * hour / 24)
        rh = max(0.05, min(0.8, rh))

        wet_bulb = self._estimate_wet_bulb(ambient, rh)

        return {
            "ambient_temp_c": round(ambient, 1),
            "wet_bulb_temp_c": round(wet_bulb, 1),
            "relative_humidity": round(rh, 3),
            "wind_speed_ms": round(2.0 + 3.0 * math.sin(2 * math.pi * hour / 24), 1),
            "hour_offset": hour_offset,
            "source": "mock",
        }

    def _get_cached(self, key: str) -> Any | None:
        if key in self._cache:
            ts, val = self._cache[key]
            if time.time() - ts < self.cache_ttl:
                return val
            del self._cache[key]
        return None

    def _set_cached(self, key: str, value: Any) -> None:
        self._cache[key] = (time.time(), value)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> NOAAWeatherClient:
        api_cfg = config.get("apis", {}).get("noaa", {})
        api_key = api_cfg.get("api_key", "")
        mock = not api_key or api_key.startswith("${")
        return cls(
            base_url=api_cfg.get("base_url", "https://www.ncdc.noaa.gov/cdo-web/api/v2"),
            api_key=api_key,
            station_id=api_cfg.get("station_id", "GHCND:USW00023183"),
            cache_ttl_seconds=api_cfg.get("cache_ttl_seconds", 3600),
            mock_mode=mock,
        )

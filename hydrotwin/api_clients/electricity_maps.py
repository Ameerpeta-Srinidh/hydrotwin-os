"""
HydroTwin OS — Plane 3: ElectricityMaps API Client

Fetches real-time and forecast grid carbon intensity per zone.
Includes in-memory caching and mock fallback for offline development.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class ElectricityMapsClient:
    """
    Client for ElectricityMaps API v3 (free tier).

    Provides real-time marginal carbon intensity (gCO₂eq/kWh) for
    electricity grid zones worldwide. Used by the RL agent and
    the dynamic weight adjuster.
    """

    def __init__(
        self,
        base_url: str = "https://api.electricitymap.org/v3",
        api_key: str = "",
        default_zone: str = "US-SW-SRP",
        cache_ttl_seconds: int = 300,
        mock_mode: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_zone = default_zone
        self.cache_ttl = cache_ttl_seconds
        self.mock_mode = mock_mode

        self._cache: dict[str, tuple[float, Any]] = {}

    def get_carbon_intensity(self, zone: str | None = None) -> dict[str, Any]:
        """
        Get current carbon intensity for a grid zone.

        Returns:
            Dict with 'carbon_intensity' (gCO₂eq/kWh), 'zone', 'datetime',
            'fossil_fuel_percentage', 'source'.
        """
        zone = zone or self.default_zone

        if self.mock_mode:
            return self._mock_intensity(zone)

        cache_key = f"intensity:{zone}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            response = httpx.get(
                f"{self.base_url}/carbon-intensity/latest",
                params={"zone": zone},
                headers={"auth-token": self.api_key},
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()

            result = {
                "carbon_intensity": data.get("carbonIntensity", 200.0),
                "zone": zone,
                "datetime": data.get("datetime"),
                "fossil_fuel_percentage": data.get("fossilFuelPercentage", 50),
                "source": "electricity_maps_api",
            }
            self._set_cached(cache_key, result)
            return result

        except Exception as e:
            logger.warning(f"ElectricityMaps API error for zone {zone}: {e}. Using fallback.")
            return self._mock_intensity(zone)

    def get_multi_region_intensity(
        self, zones: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get carbon intensity for multiple zones (for migration decisions)."""
        zones = zones or [self.default_zone]
        return [self.get_carbon_intensity(z) for z in zones]

    def get_forecast(self, zone: str | None = None) -> dict[str, Any]:
        """Get carbon intensity forecast for next 24 hours."""
        zone = zone or self.default_zone

        if self.mock_mode:
            return self._mock_forecast(zone)

        cache_key = f"forecast:{zone}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            response = httpx.get(
                f"{self.base_url}/carbon-intensity/forecast",
                params={"zone": zone},
                headers={"auth-token": self.api_key},
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()

            result = {
                "zone": zone,
                "forecast": data.get("forecast", []),
                "source": "electricity_maps_api",
            }
            self._set_cached(cache_key, result)
            return result

        except Exception as e:
            logger.warning(f"ElectricityMaps forecast error: {e}. Using fallback.")
            return self._mock_forecast(zone)

    def _mock_intensity(self, zone: str) -> dict[str, Any]:
        """Mock carbon intensity data for offline development."""
        import math
        hour = time.localtime().tm_hour
        # Simulate diurnal variation
        base = {"US-SW-SRP": 350, "US-NW-PACW": 80, "US-MIDA-PJM": 280, "CA-QC": 30, "SE": 25}
        intensity = base.get(zone, 200) + 50 * math.sin(2 * math.pi * (hour - 18) / 24)
        return {
            "carbon_intensity": max(10, intensity),
            "zone": zone,
            "datetime": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "fossil_fuel_percentage": min(90, max(5, intensity / 8)),
            "source": "mock",
        }

    def _mock_forecast(self, zone: str) -> dict[str, Any]:
        import math
        hour = time.localtime().tm_hour
        base = {"US-SW-SRP": 350, "US-NW-PACW": 80, "US-MIDA-PJM": 280, "CA-QC": 30, "SE": 25}
        b = base.get(zone, 200)
        forecast = []
        for h in range(24):
            t = hour + h
            intensity = b + 50 * math.sin(2 * math.pi * (t - 18) / 24)
            forecast.append({"datetime": f"T+{h}h", "carbonIntensity": max(10, intensity)})
        return {"zone": zone, "forecast": forecast, "source": "mock"}

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
    def from_config(cls, config: dict[str, Any]) -> ElectricityMapsClient:
        api_cfg = config.get("apis", {}).get("electricity_maps", {})
        api_key = api_cfg.get("api_key", "")
        mock = not api_key or api_key.startswith("${")
        return cls(
            base_url=api_cfg.get("base_url", "https://api.electricitymap.org/v3"),
            api_key=api_key,
            default_zone=api_cfg.get("default_zone", "US-SW-SRP"),
            cache_ttl_seconds=api_cfg.get("cache_ttl_seconds", 300),
            mock_mode=mock,
        )

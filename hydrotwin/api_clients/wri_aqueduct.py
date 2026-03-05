"""
HydroTwin OS — Plane 3: WRI Aqueduct API Client

Fetches water stress index by geographic coordinates from the
World Resources Institute Aqueduct water risk framework.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


# WRI Aqueduct water stress scale:
# 0 = Low (<10%), 1 = Low-Medium (10-20%), 2 = Medium-High (20-40%),
# 3 = High (40-80%), 4 = Extremely High (>80%), 5 = Arid / Low water use

_KNOWN_STRESS_LEVELS: dict[str, float] = {
    # Major data center locations and their approximate water stress
    "phoenix_az": 4.2,
    "las_vegas_nv": 4.5,
    "dallas_tx": 2.8,
    "chicago_il": 1.2,
    "portland_or": 0.8,
    "ashburn_va": 1.5,
    "montreal_qc": 0.5,
    "stockholm_se": 0.3,
    "singapore_sg": 2.0,
    "chennai_in": 4.8,
    "hyderabad_in": 4.0,
    "mumbai_in": 3.5,
}


class WRIAqueductClient:
    """
    Client for WRI Aqueduct water risk data.

    Provides water stress index (0-5 scale) for geographic locations.
    Used by the RL agent and migration engine to factor water availability
    into cooling and migration decisions.
    """

    def __init__(
        self,
        base_url: str = "https://api.resourcewatch.org/v1",
        cache_ttl_seconds: int = 86400,
        mock_mode: bool = True,
        default_latitude: float = 33.4484,
        default_longitude: float = -112.0740,
    ):
        self.base_url = base_url.rstrip("/")
        self.cache_ttl = cache_ttl_seconds
        self.mock_mode = mock_mode
        self.default_lat = default_latitude
        self.default_lon = default_longitude
        self._cache: dict[str, tuple[float, Any]] = {}

    def get_water_stress(
        self,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> dict[str, Any]:
        """
        Get water stress index for a location.

        Returns:
            Dict with 'water_stress_index' (0-5), 'risk_label',
            'latitude', 'longitude', 'source'.
        """
        lat = latitude or self.default_lat
        lon = longitude or self.default_lon

        if self.mock_mode:
            return self._mock_stress(lat, lon)

        cache_key = f"stress:{lat:.2f}:{lon:.2f}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            # WRI Aqueduct query via CARTO endpoint
            response = httpx.get(
                f"{self.base_url}/query",
                params={
                    "sql": (
                        f"SELECT bws_raw, bws_label FROM aqueduct30 "
                        f"WHERE ST_Contains(the_geom, ST_Point({lon}, {lat}))"
                    ),
                },
                timeout=15.0,
            )
            response.raise_for_status()
            data = response.json()
            rows = data.get("data", [])

            if rows:
                row = rows[0]
                stress = min(5.0, max(0.0, row.get("bws_raw", 2.0)))
                label = row.get("bws_label", self._stress_label(stress))
            else:
                stress = 2.0
                label = "Medium-High"

            result = {
                "water_stress_index": stress,
                "risk_label": label,
                "latitude": lat,
                "longitude": lon,
                "source": "wri_aqueduct_api",
            }
            self._set_cached(cache_key, result)
            return result

        except Exception as e:
            logger.warning(f"WRI Aqueduct API error: {e}. Using mock data.")
            return self._mock_stress(lat, lon)

    def get_stress_for_regions(
        self, regions: list[dict[str, float]],
    ) -> list[dict[str, Any]]:
        """
        Get water stress for multiple regions.

        Args:
            regions: List of dicts with 'latitude' and 'longitude' keys.
        """
        return [
            self.get_water_stress(r.get("latitude"), r.get("longitude"))
            for r in regions
        ]

    def _mock_stress(self, lat: float, lon: float) -> dict[str, Any]:
        """Estimate water stress based on known locations and geography."""
        # Find nearest known location
        best_match = "phoenix_az"
        best_dist = float("inf")

        location_coords = {
            "phoenix_az": (33.45, -112.07),
            "las_vegas_nv": (36.17, -115.14),
            "dallas_tx": (32.78, -96.80),
            "chicago_il": (41.88, -87.63),
            "portland_or": (45.52, -122.68),
            "ashburn_va": (39.04, -77.49),
            "montreal_qc": (45.50, -73.57),
            "stockholm_se": (59.33, 18.07),
            "chennai_in": (13.08, 80.27),
            "hyderabad_in": (17.39, 78.49),
        }

        for name, (rlat, rlon) in location_coords.items():
            dist = ((lat - rlat) ** 2 + (lon - rlon) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_match = name

        stress = _KNOWN_STRESS_LEVELS.get(best_match, 2.0)

        return {
            "water_stress_index": stress,
            "risk_label": self._stress_label(stress),
            "latitude": lat,
            "longitude": lon,
            "nearest_known": best_match,
            "source": "mock",
        }

    @staticmethod
    def _stress_label(index: float) -> str:
        if index < 1:
            return "Low"
        elif index < 2:
            return "Low-Medium"
        elif index < 3:
            return "Medium-High"
        elif index < 4:
            return "High"
        else:
            return "Extremely High"

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
    def from_config(cls, config: dict[str, Any]) -> WRIAqueductClient:
        api_cfg = config.get("apis", {}).get("wri_aqueduct", {})
        facility = config.get("facility", {}).get("location", {})
        return cls(
            base_url=api_cfg.get("base_url", "https://api.resourcewatch.org/v1"),
            cache_ttl_seconds=api_cfg.get("cache_ttl_seconds", 86400),
            mock_mode=True,  # WRI free tier is limited; default to mock
            default_latitude=facility.get("latitude", 33.4484),
            default_longitude=facility.get("longitude", -112.0740),
        )

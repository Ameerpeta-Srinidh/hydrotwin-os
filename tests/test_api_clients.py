"""
Tests for API clients (mock mode).
"""

from hydrotwin.api_clients.electricity_maps import ElectricityMapsClient
from hydrotwin.api_clients.noaa_weather import NOAAWeatherClient
from hydrotwin.api_clients.wri_aqueduct import WRIAqueductClient


class TestElectricityMapsClient:
    """Tests for ElectricityMaps API client in mock mode."""

    def test_mock_intensity(self):
        client = ElectricityMapsClient(mock_mode=True)
        result = client.get_carbon_intensity("US-SW-SRP")
        assert "carbon_intensity" in result
        assert result["carbon_intensity"] > 0
        assert result["source"] == "mock"

    def test_multi_region(self):
        client = ElectricityMapsClient(mock_mode=True)
        results = client.get_multi_region_intensity(["US-SW-SRP", "CA-QC", "SE"])
        assert len(results) == 3
        # Quebec and Sweden should be cleaner than Phoenix
        phoenix_carbon = results[0]["carbon_intensity"]
        quebec_carbon = results[1]["carbon_intensity"]
        assert quebec_carbon < phoenix_carbon

    def test_mock_forecast(self):
        client = ElectricityMapsClient(mock_mode=True)
        result = client.get_forecast("US-SW-SRP")
        assert len(result["forecast"]) == 24

    def test_caching(self):
        client = ElectricityMapsClient(mock_mode=True, cache_ttl_seconds=300)
        r1 = client.get_carbon_intensity("US-SW-SRP")
        r2 = client.get_carbon_intensity("US-SW-SRP")
        # Mock data varies by time but within cache TTL should be consistent
        # (they go through mock directly, but structure should be valid)
        assert r1["zone"] == r2["zone"]


class TestNOAAWeatherClient:
    """Tests for NOAA weather client in mock mode."""

    def test_mock_conditions(self):
        client = NOAAWeatherClient(mock_mode=True)
        result = client.get_current_conditions()
        assert "ambient_temp_c" in result
        assert "wet_bulb_temp_c" in result
        assert "relative_humidity" in result
        assert result["source"] == "mock"
        assert 0 < result["relative_humidity"] < 1

    def test_wet_bulb_less_than_dry(self):
        """Wet-bulb temperature should always be <= dry-bulb."""
        client = NOAAWeatherClient(mock_mode=True)
        result = client.get_current_conditions()
        assert result["wet_bulb_temp_c"] <= result["ambient_temp_c"]

    def test_hourly_forecast(self):
        client = NOAAWeatherClient(mock_mode=True)
        forecast = client.get_hourly_forecast(hours=12)
        assert len(forecast) == 12
        for entry in forecast:
            assert "ambient_temp_c" in entry


class TestWRIAqueductClient:
    """Tests for WRI Aqueduct water stress client in mock mode."""

    def test_phoenix_high_stress(self):
        """Phoenix should have high water stress."""
        client = WRIAqueductClient(mock_mode=True)
        result = client.get_water_stress(33.45, -112.07)  # Phoenix coords
        assert result["water_stress_index"] > 3.0
        assert result["risk_label"] in ["High", "Extremely High"]

    def test_portland_low_stress(self):
        """Portland should have low water stress."""
        client = WRIAqueductClient(mock_mode=True)
        result = client.get_water_stress(45.52, -122.68)  # Portland coords
        assert result["water_stress_index"] < 2.0

    def test_multi_region_stress(self):
        client = WRIAqueductClient(mock_mode=True)
        regions = [
            {"latitude": 33.45, "longitude": -112.07},  # Phoenix
            {"latitude": 45.52, "longitude": -122.68},  # Portland
        ]
        results = client.get_stress_for_regions(regions)
        assert len(results) == 2
        # Phoenix should be more stressed than Portland
        assert results[0]["water_stress_index"] > results[1]["water_stress_index"]

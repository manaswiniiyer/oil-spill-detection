"""Weather provider boundary with deterministic development data."""

from typing import Protocol


class WeatherService(Protocol):
    def get_wind(self, latitude: float, longitude: float, timestamp: str | None) -> dict[str, float]: ...


class MockWeatherService:
    def get_wind(self, latitude: float, longitude: float, timestamp: str | None) -> dict[str, float]:
        return {"wind_u": 4.2, "wind_v": -1.8}
"""Ocean-current provider boundary with deterministic development data."""

from typing import Protocol


class OceanService(Protocol):
    def get_current(self, latitude: float, longitude: float, timestamp: str | None) -> dict[str, float]: ...


class MockOceanService:
    def get_current(self, latitude: float, longitude: float, timestamp: str | None) -> dict[str, float]:
        return {"u_current": 0.18, "v_current": 0.07}
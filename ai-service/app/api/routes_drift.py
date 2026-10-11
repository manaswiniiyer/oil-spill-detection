"""Backward drift endpoint using interchangeable weather and ocean providers."""

import math

from fastapi import APIRouter
from pydantic import BaseModel

from app.services.ocean_service import MockOceanService
from app.services.weather_service import MockWeatherService

router = APIRouter(prefix="/api/v1/drift", tags=["drift"])
weather_service = MockWeatherService()
ocean_service = MockOceanService()


class HindcastRequest(BaseModel):
    latitude: float = 18.5214
    longitude: float = 72.8422
    timestamp: str | None = None
    elapsed_hours: float = 10.0
    wind_u: float | None = None
    wind_v: float | None = None
    u_current: float | None = None
    v_current: float | None = None


@router.post("/hindcast")
def hindcast(request: HindcastRequest) -> dict:
    wind = weather_service.get_wind(request.latitude, request.longitude, request.timestamp)
    current = ocean_service.get_current(request.latitude, request.longitude, request.timestamp)
    velocity_u = (request.wind_u if request.wind_u is not None else wind["wind_u"]) * 0.03 + (
        request.u_current if request.u_current is not None else current["u_current"]
    )
    velocity_v = (request.wind_v if request.wind_v is not None else wind["wind_v"]) * 0.03 + (
        request.v_current if request.v_current is not None else current["v_current"]
    )
    elapsed_seconds = request.elapsed_hours * 3600
    origin_latitude = request.latitude - velocity_v * elapsed_seconds / 111_000
    longitude_scale = max(0.01, abs(math.cos(math.radians(request.latitude))))
    origin_longitude = request.longitude - velocity_u * elapsed_seconds / (111_000 * longitude_scale)
    return {
        "success": True,
        "origin": {"latitude": round(origin_latitude, 5), "longitude": round(origin_longitude, 5)},
        "elapsed_hours": request.elapsed_hours,
        "wind": {"wind_u": wind["wind_u"], "wind_v": wind["wind_v"]},
        "current": {
            "u_current": request.u_current if request.u_current is not None else current["u_current"],
            "v_current": request.v_current if request.v_current is not None else current["v_current"],
        },
        "data_source": "sample_weather+sample_ocean",
        "model": "sample linear backtrack",
        "is_demo": True,
    }
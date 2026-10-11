"""AIS vessel analysis endpoint backed by an interchangeable AIS provider."""

from fastapi import APIRouter
from pydantic import BaseModel

from app.services.ais_service import MockAISService

router = APIRouter(prefix="/api/v1/vessels", tags=["vessels"])
ais_service = MockAISService()


class VesselAnalysisRequest(BaseModel):
    latitude: float
    longitude: float
    timestamp: str | None = None
    radius_km: float = 50


@router.post("/analyze")
def analyze_vessels(request: VesselAnalysisRequest) -> dict:
    vessels = ais_service.get_vessels(request.latitude, request.longitude, request.timestamp)
    return {
        "success": True,
        "vessels": vessels,
        "candidate_count": len({vessel["mmsi"] for vessel in vessels}),
        "search_origin": {"latitude": request.latitude, "longitude": request.longitude},
        "radius_km": request.radius_km,
        "data_source": "sample_ais",
        "is_demo": True,
        "disclaimer": "Sample AIS records are synthetic and do not attribute responsibility.",
    }
"""FastAPI application and route registration for OilTrace."""

from fastapi import FastAPI

from app.api.routes_detection import router as detection_router
from app.api.routes_drift import router as drift_router
from app.api.routes_vessels import router as vessels_router

app = FastAPI(title="OilTrace API", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "operational", "engine": "OilTrace Python Vision Service"}


app.include_router(detection_router)
app.include_router(drift_router)
app.include_router(vessels_router)
"""Versioned spill-detection endpoints and legacy inference compatibility."""

import base64
from uuid import uuid4

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.satellite_service import MockSatelliteService
from inference import predict
from preprocessing import decode_image, preprocess

router = APIRouter()
satellite_service = MockSatelliteService()


class DetectionRequest(BaseModel):
    image_base64: str = Field(min_length=1)
    latitude: float | None = None
    longitude: float | None = None
    timestamp: str | None = None
    scene_area_km2: float | None = Field(default=None, gt=0)
    ground_sample_distance_m: float | None = Field(default=None, gt=0)


def _georeference_mask(
    mask_base64: str,
    image_width: int,
    image_height: int,
    latitude: float,
    longitude: float,
    ground_sample_distance_m: float,
) -> tuple[dict, dict, float]:
    mask_bytes = base64.b64decode(mask_base64, validate=True)
    mask = cv2.imdecode(np.frombuffer(mask_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("Unable to decode predicted segmentation mask")
    binary_mask = (mask > 0).astype(np.uint8)
    mask_height, mask_width = binary_mask.shape
    pixel_scale_x = image_width / mask_width
    pixel_scale_y = image_height / mask_height
    meters_per_longitude_degree = max(1.0, 111_320 * abs(np.cos(np.radians(latitude))))

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        simplified = cv2.approxPolyDP(contour, 1.0, True).reshape(-1, 2)
        ring = []
        for pixel_x, pixel_y in simplified:
            point_latitude = latitude - ((float(pixel_y) + 0.5 - mask_height / 2) * pixel_scale_y * ground_sample_distance_m) / 111_320
            point_longitude = longitude + ((float(pixel_x) + 0.5 - mask_width / 2) * pixel_scale_x * ground_sample_distance_m) / meters_per_longitude_degree
            ring.append([round(point_longitude, 7), round(point_latitude, 7)])
        if ring:
            ring.append(ring[0])
            polygons.append([ring])

    pixel_area_km2 = float(binary_mask.sum()) * pixel_scale_x * pixel_scale_y * ground_sample_distance_m**2 / 1_000_000
    footprint_half_width_m = image_width * ground_sample_distance_m / 2
    footprint_half_height_m = image_height * ground_sample_distance_m / 2
    footprint = [
        [
            [round(longitude - footprint_half_width_m / meters_per_longitude_degree, 7), round(latitude + footprint_half_height_m / 111_320, 7)],
            [round(longitude + footprint_half_width_m / meters_per_longitude_degree, 7), round(latitude + footprint_half_height_m / 111_320, 7)],
            [round(longitude + footprint_half_width_m / meters_per_longitude_degree, 7), round(latitude - footprint_half_height_m / 111_320, 7)],
            [round(longitude - footprint_half_width_m / meters_per_longitude_degree, 7), round(latitude - footprint_half_height_m / 111_320, 7)],
            [round(longitude - footprint_half_width_m / meters_per_longitude_degree, 7), round(latitude + footprint_half_height_m / 111_320, 7)],
        ]
    ]
    return {"type": "MultiPolygon", "coordinates": polygons}, {"type": "Polygon", "coordinates": footprint}, round(pixel_area_km2, 6)


def _infer_image(image_base64: str):
    try:
        image_bytes = satellite_service.get_image(image_base64)
        if image_bytes is None:
            return None
        image = decode_image(image_bytes)
        prepared, quality = preprocess(image)
        result = predict(prepared, quality)
        return image, quality, result
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Unable to process satellite image: {exc}") from exc


def _run_detection(request: DetectionRequest) -> dict:
    inference = _infer_image(request.image_base64)
    image, _, result = inference
    ratio = result["mask_area_ratio"]
    area = ratio * request.scene_area_km2 if request.scene_area_km2 is not None else None
    geometry = None
    image_footprint = None
    if (
        request.ground_sample_distance_m is not None
        and request.latitude is not None
        and request.longitude is not None
    ):
        geometry, image_footprint, area = _georeference_mask(
            result["segmentation_mask_base64"],
            image.width,
            image.height,
            request.latitude,
            request.longitude,
            request.ground_sample_distance_m,
        )
    return {
        "success": True,
        "detection_id": f"OT-{uuid4().hex[:8].upper()}",
        "detection": {
            "oil_spill_detected": result["spill_detected"],
            "confidence": result["confidence"],
            "area_km2": round(area, 4) if area is not None else None,
            "mask_area_ratio": ratio,
            "confidence_is_calibrated": result["confidence_is_calibrated"],
            "segmentation_mask_base64": result["segmentation_mask_base64"],
            "mask_width": result["mask_width"],
            "mask_height": result["mask_height"],
        },
        "location": (
            {"latitude": request.latitude, "longitude": request.longitude}
            if request.latitude is not None and request.longitude is not None
            else None
        ),
        "geometry": geometry,
        "image_footprint": image_footprint,
        "ground_sample_distance_m": request.ground_sample_distance_m,
        "model_name": result["model_name"],
        "inference_mode": result["inference_mode"],
        "image_width": image.width,
        "image_height": image.height,
        "data_source": "uploaded_image",
        "is_demo": result["model_name"] == "OilTrace-CV-Bootstrap",
    }


@router.post("/api/v1/detect")
def detect(request: DetectionRequest) -> dict:
    return _run_detection(request)


class AnalyzeRequest(BaseModel):
    image_base64: str = Field(min_length=1)
    latitude: float | None = None
    longitude: float | None = None
    timestamp: str | None = None


@router.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    inferred = _infer_image(request.image_base64)
    if inferred is None:
        raise HTTPException(status_code=422, detail="An image is required")
    image, quality, result = inferred
    return {
        **result,
        "quality": quality,
        "image_width": image.width,
        "image_height": image.height,
    }
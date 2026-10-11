"""AIS provider boundary with a small, clearly synthetic development dataset."""

from datetime import datetime, timedelta, timezone
import math
from typing import Protocol


class AISService(Protocol):
    def get_vessels(self, latitude: float, longitude: float, timestamp: str | None) -> list[dict]: ...


class MockAISService:
    def get_vessels(self, latitude: float, longitude: float, timestamp: str | None) -> list[dict]:
        release_time = (
            datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if timestamp
            else datetime.now(timezone.utc)
        )
        candidates = [
            ("900000001", "Sample Tanker Alpha", "Crude Oil Tanker", 0.008, 0.01, 8.5, 135.0),
            ("900000002", "Sample Cargo Bravo", "General Cargo", 0.08, 0.1, 15.0, 270.0),
            ("900000003", "Sample Fishing Vessel Charlie", "Fishing Vessel", -0.12, -0.16, 5.5, 45.0),
        ]
        records = []
        for mmsi, name, vessel_type, offset_lat, offset_lon, speed_knots, course in candidates:
            for minutes in (-60, 0, 60):
                elapsed_seconds = minutes * 60
                course_radians = math.radians(course)
                speed_meters_per_second = speed_knots * 0.514444
                north_meters = speed_meters_per_second * elapsed_seconds * math.cos(course_radians)
                east_meters = speed_meters_per_second * elapsed_seconds * math.sin(course_radians)
                latitude_at_time = latitude + offset_lat + north_meters / 111_320
                meters_per_longitude_degree = max(1.0, 111_320 * abs(math.cos(math.radians(latitude))))
                longitude_at_time = longitude + offset_lon + east_meters / meters_per_longitude_degree
                records.append({
                    "mmsi": mmsi,
                    "name": name,
                    "vessel_type": vessel_type,
                    "timestamp": (release_time + timedelta(seconds=elapsed_seconds)).isoformat(),
                    "latitude": round(latitude_at_time, 6),
                    "longitude": round(longitude_at_time, 6),
                    "speed_knots": speed_knots,
                    "course_over_ground": course,
                    "source": "synthetic_sample",
                })
        return records
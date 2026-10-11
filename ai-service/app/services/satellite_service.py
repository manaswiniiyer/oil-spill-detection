"""Satellite imagery provider boundary; the initial provider uses request data."""

import base64
from typing import Protocol


class SatelliteService(Protocol):
    def get_image(self, image_base64: str | None) -> bytes | None: ...


class MockSatelliteService:
    """Accept a local image supplied by the caller; never contacts a provider."""

    def get_image(self, image_base64: str | None) -> bytes | None:
        if not image_base64:
            return None
        encoded = image_base64.split(",", 1)[-1]
        return base64.b64decode(encoded, validate=True)
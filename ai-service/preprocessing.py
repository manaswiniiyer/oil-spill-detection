"""Image preparation shared by the OilTrace inference service."""

from io import BytesIO

import numpy as np
from PIL import Image, ImageOps

IMAGE_SIZE = (256, 256)


def decode_image(image_bytes: bytes) -> Image.Image:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return ImageOps.exif_transpose(image)


def preprocess(image: Image.Image) -> tuple[Image.Image, dict[str, float]]:
    """Prepare a grayscale 256px tile; SAR calibration must happen upstream."""
    normalized = image.convert("L").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    gray = np.asarray(normalized, dtype=np.float32) / 255.0
    return normalized, {
        "mean_intensity": float(gray.mean()),
        "contrast": float(gray.std()),
        "dynamic_range": float(gray.max() - gray.min()),
    }

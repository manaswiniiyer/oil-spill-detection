"""Oil-spill segmentation inference with a trained U-Net or explicit CV fallback."""

import base64
from functools import lru_cache
from io import BytesIO
import os
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_PATH = Path(os.environ.get(
    "OILTRACE_MODEL_PATH",
    str(Path(__file__).parent / "model" / "oiltrace_unet.pth"),
))
IMAGE_SIZE = (256, 256)
MASK_THRESHOLD = 0.5
MIN_SPILL_AREA_RATIO = 0.001


@lru_cache(maxsize=1)
def _load_model():
    import torch

    from unet import UNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model = UNet(
        in_channels=checkpoint.get("in_channels", 1),
        base_channels=checkpoint.get("base_channels", 32),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return torch, model, device


def _mask_as_base64(mask: np.ndarray) -> str:
    buffer = BytesIO()
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def predict(image: Image.Image, quality: dict[str, float]) -> dict:
    """Return a binary mask and summary for a calibrated, single-channel SAR tile."""
    if MODEL_PATH.exists():
        torch, model, device = _load_model()
        tile = image.convert("L").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        pixels = np.asarray(tile, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(pixels).unsqueeze(0).unsqueeze(0).to(device)
        with torch.inference_mode():
            probabilities = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
        mask = probabilities >= MASK_THRESHOLD
        area_ratio = float(mask.mean())
        confidence = float(probabilities[mask].mean()) if mask.any() else float((1 - probabilities).mean())
        return {
            "spill_detected": area_ratio >= MIN_SPILL_AREA_RATIO,
            "confidence": round(confidence, 4),
            "confidence_is_calibrated": False,
            "mask_area_ratio": round(area_ratio, 6),
            "segmentation_mask_base64": _mask_as_base64(mask),
            "mask_width": IMAGE_SIZE[0],
            "mask_height": IMAGE_SIZE[1],
            "model_name": "OilTrace-SAR-U-Net",
            "inference_mode": "trained PyTorch U-Net",
        }

    # Demo-only fallback: dark regions are not sufficient evidence of oil.
    grayscale = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    threshold = min(0.42, max(0.18, quality["mean_intensity"] - quality["contrast"] * 0.35))
    mask = grayscale < threshold
    area_ratio = float(mask.mean())
    confidence = min(0.985, max(0.52, 0.62 + quality["contrast"] * 0.9 + area_ratio * 0.35))
    return {
        "spill_detected": bool(area_ratio > 0.025),
        "confidence": round(confidence, 4),
        "confidence_is_calibrated": False,
        "mask_area_ratio": round(area_ratio, 6),
        "segmentation_mask_base64": _mask_as_base64(mask),
        "mask_width": image.width,
        "mask_height": image.height,
        "model_name": "OilTrace-CV-Bootstrap",
        "inference_mode": "demo-only dark-pixel heuristic; not a trained model",
    }

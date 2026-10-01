"""Input handling: image (file / URL / base64) + product JSON validation.

Spec (Section 2):
  - Image: file upload, image URL, ya base64 encoded.
  - Product JSON: fields optional ho sakti hain - system phir bhi chalna chahiye.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests


class InputError(ValueError):
    """Galat ya missing input ke liye."""


def decode_image_bytes(data: bytes) -> np.ndarray:
    if not data:
        raise InputError("Image data is empty.")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise InputError("Could not decode image (corrupted or unsupported format).")
    if img.shape[0] < 10 or img.shape[1] < 10:
        raise InputError("Image too small to analyze (min 10x10).")
    return img


# backward-compatible private alias
_decode_image_bytes = decode_image_bytes


decode_image_bytes = _decode_image_bytes  # public alias (API/uploads ke liye)


def load_image(
    source: str | Path | None = None,
    *,
    base64_str: str | None = None,
    url: str | None = None,
    timeout: int = 15,
) -> np.ndarray:
    """Image lo (file path, URL, ya base64) -> BGR numpy array."""
    if base64_str:
        payload = base64_str.strip()
        if "," in payload and payload[:5].lower() in ("data:",):
            payload = payload.split(",", 1)[1]
        try:
            data = base64.b64decode(payload, validate=False)
        except Exception as exc:  # noqa: BLE001
            raise InputError(f"Invalid base64 image: {exc}") from exc
        return _decode_image_bytes(data)

    if url:
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            raise InputError(f"Could not download image from URL: {exc}") from exc
        return _decode_image_bytes(resp.content)

    if source:
        path = Path(str(source))
        if path.is_dir():
            raise InputError(f"Expected a file but got a directory: {path}")
        if not path.exists():
            raise InputError(f"Image file not found: {path}")
        return _decode_image_bytes(path.read_bytes())

    raise InputError("No image provided (need file, URL, or base64).")


def load_product_json(source: str | Path | dict | None) -> dict[str, Any]:
    """Product JSON lo (dict, JSON string, ya file path) -> normalized dict.

    Spec (Section 2.3): optional fields missing ho sakte hain - yeh sirf
    dict hai aur default values nahi lagata; checks missing fields handle karte hain.
    """
    if source is None:
        raise InputError("No product JSON provided.")

    if isinstance(source, dict):
        data: Any = source
    elif isinstance(source, Path) or (isinstance(source, str) and source.strip().startswith("{")):
        text = str(source)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise InputError(f"Invalid product JSON string: {exc}") from exc
    else:
        path = Path(str(source))
        if not path.exists():
            raise InputError(f"Product JSON file not found: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise InputError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise InputError("Product JSON must be an object.")

    # id -> product_id alias normalize
    if "product_id" not in data and "id" in data:
        data["product_id"] = data["id"]
    return data


def product_description(product: dict[str, Any]) -> str:
    """JSON se ek natural description text banao (CLIP / matching ke liye)."""
    parts: list[str] = []
    for key in ("name", "brand", "laboratory", "category", "form", "dosage", "packaging", "quantity"):
        value = product.get(key)
        if value:
            parts.append(str(value))
    if not parts:
        extra = {k: v for k, v in product.items() if isinstance(v, (str, int, float))}
        parts = [f"{k}: {v}" for k, v in list(extra.items())[:6]]
    return ", ".join(parts) if parts else "unknown product"

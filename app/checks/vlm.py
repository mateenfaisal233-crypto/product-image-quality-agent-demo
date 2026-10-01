"""Optional VLM validation layer - Gemini / Groq API.

Spec (Section 8): "An LLM/VLM may be used as an additional reasoning or
validation layer" - yeh layer status change NAHI karti, sirf extra
validation notes issues mein add karti hai. Default OFF.

Providers:
  - gemini (default): GEMINI_API_KEY  (Google Generative Language API)
  - groq:             GROQ_API_KEY    (OpenAI-compatible)
"""

from __future__ import annotations

import base64
import json
import os
import re

import cv2
import numpy as np
import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

PROMPT = (
    "You are a product image validator. A catalog entry is provided as JSON. "
    "Look at the image and answer with STRICT JSON only: "
    '{"is_correct_product": true|false, "image_issues": ["short issue", ...], '
    '"detected_product_name": "...", "detected_brand": "..."} '
    "Do not write any text outside the JSON."
)


def _encode_image_b64(img_bgr: np.ndarray, max_side: int = 512) -> str:
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise ValueError("Could not encode image for VLM.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _api_key(cfg: dict) -> str | None:
    vlm = cfg.get("vlm", {})
    return os.environ.get(vlm.get("api_key_env", "GEMINI_API_KEY")) or None


def is_configured(cfg: dict) -> bool:
    vlm = cfg.get("vlm", {})
    if not vlm.get("enabled"):
        return False
    if vlm.get("provider", "gemini") not in {"gemini", "groq"}:
        return False
    return _api_key(cfg) is not None


def _parse_json(content: str) -> dict | None:
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _notes_from(data: dict) -> list[str]:
    notes: list[str] = []
    if data.get("is_correct_product") is False:
        notes.append("VLM validation: model suggests the image may not be the correct product.")
    for issue in (data.get("image_issues") or [])[:5]:
        notes.append(f"VLM validation: {issue}")
    return notes


def _validate_gemini(img_bgr: np.ndarray, product: dict, cfg: dict, timeout: int) -> list[str]:
    vlm = cfg["vlm"]
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            f"Catalog product JSON:\n{json.dumps(product, ensure_ascii=False)}\n\n{PROMPT}"
                        )
                    },
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": _encode_image_b64(img_bgr),
                        }
                    },
                ],
            }
        ],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 500},
    }
    resp = requests.post(
        GEMINI_URL.format(model=vlm.get("model", "gemini-2.5-flash")),
        params={"key": _api_key(cfg)},
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    body = resp.json()
    parts = body.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts)
    data = _parse_json(text)
    return _notes_from(data) if data else []


def _validate_groq(img_bgr: np.ndarray, product: dict, cfg: dict, timeout: int) -> list[str]:
    vlm = cfg["vlm"]
    payload = {
        "model": vlm.get("model", "meta-llama/llama-4-scout-17b-16e-instruct"),
        "temperature": 0,
        "max_tokens": 400,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Catalog product JSON:\n{json.dumps(product, ensure_ascii=False)}\n\n{PROMPT}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64," + _encode_image_b64(img_bgr)},
                    },
                ],
            }
        ],
    }
    resp = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {_api_key(cfg)}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    data = _parse_json(content)
    return _notes_from(data) if data else []


def validate(img_bgr: np.ndarray, product: dict, cfg: dict, timeout: int = 45) -> list[str]:
    """VLM se validation notes. Fail ho to best-effort message (agent chalta rahe)."""
    if not is_configured(cfg):
        return []
    provider = cfg["vlm"].get("provider", "gemini")
    last_exc: Exception | None = None
    for attempt in range(2):  # transient 5xx/429 pe ek retry
        try:
            if provider == "groq":
                return _validate_groq(img_bgr, product, cfg, timeout)
            return _validate_gemini(img_bgr, product, cfg, timeout)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if attempt == 0 and status in {429, 500, 502, 503, 504}:
                import time

                time.sleep(2)
                continue
            break
    return [f"VLM validation skipped ({type(last_exc).__name__}: {last_exc})."]

"""Section 5 - Quality scoring (0..100 per score + weighted overall)."""

from __future__ import annotations

SCORE_KEYS = [
    "product_match",
    "image_quality",
    "sharpness",
    "resolution",
    "visibility",
    "composition",
    "background",
    "lighting",
    "text_readability",
]


def normalize_weights(weights: dict) -> dict:
    """Weights ko 1.0 pe normalize (missing -> 0)."""
    clean = {k: max(float(v or 0.0), 0.0) for k, v in (weights or {}).items()}
    total = sum(clean.values())
    if total <= 0:
        n = len(SCORE_KEYS)
        return {k: 1.0 / n for k in SCORE_KEYS}
    return {k: v / total for k, v in clean.items()}


def image_quality_score(parts: dict) -> float:
    """image_quality = sharpness/resolution/lighting/contrast/noise ka blend
    (alag se overall mein weights ke sath aata hai)."""
    weights = {"sharpness": 0.40, "resolution": 0.20, "lighting": 0.20, "contrast": 0.10, "noise": 0.10}
    total, acc = 0.0, 0.0
    for key, weight in weights.items():
        value = float(parts.get(key, 75.0))
        acc += weight * value
        total += weight
    return round(acc / total, 1) if total else 0.0


def compute_overall(scores: dict, weights: dict) -> float:
    """S_overall = sum(w_i * s_i), weights normalized (spec section 5)."""
    norm = normalize_weights(weights)
    overall = 0.0
    for key in SCORE_KEYS:
        overall += norm.get(key, 0.0) * float(scores.get(key, 0.0))
    return round(overall, 1)

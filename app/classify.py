"""Section 6 - Final classification.

Priority: WRONG_* > quality/anomaly statuses > VALID

Mapping (README mein documented):
  - correctness fail           -> WRONG_PRODUCT / WRONG_BRAND / WRONG_VARIANT / WRONG_CATEGORY
  - koi anomaly nahi           -> VALID
  - quality-related anomalies  -> VALID_BUT_LOW_QUALITY (spec section 9:
    Correct Product + Blurry Image => VALID_BUT_LOW_QUALITY; blur EXCESSIVE bhi ho)
  - overall bahut neeche (<50) ya corrupted -> LOW_IMAGE_QUALITY
  - sirf composition/bg/vis    -> ANOMALY
"""

from __future__ import annotations

QUALITY_CODES = {
    "BLUR",
    "EXCESSIVE_BLUR",
    "LOW_RESOLUTION",
    "OVEREXPOSURE",
    "UNDEREXPOSURE",
    "POOR_CONTRAST",
    "NOISE",
    "PIXELATION",
    "COMPRESSION_ARTIFACTS",
    "DISTORTION",
    "CORRUPTED_IMAGE",
    "TEXT_UNREADABLE",
}

# Corrupted image hi "severe" - blur spec section 9 ke mutabiq VBLQ jaata hai
SEVERE_CODES = {"CORRUPTED_IMAGE"}

WRONG_STATUSES = {"WRONG_PRODUCT", "WRONG_BRAND", "WRONG_VARIANT", "WRONG_CATEGORY"}


def classify(
    wrong_status: str | None,
    anomaly_types: list[str],
    scores: dict,
    cfg: dict,
) -> str:
    if wrong_status in WRONG_STATUSES:
        return wrong_status

    if not anomaly_types:
        overall = float(scores.get("overall", 100.0))
        if overall < cfg["classification"]["low_quality_max_overall"]:
            return "LOW_IMAGE_QUALITY"
        if overall < cfg["classification"]["valid_min_overall"]:
            return "VALID_BUT_LOW_QUALITY"
        return "VALID"

    if SEVERE_CODES & set(anomaly_types):
        return "LOW_IMAGE_QUALITY"

    if float(scores.get("overall", 0.0)) < cfg["classification"]["low_quality_max_overall"]:
        return "LOW_IMAGE_QUALITY"

    if QUALITY_CODES & set(anomaly_types):
        return "VALID_BUT_LOW_QUALITY"

    return "ANOMALY"

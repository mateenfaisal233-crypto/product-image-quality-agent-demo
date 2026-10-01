"""Section 7 - Required JSON output (exact schema)."""

from __future__ import annotations

from typing import Any


def build_report(
    product: dict[str, Any],
    status: str,
    is_correct: bool,
    anomaly_types: list[str],
    scores: dict,
    detected_product: dict,
    issues: list[str],
    explanation: str,
) -> dict:
    product_id = product.get("product_id", product.get("id", ""))
    return {
        "product_id": product_id,
        "is_correct_product": bool(is_correct),
        "is_anomaly": status != "VALID",
        "status": status,
        "anomaly_types": anomaly_types,
        "scores": {k: scores[k] for k in (
            "product_match",
            "image_quality",
            "sharpness",
            "resolution",
            "visibility",
            "composition",
            "background",
            "lighting",
            "text_readability",
            "overall",
        )},
        "detected_product": {
            "name": detected_product.get("name", "Unknown"),
            "brand": detected_product.get("brand", "Unknown"),
            "category": detected_product.get("category", "Unknown"),
        },
        "issues": issues,
        "explanation": explanation,
    }


def make_explanation(status: str, is_correct: bool, issues: list[str]) -> str:
    """Ek saaf saaf wajuhaat wala jumla (spec: bina wajuha 'bad' nahi kehna)."""
    if status == "VALID":
        return (
            "The image corresponds to the provided product and has good visual quality; "
            "it is suitable for a professional product catalog."
        )
    reason = issues[0] if issues else "issues were detected in the image."
    if status in {"WRONG_PRODUCT", "WRONG_BRAND", "WRONG_VARIANT", "WRONG_CATEGORY"}:
        return f"The image does not match the provided product metadata. {reason}"
    if status == "VALID_BUT_LOW_QUALITY":
        return (
            f"The image appears to represent the correct product, but its visual quality "
            f"is not suitable for a professional product catalog. {reason}"
        )
    if status == "LOW_IMAGE_QUALITY":
        return f"The image quality is too low for a professional product catalog. {reason}"
    if status == "ANOMALY":
        return f"The product is correct but the image contains visual anomalies. {reason}"
    return reason

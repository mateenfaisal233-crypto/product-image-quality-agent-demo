"""Report ka JSON schema + scoring rules (spec sections 5 & 7)."""

from __future__ import annotations

REQUIRED_KEYS = {
    "product_id",
    "is_correct_product",
    "is_anomaly",
    "status",
    "anomaly_types",
    "scores",
    "detected_product",
    "issues",
    "explanation",
}

SCORE_KEYS = {
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
}


def assert_valid_report(report: dict) -> None:
    missing = REQUIRED_KEYS - set(report)
    assert not missing, f"missing keys: {missing}"

    missing_scores = SCORE_KEYS - set(report["scores"])
    assert not missing_scores, f"missing scores: {missing_scores}"

    for key, value in report["scores"].items():
        assert 0 <= value <= 100, f"{key}={value} out of [0,100]"

    assert report["status"] in {
        "VALID",
        "VALID_BUT_LOW_QUALITY",
        "ANOMALY",
        "WRONG_PRODUCT",
        "WRONG_BRAND",
        "WRONG_VARIANT",
        "WRONG_CATEGORY",
        "LOW_IMAGE_QUALITY",
    }, f"unexpected status: {report['status']}"

    assert isinstance(report["is_correct_product"], bool)
    assert isinstance(report["is_anomaly"], bool)
    assert isinstance(report["anomaly_types"], list)
    assert isinstance(report["issues"], list)
    assert report["explanation"], "explanation must not be empty"
    assert set(report["detected_product"]) == {"name", "brand", "category"}

    if report["status"] == "VALID":
        assert report["anomaly_types"] == [], "VALID must have no anomalies"
        assert report["is_anomaly"] is False
        assert report["is_correct_product"] is True
    else:
        assert report["is_anomaly"] is True, "non-VALID status must set is_anomaly"
        assert report["anomaly_types"], "non-VALID status must list anomaly_types"
        assert report["issues"], "non-VALID status must explain the reason"

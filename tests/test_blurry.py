"""Spec sections 4.1 + 9: sahi product par blurry image -> quality status
(WRONG_PRODUCT nahi) - dono independent evaluate hote hain."""

from __future__ import annotations

from helpers import assert_valid_report


def test_blurry_image_is_quality_issue_not_wrong_product(analyze):
    report = analyze("doliprane_blurry.jpg")
    assert_valid_report(report)
    assert report["is_correct_product"] is True, "blur must NOT mark product wrong"
    # Spec section 9 ka canonical example: Correct Product + Blurry Image
    # => VALID_BUT_LOW_QUALITY
    assert report["status"] == "VALID_BUT_LOW_QUALITY", report["status"]
    assert "BLUR" in report["anomaly_types"] or "EXCESSIVE_BLUR" in report["anomaly_types"], report["anomaly_types"]
    assert report["scores"]["sharpness"] < 70, report["scores"]
    assert report["issues"]

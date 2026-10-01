"""Spec section 3.1: Doliprane expected, image mein Maalox -> WRONG_PRODUCT."""

from __future__ import annotations

from helpers import assert_valid_report


def test_wrong_product_flagged(analyze):
    report = analyze("maalox.jpg")
    assert_valid_report(report)
    assert report["status"] == "WRONG_PRODUCT", report
    assert report["is_correct_product"] is False
    assert report["is_anomaly"] is True
    assert "WRONG_PRODUCT" in report["anomaly_types"]
    assert report["scores"]["product_match"] < 50, report["scores"]
    assert report["issues"], "must explain why"
    assert report["explanation"]


def test_wrong_product_detection_guess(analyze):
    report = analyze("maalox.jpg")
    name = report["detected_product"]["name"].lower()
    assert "maalox" in name, f"detected name should show the actual product: {name}"

"""Spec section 3.4: Medicament expected, image mein Shampoo -> WRONG_CATEGORY."""

from __future__ import annotations

from helpers import assert_valid_report


def test_wrong_category_flagged(analyze):
    report = analyze("shampoo.jpg")
    assert_valid_report(report)
    assert report["status"] == "WRONG_CATEGORY", report
    assert report["is_correct_product"] is False
    assert "WRONG_CATEGORY" in report["anomaly_types"]
    assert report["issues"]

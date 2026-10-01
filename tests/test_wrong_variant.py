"""Spec section 3.3: Doliprane 1000mg expected, image mein 500mg -> WRONG_VARIANT."""

from __future__ import annotations

from helpers import assert_valid_report


def test_wrong_dosage_is_wrong_variant(analyze):
    report = analyze("doliprane_500.jpg")
    assert_valid_report(report)
    assert report["status"] == "WRONG_VARIANT", report
    assert report["is_correct_product"] is False
    assert "WRONG_VARIANT" in report["anomaly_types"]
    assert report["scores"]["product_match"] < 70, report["scores"]
    assert any("500" in issue or "dosage" in issue.lower() or "strength" in issue.lower()
               for issue in report["issues"]), report["issues"]

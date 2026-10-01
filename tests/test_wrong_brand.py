"""Spec section 3.2: Sanofi expected, image mein doosra brand -> WRONG_BRAND."""

from __future__ import annotations

from helpers import assert_valid_report


def test_wrong_brand_flagged(analyze):
    report = analyze("doliprane_badbrand.jpg")
    assert_valid_report(report)
    assert report["status"] == "WRONG_BRAND", report
    assert report["is_correct_product"] is False
    assert "WRONG_BRAND" in report["anomaly_types"]
    assert report["detected_product"]["brand"].lower() in {"unknown", "novartis"}

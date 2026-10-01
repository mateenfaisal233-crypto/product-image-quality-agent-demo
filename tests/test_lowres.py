"""Spec section 4.1: low resolution image -> LOW_RESOLUTION anomaly."""

from __future__ import annotations

from helpers import assert_valid_report


def test_low_resolution_detected(analyze):
    report = analyze("doliprane_lowres.jpg")
    assert_valid_report(report)
    assert report["is_correct_product"] is True, "low-res must not mark product wrong"
    assert "LOW_RESOLUTION" in report["anomaly_types"], report["anomaly_types"]
    assert report["status"] in {"VALID_BUT_LOW_QUALITY", "LOW_IMAGE_QUALITY", "ANOMALY"}, report["status"]
    assert report["scores"]["resolution"] < 60, report["scores"]

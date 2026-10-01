"""Spec test: correct product + achi quality -> VALID."""

from __future__ import annotations

from helpers import assert_valid_report


def test_correct_product_is_valid(analyze):
    report = analyze("doliprane_ok.jpg")
    assert_valid_report(report)
    assert report["status"] == "VALID", report
    assert report["is_correct_product"] is True
    assert report["anomaly_types"] == []
    assert report["product_id"] == "12345"
    assert report["scores"]["product_match"] >= 70, report["scores"]
    assert report["scores"]["overall"] >= 70, report["scores"]


def test_detected_product_matches(analyze):
    report = analyze("doliprane_ok.jpg")
    detected = report["detected_product"]
    assert "doliprane" in detected["name"].lower()
    assert detected["category"].lower().startswith("medicament")


def test_explanation_is_meaningful(analyze):
    report = analyze("doliprane_ok.jpg")
    text = report["explanation"].lower()
    assert "provided product" in text and "good visual quality" in text

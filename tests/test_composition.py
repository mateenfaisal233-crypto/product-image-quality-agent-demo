"""Spec section 4.3: product corner mein + bohot khali jagah -> BAD_COMPOSITION."""

from __future__ import annotations

from helpers import assert_valid_report


def test_poor_composition_detected(analyze):
    report = analyze("doliprane_composition.jpg")
    assert_valid_report(report)
    assert report["is_correct_product"] is True, "composition issue must not mark product wrong"
    composition_codes = {"BAD_COMPOSITION", "EXCESSIVE_EMPTY_SPACE", "MULTIPLE_PRODUCTS"}
    hit = composition_codes & set(report["anomaly_types"])
    assert hit, f"expected composition anomaly, got: {report['anomaly_types']}"
    assert report["status"] in {"ANOMALY", "VALID_BUT_LOW_QUALITY", "LOW_IMAGE_QUALITY"}, report["status"]
    assert report["scores"]["composition"] < 80, report["scores"]


def test_cluttered_background_detected(analyze):
    report = analyze("doliprane_clutter.jpg")
    assert_valid_report(report)
    assert report["is_correct_product"] is True
    bg_codes = {"DISTRACTING_BACKGROUND", "BACKGROUND_CLUTTER", "MULTIPLE_PRODUCTS"}
    hit = bg_codes & set(report["anomaly_types"])
    assert hit, f"expected background anomaly, got: {report['anomaly_types']}"
    assert report["scores"]["background"] < 90, report["scores"]

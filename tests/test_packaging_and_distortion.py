from __future__ import annotations

import cv2
import numpy as np

from app.checks.correctness import evaluate_correctness
from app.checks.quality import distortion_ratio, evaluate_quality
from app.config import load_config


def test_packaging_quantity_mismatch_is_wrong_variant():
    product = {
        "name": "Doliprane 1000 mg Comprimé",
        "category": "Medicament",
        "brand": "Sanofi",
        "dosage": "1000 mg",
        "packaging": "Boîte de 8 comprimés",
    }
    ocr = [
        {"text": "DOLIPRANE", "score": 99},
        {"text": "1000 mg", "score": 99},
        {"text": "Comprimés", "score": 99},
        {"text": "Sanofi", "score": 99},
        {"text": "Boîte de 16 comprimés", "score": 99},
    ]
    result = evaluate_correctness(
        np.zeros((100, 100, 3), dtype=np.uint8),
        product,
        ocr,
        None,
        load_config()["thresholds"],
    )
    assert result["wrong_status"] == "WRONG_VARIANT"
    assert any("quantity" in issue.lower() for issue in result["issues"])


def test_strong_perspective_warp_is_distortion():
    mask = np.zeros((400, 400), dtype=np.uint8)
    polygon = np.array([[40, 40], [360, 120], [280, 360], [120, 280]], dtype=np.int32)
    cv2.fillConvexPoly(mask, polygon, 255)
    ratio = distortion_ratio(mask)
    assert ratio > 1.35

    cfg = load_config()
    image = np.full((400, 400, 3), 255, dtype=np.uint8)
    _, anomalies, issues = evaluate_quality(image, cfg["thresholds"], product_mask=mask)
    assert "DISTORTION" in anomalies
    assert any("distort" in issue.lower() for issue in issues)

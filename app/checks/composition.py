"""Section 4.3 - Composition anomalies."""

from __future__ import annotations

import numpy as np

from .visibility import Segmentation


def evaluate_composition(seg: Segmentation, img: np.ndarray, th: dict) -> tuple[dict, list[str], list[str]]:
    anomalies: list[str] = []
    issues: list[str] = []
    h, w = img.shape[:2]
    total = float(h * w)
    area = seg.area_ratio

    if seg.bbox is None or area < th["product_min_area_ratio"]:
        score = 20.0
        if "PRODUCT_NOT_VISIBLE" not in anomalies:
            pass
        return {"composition": score, "center_offset": None}, anomalies, issues

    x, y, bw, bh = seg.bbox

    # --- centering ---
    cx = (x + bw / 2.0) / w
    cy = (y + bh / 2.0) / h
    offset = float(np.hypot(cx - 0.5, cy - 0.5))
    if offset > th["center_offset_max"]:
        anomalies.append("BAD_COMPOSITION")
        issues.append("Product is poorly positioned in the frame.")

    # --- margins / edge proximity ---
    margin_ratio = min(x, y, w - (x + bw), h - (y + bh)) / float(min(w, h))
    if margin_ratio < th["edge_margin_min"] and area > th["product_small_area_ratio"]:
        if "BAD_COMPOSITION" not in anomalies:
            anomalies.append("BAD_COMPOSITION")
        issues.append("Product is placed too close to the edge of the image.")

    # --- empty space ---
    if area < (1.0 - th["empty_space_max"]):
        anomalies.append("EXCESSIVE_EMPTY_SPACE")
        issues.append("Excessive empty space around the product.")

    # --- framing / portion ---
    if area > 0.95:
        if "BAD_COMPOSITION" not in anomalies:
            anomalies.append("BAD_COMPOSITION")
        issues.append("Product fills almost the entire frame (no margins).")

    # --- multiple unrelated products ---
    big_contours = [
        c for c in seg.contours
        if (cv2_area(c) / total) > 0.04
    ]
    secondary = len(big_contours) - 1
    if secondary >= 1:
        anomalies.append("MULTIPLE_PRODUCTS")
        issues.append("Multiple objects detected - image may contain unrelated products.")

    score = 100.0
    if "BAD_COMPOSITION" in anomalies:
        score -= 35.0
    if "EXCESSIVE_EMPTY_SPACE" in anomalies:
        score -= 30.0
    if "MULTIPLE_PRODUCTS" in anomalies:
        score -= 25.0
    score -= float(np.clip(offset, 0.0, 0.5) * 60.0)
    score = float(np.clip(score, 0.0, 100.0))

    return {
        "composition": round(score, 1),
        "center_offset": round(offset, 4),
        "margin_ratio": round(margin_ratio, 4),
        "area_ratio": round(area, 4),
        "object_count": len(big_contours),
    }, anomalies, issues


def cv2_area(contour) -> float:
    import cv2

    return float(cv2.contourArea(contour))

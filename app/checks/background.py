"""Section 4.4 - Background anomalies."""

from __future__ import annotations

import cv2
import numpy as np

from .visibility import Segmentation


def evaluate_background(img: np.ndarray, seg: Segmentation, th: dict) -> tuple[dict, list[str], list[str]]:
    anomalies: list[str] = []
    issues: list[str] = []
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)

    bg_mask = np.ones_like(edges, dtype=bool)
    if seg.bbox is not None and seg.area_ratio > th["product_min_area_ratio"]:
        x, y, bw, bh = seg.bbox
        pad = max(2, min(w, h) // 60)
        bg_mask[max(0, y - pad): y + bh + pad, max(0, x - pad): x + bw + pad] = False

    bg_edges = edges[bg_mask]
    bg_pixels = max(int(bg_edges.size), 1)
    edge_density = float((bg_edges > 0).sum()) / bg_pixels

    bg_colors = img[bg_mask] if bg_mask.any() else np.zeros((1, 3), np.uint8)
    color_std = float(bg_colors.astype(np.float32).std())

    # Flat colorful shapes bhi distracting hoti hain (Canny unhe miss karta hai)
    # - isliye color variance bhi ek signal hai.
    if edge_density > th["bg_edge_density_max"]:
        anomalies.append("DISTRACTING_BACKGROUND")
        issues.append("Distracting / cluttered background around the product.")
    elif color_std > th.get("bg_color_std_max", 30.0):
        anomalies.append("DISTRACTING_BACKGROUND")
        issues.append("Background contains unwanted colored objects / clutter.")

    score = 100.0
    score -= float(np.clip((edge_density - th["bg_edge_density_max"]) / max(th["bg_edge_density_max"], 1e-6), 0.0, 1.0) * 70.0)
    score -= float(np.clip((color_std - th.get("bg_color_std_max", 30.0)) / 60.0, 0.0, 1.0) * 60.0)
    if color_std > 75.0 and edge_density > th["bg_edge_density_max"] * 0.7:
        if "DISTRACTING_BACKGROUND" not in anomalies:
            anomalies.append("BACKGROUND_CLUTTER")
            issues.append("Background contains many objects/colors that compete with the product.")
        score -= 15.0

    score = float(np.clip(score, 0.0, 100.0))
    return {
        "background": round(score, 1),
        "bg_edge_density": round(edge_density, 4),
        "bg_color_std": round(color_std, 2),
    }, anomalies, issues

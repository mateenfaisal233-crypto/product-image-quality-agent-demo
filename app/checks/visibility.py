"""Section 4.2 - Product visibility anomalies.

Product segmentation border-color model se (simple + fast, catalog photos
par acha chalta hai): image ke kinare ka median color model banate hain,
us se farq wala hissa foreground mana jata hai.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Segmentation:
    mask: np.ndarray
    bbox: tuple[int, int, int, int] | None  # x, y, w, h
    area_ratio: float
    contours: list


def segment_product(img: np.ndarray) -> Segmentation:
    h, w = img.shape[:2]
    small = img
    if max(h, w) > 800:
        scale = 800.0 / max(h, w)
        small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    sh, sw = small.shape[:2]
    border = max(4, min(sh, sw) // 40)
    strips = [
        small[:border, :, :].reshape(-1, 3),
        small[-border:, :, :].reshape(-1, 3),
        small[:, :border, :].reshape(-1, 3),
        small[:, -border:, :].reshape(-1, 3),
    ]
    bg_color = np.median(np.vstack(strips), axis=0)
    # Raw distance threshold: white/pure background ke khilaf product body
    # (raw dist ~25) ko pakarte hain. Otsu kabhi high threshold chun kar sirf
    # dark text bacha deta hai - isliye fixed loose threshold + morphology.
    dist = np.linalg.norm(small.astype(np.float32) - bg_color.astype(np.float32), axis=2)
    thr = 16.0
    mask_small = ((dist > thr).astype(np.uint8)) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_OPEN, kernel)
    mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    full_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    full_contours = sorted(full_contours, key=cv2.contourArea, reverse=True)

    total = float(h * w)
    if full_contours:
        largest = full_contours[0]
        x, y, bw, bh = cv2.boundingRect(largest)
        area_ratio = float(cv2.contourArea(largest)) / total
    else:
        x = y = bw = bh = 0
        area_ratio = 0.0

    return Segmentation(mask=mask, bbox=(x, y, bw, bh), area_ratio=area_ratio, contours=full_contours)


def evaluate_visibility(seg: Segmentation, img: np.ndarray, th: dict) -> tuple[dict, list[str], list[str]]:
    anomalies: list[str] = []
    issues: list[str] = []
    h, w = img.shape[:2]
    total = float(h * w)
    area = seg.area_ratio

    if area < th["product_min_area_ratio"]:
        anomalies.append("PRODUCT_NOT_VISIBLE")
        issues.append("Product is not clearly visible in the image.")
    elif area < th["product_small_area_ratio"]:
        anomalies.append("PRODUCT_TOO_SMALL")
        issues.append("Product occupies a very small part of the image.")

    cut_off = False
    if seg.bbox and seg.contours:
        x, y, bw, bh = seg.bbox
        cnt = seg.contours[0]
        touch = {
            "left": x <= 1,
            "top": y <= 1,
            "right": (x + bw) >= w - 1,
            "bottom": (y + bh) >= h - 1,
        }
        touching = [k for k, v in touch.items() if v]
        if len(touching) >= 1 and area > th["product_small_area_ratio"]:
            xs = cnt[:, :, 0]
            ys = cnt[:, :, 1]
            edge_px = int(((xs <= 1).sum() + (xs >= w - 2).sum() + (ys <= 1).sum() + (ys >= h - 2).sum()))
            if edge_px > 0.25 * len(cnt.reshape(-1, 2)):
                cut_off = True
        if cut_off:
            anomalies.append("PRODUCT_CUT_OFF")
            issues.append("Product is partially cut off / important parts outside the image.")

    if area < th["product_min_area_ratio"]:
        visibility = 15.0
    else:
        visibility = 100.0
        visibility -= 30.0 if area < th["product_small_area_ratio"] else 0.0
        visibility -= 40.0 if cut_off else 0.0
        visibility = float(np.clip(visibility, 0.0, 100.0))

    metrics = {
        "visibility": round(visibility, 1),
        "area_ratio": round(area, 4),
        "cut_off": cut_off,
        "bbox": list(seg.bbox) if seg.bbox else None,
    }
    return metrics, anomalies, issues

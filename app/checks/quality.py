"""Section 4.1 - Image quality anomalies.

Blur, resolution, exposure, contrast, noise, pixelation, compression
artifacts, corrupted image. Sab classic OpenCV metrics - koi LLM nahi.

Exposure/contrast/lighting product region par napti hai (agar segmentation
mask ho) - warna pure-white catalog background ko "overexposed" nahi mante.
"""

from __future__ import annotations

import cv2
import numpy as np


def _score_range(value: float, bad: float, good: float) -> float:
    """bad (score 0) se good (score 100) tak linear map, clamp [0,100]."""
    if good == bad:
        return 100.0
    score = (value - bad) / (good - bad) * 100.0
    return float(np.clip(score, 0.0, 100.0))


def _score_inverse(value: float, good: float, bad: float) -> float:
    """Zyada value achi ho to (good -> 100, bad -> 0)."""
    return _score_range(value, bad, good)


def blockiness_ratio(gray: np.ndarray) -> float:
    """JPEG blocking estimate: 8px grid edges vs 1px edges ki strength ratio."""
    g = gray.astype(np.float32)
    h, w = g.shape
    if h < 16 or w < 16:
        return 1.0
    col8 = np.abs(np.diff(g, axis=1))[:, 7::8]
    col1 = np.abs(np.diff(g, axis=1))[:, 0::1]
    row8 = np.abs(np.diff(g, axis=0))[7::8, :]
    row1 = np.abs(np.diff(g, axis=0))[0::1, :]
    base8 = float(col8.mean() + row8.mean()) / 2.0 + 1e-6
    base1 = float(col1.mean() + row1.mean()) / 2.0 + 1e-6
    return base8 / base1


def noise_level(gray: np.ndarray) -> float:
    """Median absolute deviation of fine detail - noise estimate."""
    smooth = cv2.medianBlur(gray, 3)
    dev = cv2.absdiff(gray, smooth)
    return float(np.median(dev))


def _lighting_score(mean_val: float) -> float:
    """Catalog photos (white bg) ko fair score - sirf extreme dark/bright ko penalty."""
    if mean_val < 50.0:
        return float(np.clip(mean_val / 50.0 * 100.0, 0.0, 100.0))
    if mean_val > 242.0:
        return float(np.clip(100.0 - (mean_val - 242.0) * 6.0, 0.0, 100.0))
    return float(np.clip(100.0 - abs(mean_val - 140.0) * 0.12, 0.0, 100.0))


def distortion_ratio(product_mask: np.ndarray | None) -> float:
    """Estimate perspective/warping from the largest product quadrilateral.

    A value of 1.0 is rectangular. Larger values mean opposite edges have
    increasingly different lengths, which is a useful conservative signal for
    a strongly skewed or warped catalog product region.
    """
    if product_mask is None or product_mask.size == 0:
        return 1.0
    contours, _ = cv2.findContours(product_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1.0
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return 1.0
    polygon = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    if len(polygon) != 4:
        return 1.0
    points = polygon.reshape(4, 2).astype(np.float32)
    sides = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    if np.any(sides <= 1.0):
        return 1.0
    opposite = [max(sides[0], sides[2]) / min(sides[0], sides[2]),
                max(sides[1], sides[3]) / min(sides[1], sides[3])]
    return float(max(opposite))


def evaluate_quality(
    img: np.ndarray,
    th: dict,
    product_mask: np.ndarray | None = None,
) -> tuple[dict, list[str], list[str]]:
    """Returns (metrics, anomaly_types, issues)."""
    anomalies: list[str] = []
    issues: list[str] = []

    if img is None or img.size == 0 or min(img.shape[:2]) < 10:
        return {}, ["CORRUPTED_IMAGE"], ["Image is missing or corrupted."]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # --- sharpness (Laplacian variance) ---
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var < th["blur_laplacian_severe"]:
        anomalies.append("EXCESSIVE_BLUR")
        issues.append("Image is excessively blurry (details are unreadable).")
    elif lap_var < th["blur_laplacian_min"]:
        anomalies.append("BLUR")
        issues.append("Image is noticeably blurry.")
    sharpness = _score_range(np.log1p(lap_var), np.log1p(15.0), np.log1p(600.0))

    # --- resolution ---
    min_side = min(h, w)
    if min_side < th["min_resolution"]:
        anomalies.append("LOW_RESOLUTION")
        issues.append(f"Low resolution ({w}x{h} px).")
    resolution = _score_range(float(min_side), 200.0, float(th["good_resolution"]))

    # --- exposure / lighting / contrast: product region par ---
    region = gray
    if product_mask is not None and product_mask.shape[:2] == gray.shape:
        masked = gray[product_mask > 0]
        if masked.size >= 0.02 * gray.size:
            region = masked

    mean_val = float(region.mean())
    if mean_val > th["exposure_mean_max"]:
        anomalies.append("OVEREXPOSURE")
        issues.append("Image is overexposed (too bright).")
    elif mean_val < th["exposure_mean_min"]:
        anomalies.append("UNDEREXPOSURE")
        issues.append("Image is underexposed (too dark).")
    lighting = _lighting_score(mean_val)

    # --- contrast ---
    contrast_val = float(region.std())
    if contrast_val < th["contrast_min"]:
        anomalies.append("POOR_CONTRAST")
        issues.append("Poor contrast - product is hard to distinguish.")
    contrast = _score_range(contrast_val, 12.0, 65.0)

    # --- noise (global) ---
    noise_val = noise_level(gray)
    if noise_val > th["noise_median_max"]:
        anomalies.append("NOISE")
        issues.append("Excessive noise / grain in the image.")
    noise_score = _score_inverse(noise_val, 3.0, 30.0)

    # --- pixelation + compression artifacts (blockiness) ---
    ratio = blockiness_ratio(gray)
    if ratio > th["blockiness_ratio_max"] + 0.6 and min_side < th["min_resolution"] * 2:
        anomalies.append("PIXELATION")
        issues.append("Pixelated image (visible block edges).")
    elif ratio > th["blockiness_ratio_max"]:
        anomalies.append("COMPRESSION_ARTIFACTS")
        issues.append("Visible compression artifacts.")

    # --- geometric distortion / perspective warp ---
    distortion = distortion_ratio(product_mask)
    if distortion > th.get("distortion_side_ratio_max", 1.35):
        anomalies.append("DISTORTION")
        issues.append("Product edges appear strongly distorted or perspective-warped.")

    # --- corrupted: gray sab ek rang ---
    if float(gray.std()) < 1.0:
        anomalies.append("CORRUPTED_IMAGE")
        issues.append("Image appears corrupted (no visual content).")

    metrics = {
        "sharpness": round(sharpness, 1),
        "resolution": round(resolution, 1),
        "lighting": round(lighting, 1),
        "contrast": round(contrast, 1),
        "noise": round(noise_score, 1),
        "lap_var": round(lap_var, 2),
        "mean_val": round(mean_val, 1),
        "blockiness": round(ratio, 3),
        "distortion": round(distortion, 3),
        "width": w,
        "height": h,
    }
    return metrics, anomalies, issues

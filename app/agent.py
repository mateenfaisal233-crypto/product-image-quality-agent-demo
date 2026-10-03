"""Product-Aware Image Anomaly Detection + Quality Evaluation Agent.

Pipeline (spec sections 2-7 ke mutabiq):
  image + product JSON -> independent checks -> scores -> status -> JSON report.

Critical rule (section 9): product correctness aur image quality INDEPENDENTLY
evaluate hoti hain - khoobsurat tasveer + ghalat product = WRONG_PRODUCT,
sahi product + blurry = VALID_BUT_LOW_QUALITY.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any

import numpy as np

from .checks import background as background_check
from .checks import composition as composition_check
from .checks import correctness as correctness_check
from .checks import quality as quality_check
from .checks import text_ocr as text_ocr_check
from .checks import vlm as vlm_check
from .checks import visibility as visibility_check
from .classify import classify
from .config import load_config, project_root
from .input_loader import load_image, load_product_json, product_description
from .report import build_report, make_explanation
from .scoring import compute_overall, image_quality_score, normalize_weights

WRONG_STATUSES = {"WRONG_PRODUCT", "WRONG_BRAND", "WRONG_VARIANT", "WRONG_CATEGORY"}

# VLM (Gemini) slow ho sakta hai - CV checks ke sath sath background mein chalta
# hai, report banate waqt uska result join ho jata hai (speed fix).
_VLM_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm")


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


class ProductQualityAgent:
    def __init__(self, config: dict | None = None, config_path: str | Path | None = None):
        self.cfg = config if config is not None else load_config(config_path)
        self._clip = None
        self._clip_tried = False
        self._clip_lock = threading.Lock()  # warmup + request race guard

    # ------------------------------------------------------------- backends
    @property
    def weights(self) -> dict:
        return normalize_weights(self.cfg["scoring"]["weights"])

    def _get_clip(self):
        if self.cfg["correctness"].get("backend", "auto") == "heuristics":
            return None
        # Hamesha lock ke andar - load complete hone tak doosra thread wait kare,
        # warna health "CLIP off" dikhata hai (loading race).
        with self._clip_lock:
            if self._clip_tried:
                return self._clip
            self._clip_tried = True
            try:
                self._clip = correctness_check.ClipBackend(self.cfg["correctness"]["clip_model"])
                if not self._clip.available:
                    self._clip = None
            except Exception:  # noqa: BLE001
                self._clip = None
        return self._clip

    def clip_available(self) -> bool:
        return self._get_clip() is not None

    def ocr_available(self) -> bool:
        return text_ocr_check.engine_available()

    # ------------------------------------------------------------- pipeline
    def analyze(
        self,
        image: str | Path | dict | np.ndarray | None = None,
        product: str | Path | dict | None = None,
        *,
        base64_str: str | None = None,
        url: str | None = None,
    ) -> dict[str, Any]:
        th = self.cfg["thresholds"]

        # --- inputs ---
        if isinstance(image, np.ndarray):
            img = image
        else:
            timeout = self.cfg["runtime"]["url_timeout_sec"]
            img = load_image(image, base64_str=base64_str, url=url, timeout=timeout)
        prod = load_product_json(product)

        # --- VLM validation (background thread - CV ke sath sath) ---
        vlm_future: Future | None = None
        vlm_deadline = 0.0
        if vlm_check.is_configured(self.cfg):
            vlm_timeout = float(self.cfg["vlm"].get("timeout_sec", 10))
            vlm_deadline = time.monotonic() + vlm_timeout
            vlm_future = _VLM_POOL.submit(vlm_check.validate, img, prod, self.cfg, int(vlm_timeout))

        # --- speed: sirf neeche resize (original ki resolution record rehti hai) ---
        orig_h, orig_w = img.shape[:2]
        max_side = int(self.cfg["runtime"].get("image_max_side", 1600))
        if max(orig_h, orig_w) > max_side:
            scale = max_side / float(max(orig_h, orig_w))
            import cv2

            img = cv2.resize(img, (int(orig_w * scale), int(orig_h * scale)), interpolation=cv2.INTER_AREA)

        # --- independent checks (segmentation pehle: quality metrics product
        # region par napti hain, white background bias nahi) ---
        seg = visibility_check.segment_product(img)
        q_metrics, q_anom, q_issues = quality_check.evaluate_quality(img, th, product_mask=seg.mask)
        v_metrics, v_anom, v_issues = visibility_check.evaluate_visibility(seg, img, th)
        c_metrics, c_anom, c_issues = composition_check.evaluate_composition(seg, img, th)
        b_metrics, b_anom, b_issues = background_check.evaluate_background(img, seg, th)

        ocr_results: list[dict] = []
        if self.cfg["ocr"].get("enabled", True):
            ocr_results = text_ocr_check.read_text(img)
        t_metrics, t_anom, t_issues = text_ocr_check.evaluate_text(ocr_results, th)

        clip = self._get_clip()
        # Do not turn a degraded image into a product mismatch solely because
        # CLIP similarity is unreliable when blur/low resolution removes the
        # visual evidence. OCR-based mismatches remain fully active.
        degraded = {"BLUR", "EXCESSIVE_BLUR", "LOW_RESOLUTION", "TEXT_UNREADABLE"}
        corr = correctness_check.evaluate_correctness(
            img,
            prod,
            ocr_results,
            clip,
            th,
            allow_visual_mismatch=not (degraded & set(q_anom + t_anom)),
        )

        # --- anomaly list (WRONG_* pehle, phir quality) ---
        anomalies: list[str] = []
        if corr["wrong_status"] in WRONG_STATUSES:
            anomalies.append(corr["wrong_status"])
        anomalies += q_anom + v_anom + c_anom + b_anom + t_anom
        anomalies = _dedupe(anomalies)

        issues = _dedupe(q_issues + v_issues + c_issues + b_issues + t_issues + corr["issues"])

        # --- scores (sab 0-100) ---
        quality_parts = {
            "sharpness": q_metrics.get("sharpness", 75.0),
            "resolution": q_metrics.get("resolution", 75.0),
            "lighting": q_metrics.get("lighting", 75.0),
            "contrast": q_metrics.get("contrast", 75.0),
            "noise": q_metrics.get("noise", 75.0),
        }
        scores: dict[str, float] = {
            "product_match": corr["product_match"],
            "image_quality": image_quality_score(quality_parts),
            "sharpness": quality_parts["sharpness"],
            "resolution": quality_parts["resolution"],
            "visibility": v_metrics.get("visibility", 75.0),
            "composition": c_metrics.get("composition", 75.0),
            "background": b_metrics.get("background", 75.0),
            "lighting": quality_parts["lighting"],
            "text_readability": t_metrics.get("text_readability", 75.0),
        }
        scores["overall"] = compute_overall(scores, self.weights)

        # --- classification ---
        wrong_status = corr["wrong_status"] if not corr["is_correct"] else None
        status = classify(wrong_status, anomalies, scores, self.cfg)

        # --- optional VLM validation layer - sirf notes, status nahi badalta ---
        # Budget: submit ke waqt se deadline - CV chalte chalte VLM ko time milta
        # hai, report kabhi VLM ki wajah se 20-30s nahi rukta.
        if vlm_future is not None:
            try:
                wait = max(0.1, vlm_deadline - time.monotonic() + 1.0)
                for note in vlm_future.result(timeout=wait):
                    if note not in issues:
                        issues.append(note)
            except FuturesTimeoutError:
                issues.append("VLM validation skipped (timeout).")

        explanation = make_explanation(status, corr["is_correct"], issues)

        return build_report(
            product=prod,
            status=status,
            is_correct=corr["is_correct"],
            anomaly_types=anomalies,
            scores=scores,
            detected_product=corr["detected_product"],
            issues=issues,
            explanation=explanation,
        )

    # convenience -----------------------------------------------------------
    def analyze_pair(self, image_path: str | Path, product_json_path: str | Path) -> dict:
        return self.analyze(image_path, product_json_path)


def default_agent() -> ProductQualityAgent:
    """Test/CLI ke liye shared agent instance."""
    return ProductQualityAgent()


__all__ = ["ProductQualityAgent", "default_agent", "product_description", "project_root"]

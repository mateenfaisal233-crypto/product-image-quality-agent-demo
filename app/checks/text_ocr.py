"""Section 4.5 - Text and packaging (OCR) + text readability score.

Engine: rapidocr (onnx, offline). Agar available na ho to sab neutral
rehenge aur koi TEXT anomaly nahi lagega (documented behavior).
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import numpy as np

_ENGINE = None
_ENGINE_FAILED = False


def _get_engine():
    global _ENGINE, _ENGINE_FAILED
    if _ENGINE is not None or _ENGINE_FAILED:
        return _ENGINE
    try:
        from rapidocr_onnxruntime import RapidOCR

        _ENGINE = RapidOCR()
    except Exception:  # noqa: BLE001
        _ENGINE_FAILED = True
        _ENGINE = None
    return _ENGINE


def engine_available() -> bool:
    return _get_engine() is not None


def read_text(img: np.ndarray) -> list[dict]:
    """OCR chalao -> [{text, score}], score 0..100."""
    engine = _get_engine()
    if engine is None:
        return []
    try:
        result, _elapsed = engine(img)
    except Exception:  # noqa: BLE001
        return []
    if not result:
        return []
    out = []
    for item in result:
        box, text, score = item[0], item[1], item[2]
        text = (text or "").strip()
        if not text:
            continue
        out.append({"text": text, "score": round(float(score) * 100.0, 1), "box": box})
    return out


def normalize_token(token: str) -> str:
    """Lowercase + accent-free (Maalox/maalox, comprimé/comprime)."""
    token = unicodedata.normalize("NFKD", token)
    token = "".join(c for c in token if not unicodedata.combining(c))
    token = re.sub(r"[^a-z0-9]", "", token.lower())
    return token


def ocr_tokens(ocr_results: list[dict]) -> list[str]:
    tokens: list[str] = []
    for entry in ocr_results:
        for raw in re.split(r"[^\w]+", entry["text"]):
            tok = normalize_token(raw)
            if tok:
                tokens.append(tok)
    return tokens


def ocr_text(ocr_results: list[dict]) -> str:
    return " ".join(e["text"] for e in ocr_results)


def evaluate_text(
    ocr_results: list[dict],
    th: dict,
) -> tuple[dict, list[str], list[str]]:
    """OCR results se readability score + anomalies."""
    anomalies: list[str] = []
    issues: list[str] = []

    if not ocr_results:
        # Koi text visible nahi (ya OCR unavailable) -> neutral, spec:
        # "When packaging text is visible..." - text hi nahi to check nahi.
        return {"text_readability": 75.0, "text_found": False, "mean_conf": 0.0}, anomalies, issues

    confs = [e["score"] for e in ocr_results]
    mean_conf = float(np.mean(confs))
    chars = sum(len(e["text"]) for e in ocr_results)

    readability = float(np.clip(mean_conf, 0.0, 100.0))
    if chars < 4:
        readability = min(readability, 40.0)

    if mean_conf < th["ocr_confidence_min"]:
        anomalies.append("TEXT_UNREADABLE")
        issues.append("Packaging text is not readable / low confidence OCR.")

    return {
        "text_readability": round(readability, 1),
        "text_found": True,
        "mean_conf": round(mean_conf, 1),
        "char_count": chars,
    }, anomalies, issues

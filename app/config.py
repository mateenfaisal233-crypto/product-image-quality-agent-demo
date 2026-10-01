"""Config loading: config.yaml + defaults ka deep merge."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "scoring": {
        "weights": {
            "product_match": 0.25,
            "image_quality": 0.15,
            "sharpness": 0.10,
            "resolution": 0.08,
            "visibility": 0.12,
            "composition": 0.10,
            "background": 0.08,
            "lighting": 0.07,
            "text_readability": 0.05,
        }
    },
    "thresholds": {
        "blur_laplacian_min": 60.0,
        "blur_laplacian_severe": 25.0,
        "min_resolution": 400,
        "good_resolution": 1000,
        "exposure_mean_min": 45,
        "exposure_mean_max": 240,
        "contrast_min": 35.0,
        "noise_median_max": 18.0,
        "blockiness_ratio_max": 1.6,
        "product_min_area_ratio": 0.02,
        "product_small_area_ratio": 0.05,
        "edge_margin_min": 0.02,
        "center_offset_max": 0.25,
        "empty_space_max": 0.75,
        "bg_edge_density_max": 0.12,
        "bg_color_std_max": 30.0,
        "ocr_confidence_min": 45.0,
        "clip_min_similarity": 0.20,
    },
    "classification": {
        "valid_min_overall": 70,
        "low_quality_max_overall": 50,
    },
    "correctness": {"backend": "auto", "clip_model": "openai/clip-vit-base-patch32"},
    "ocr": {"enabled": True, "engine": "rapidocr"},
    "vlm": {
        "enabled": False,
        "provider": "groq",  # groq | gemini
        "model": "qwen/qwen3.8-27b",
        "timeout_sec": 10,
        "api_key_env": "GROQ_API_KEY",
    },
    "runtime": {
        "image_max_side": 1600,
        "url_timeout_sec": 15,
        "model_cache_dir": ".modelcache",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def setup_model_cache(cfg: dict | None = None) -> Path:
    """HF/model cache folder - project ke andar taaki delete karte sab chala jaye."""
    cache = project_root() / (cfg or DEFAULTS)["runtime"]["model_cache_dir"]
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # Agar model already cache mein hai to HF hub ki network ping skip (fast boot)
    hub = cache / "hub"
    if hub.exists() and any(hub.glob("models--*")):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return cache


def _load_dotenv(path: Path) -> None:
    """Chhota .env loader (KEY=VALUE) - sirf environment set karta hai."""
    if not path.exists():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)
    except OSError:
        pass


def load_config(path: str | Path | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    _load_dotenv(project_root() / ".env")  # GEMINI_API_KEY waghera
    if path is None:
        candidate = project_root() / "config.yaml"
        path = candidate if candidate.exists() else None
    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            user_cfg = yaml.safe_load(fh) or {}
        cfg = _deep_merge(cfg, user_cfg)
    setup_model_cache(cfg)
    return cfg

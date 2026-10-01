"""Shared fixtures: sample images (synthetic) + agent instance."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
for p in (str(ROOT), str(EXAMPLES)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Model cache project folder ke andar (delete = sab gaya)
os.environ.setdefault("HF_HOME", str(ROOT / ".modelcache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / ".modelcache"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(ROOT / ".modelcache" / "hub"))


@pytest.fixture(scope="session")
def samples_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("samples")
    from make_samples import build_all

    build_all(out)
    return out


@pytest.fixture(scope="session")
def product_json() -> dict:
    from make_samples import PRODUCT_JSON

    return dict(PRODUCT_JSON)


@pytest.fixture(scope="session")
def agent():
    from app.agent import ProductQualityAgent
    from app.config import load_config

    # Unit tests VLM ke bina fast + offline chalte hain
    cfg = load_config()
    cfg["vlm"]["enabled"] = False
    return ProductQualityAgent(config=cfg)


@pytest.fixture(scope="session")
def vlm_agent():
    """VLM (Gemini) ke sath agent - sirf integration test use karta hai."""
    from app.agent import ProductQualityAgent

    return ProductQualityAgent()


@pytest.fixture(scope="session")
def analyze(agent, samples_dir, product_json):
    """analyze(image_name) -> report dict."""

    def _analyze(image_name: str) -> dict:
        return agent.analyze(samples_dir / image_name, product_json)

    return _analyze

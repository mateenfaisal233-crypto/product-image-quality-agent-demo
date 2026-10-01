"""VLM validation layer (Groq) - real API integration test.

Sirf tab chalta hai jab .env mein VLM key ho (spec section 8:
VLM optional validation layer hai - notes issues[] mein jate hain,
status/scores nahi badalte).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.checks import vlm as vlm_mod

from helpers import assert_valid_report


def _has_vlm_key() -> bool:
    env_file = Path(__file__).resolve().parents[1] / ".env"
    env_text = env_file.read_text(encoding="utf-8") if env_file.exists() else ""
    for name in ("GROQ_API_KEY", "GEMINI_API_KEY"):
        if os.environ.get(name) or f"{name}=" in env_text:
            return True
    return False


@pytest.mark.skipif(not _has_vlm_key(), reason="VLM key nahi mili (.env missing)")
def test_vlm_notes_add_without_changing_status(vlm_agent, samples_dir, product_json):
    from app.config import load_config

    cfg = load_config()
    assert vlm_mod.is_configured(cfg), "VLM configured hona chahiye (.env + enabled)"
    report = vlm_agent.analyze(samples_dir / "maalox.jpg", product_json)
    assert_valid_report(report)
    # Status system se aata hai - VLM sirf validation notes deta hai
    assert report["status"] == "WRONG_PRODUCT"
    assert any("VLM validation" in note for note in report["issues"]), report["issues"]

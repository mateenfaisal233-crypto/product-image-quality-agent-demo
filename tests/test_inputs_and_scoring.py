"""Input validation + scoring unit tests (spec sections 2 & 5)."""

from __future__ import annotations

import pytest

from app.input_loader import InputError, load_image, load_product_json, product_description
from app.scoring import SCORE_KEYS, compute_overall, normalize_weights


def test_missing_image_raises():
    with pytest.raises(InputError):
        load_image()


def test_missing_file_raises(tmp_path):
    with pytest.raises(InputError):
        load_image(tmp_path / "nope.jpg")


def test_corrupted_bytes_raise():
    import base64

    b64 = base64.b64encode(b"this is not an image").decode()
    with pytest.raises(InputError):
        load_image(base64_str=b64)


def test_base64_roundtrip(samples_dir):
    import base64

    b64 = base64.b64encode((samples_dir / "doliprane_ok.jpg").read_bytes()).decode()
    img = load_image(base64_str="data:image/jpeg;base64," + b64)
    assert img.shape[0] > 100


def test_product_json_dict():
    data = load_product_json({"id": 7, "name": "Doliprane"})
    assert data["product_id"] == 7


def test_product_json_missing_optional_fields_ok():
    data = load_product_json({"name": "Doliprane 1000 mg"})
    assert data["name"]


def test_product_json_invalid():
    with pytest.raises(InputError):
        load_product_json("[not an object]")


def test_product_json_missing():
    with pytest.raises(InputError):
        load_product_json(None)


def test_product_description():
    text = product_description({"name": "Doliprane 1000 mg", "brand": "Sanofi"})
    assert "Doliprane" in text and "Sanofi" in text


def test_weights_normalize():
    weights = normalize_weights({"product_match": 2, "sharpness": 2})
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert abs(weights["product_match"] - 0.5) < 1e-9


def test_overall_weighted_sum():
    scores = {k: 80.0 for k in SCORE_KEYS}
    weights = {k: (1.0 / len(SCORE_KEYS)) for k in SCORE_KEYS}
    assert abs(compute_overall(scores, weights) - 80.0) < 0.01


def test_config_weights_present(agent):
    w = agent.weights
    assert set(w) == set(SCORE_KEYS)
    assert abs(sum(w.values()) - 1.0) < 1e-6

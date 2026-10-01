"""Edge cases: minimal JSON, corrupted files, URL input, tiny/directory input."""

from __future__ import annotations

import functools
import http.server
import socketserver
import threading

import pytest

from app.input_loader import InputError

from helpers import assert_valid_report


def test_minimal_product_json_works(agent, samples_dir):
    """Sirf naam wala JSON (koi brand/category nahi) - spec 2.3: optional fields."""
    report = agent.analyze(samples_dir / "doliprane_ok.jpg", {"name": "Doliprane 1000 mg"})
    assert_valid_report(report)
    assert report["status"] == "VALID", report
    assert report["product_id"] == ""


def test_product_json_with_missing_id(agent, samples_dir):
    report = agent.analyze(
        samples_dir / "maalox.jpg",
        {"name": "Doliprane 1000 mg Comprimé", "category": "Medicament", "brand": "Sanofi"},
    )
    assert_valid_report(report)
    assert report["status"] == "WRONG_PRODUCT"


def test_corrupted_image_file(agent, tmp_path, product_json):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"this is definitely not an image")
    with pytest.raises(InputError):
        agent.analyze(bad, product_json)


def test_directory_instead_of_image(agent, samples_dir, product_json):
    with pytest.raises(InputError):
        agent.analyze(samples_dir, product_json)


def test_tiny_image_rejected(agent, tmp_path, product_json):
    from PIL import Image

    p = tmp_path / "tiny.png"
    Image.new("RGB", (8, 8), "white").save(p)
    with pytest.raises(InputError):
        agent.analyze(p, product_json)


def test_image_from_url(agent, samples_dir, product_json):
    """URL input (spec 2.1) - local HTTP server se."""
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(samples_dir))
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    httpd.daemon_threads = True
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        report = agent.analyze(None, product_json, url=f"http://127.0.0.1:{port}/doliprane_ok.jpg")
        assert_valid_report(report)
        assert report["status"] == "VALID"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_blurred_product_still_counted_correct(agent, samples_dir, product_json):
    """Section 9: blurry + sahi product -> WRONG_* nahi hona chahiye."""
    report = agent.analyze(samples_dir / "doliprane_blurry.jpg", product_json)
    assert report["is_correct_product"] is True
    assert report["status"] not in {"WRONG_PRODUCT", "WRONG_VARIANT", "WRONG_BRAND", "WRONG_CATEGORY"}

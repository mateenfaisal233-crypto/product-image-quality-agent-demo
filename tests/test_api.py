"""FastAPI interface tests (Section 10: API deliverable)."""

from __future__ import annotations

import json

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

from helpers import assert_valid_report  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["ocr_available"], bool)
    assert isinstance(body["clip_available"], bool)


def test_config_exposes_weights(client):
    resp = client.get("/config")
    assert resp.status_code == 200
    weights = resp.json()["weights"]
    assert abs(sum(weights.values()) - 1.0) < 0.02


def test_analyze_multipart(client, samples_dir, product_json):
    img_path = samples_dir / "doliprane_ok.jpg"
    with open(img_path, "rb") as fh:
        resp = client.post(
            "/analyze",
            files={"file": ("doliprane_ok.jpg", fh, "image/jpeg")},
            data={"product_json": json.dumps(product_json)},
        )
    assert resp.status_code == 200, resp.text
    report = resp.json()
    assert_valid_report(report)
    assert report["status"] == "VALID"


def test_analyze_base64_payload(client, samples_dir, product_json):
    import base64

    b64 = base64.b64encode((samples_dir / "maalox.jpg").read_bytes()).decode()
    resp = client.post("/analyze", json={"image_base64": b64, "product": product_json})
    assert resp.status_code == 200, resp.text
    report = resp.json()
    assert report["status"] == "WRONG_PRODUCT"


def test_analyze_bad_request(client):
    resp = client.post("/analyze")
    assert resp.status_code == 400


def test_analyze_image_url(client, samples_dir, product_json):
    """Spec section 2.1: image URL input API par bhi kaam kare."""
    import functools
    import http.server
    import socketserver
    import threading

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(samples_dir))
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    httpd.daemon_threads = True
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        resp = client.post(
            "/analyze",
            json={"image_url": f"http://127.0.0.1:{port}/doliprane_ok.jpg", "product": product_json},
        )
        assert resp.status_code == 200, resp.text
        report = resp.json()
        assert_valid_report(report)
        assert report["status"] == "VALID"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_analyze_json_without_image_is_400(client, product_json):
    resp = client.post("/analyze", json={"product": product_json})
    assert resp.status_code == 400

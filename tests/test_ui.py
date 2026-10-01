"""Web UI tests - GET / (upload UI) + sample files."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_ui_index_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert b"Product Image Anomaly Detection" in resp.content
    assert b"Analyze" in resp.content
    assert b"Product information (JSON)" in resp.content


def test_samples_static_served(client):
    resp = client.get("/samples/doliprane_ok.jpg")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/")
    resp = client.get("/samples/product_doliprane.json")
    assert resp.status_code == 200
    assert resp.json()["name"].startswith("Doliprane")

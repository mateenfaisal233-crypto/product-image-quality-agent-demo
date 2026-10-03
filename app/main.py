"""FastAPI interface - Section 10: "API or runnable interface".

Endpoints:
  POST /analyze      multipart: image file + product_json (form field)
  POST /analyze      JSON: {"image_base64": "...", "product": {...}}
  GET  /health       backends ki status
  GET  /config       effective config + weights
"""

from __future__ import annotations

import json as _json
from contextlib import asynccontextmanager
from threading import Thread
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .agent import ProductQualityAgent
from .checks import text_ocr as text_ocr_check
from .config import load_config, project_root
from .input_loader import InputError, decode_image_bytes, load_product_json

_agent: ProductQualityAgent | None = None


def get_agent() -> ProductQualityAgent:
    global _agent
    if _agent is None:
        _agent = ProductQualityAgent()
    return _agent


def _warmup() -> None:
    """Background warmup - pehli user click 30-40s load mein na lage."""
    try:
        agent = get_agent()
        agent.clip_available()  # CLIP weights load
        text_ocr_check.engine_available()  # RapidOCR engine load
    except Exception:  # noqa: BLE001
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    Thread(target=_warmup, daemon=True).start()
    yield


app = FastAPI(
    title="Product Image Anomaly Detection & Quality Agent",
    version="1.0.0",
    description="Product-aware image anomaly detection + quality evaluation (spec compliant).",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    agent = get_agent()
    vlm = agent.cfg.get("vlm", {})
    return {
        "status": "ok",
        "ocr_available": agent.ocr_available(),
        "clip_available": agent.clip_available(),
        "vlm_enabled": bool(vlm.get("enabled", False)),
        "vlm_provider": vlm.get("provider", ""),
        "vlm_model": vlm.get("model", ""),
    }


@app.get("/config")
def get_config() -> dict[str, Any]:
    cfg = load_config()
    return {
        "weights": cfg["scoring"]["weights"],
        "thresholds": cfg["thresholds"],
        "classification": cfg["classification"],
        "correctness_backend": cfg["correctness"]["backend"],
        "vlm": {k: v for k, v in cfg["vlm"].items() if k != "api_key"},
    }


@app.post("/analyze")
async def analyze(request: Request) -> JSONResponse:
    """Do tarika (ek hi endpoint):
    1) multipart/form-data: 'file' (image) + 'product_json' (JSON string)
    2) application/json: {"image_base64" | "image_url": "...", "product": {...}}
       (spec section 2.1: file / URL / base64 teeno input forms)
    """
    agent = get_agent()
    content_type = (request.headers.get("content-type") or "").lower()
    try:
        if content_type.startswith("application/json"):
            data = await request.json()
            if data.get("image_base64"):
                report = agent.analyze(
                    base64_str=data.get("image_base64"),
                    product=data.get("product") or {},
                )
            elif data.get("image_url"):
                report = agent.analyze(
                    url=data.get("image_url"),
                    product=data.get("product") or {},
                )
            else:
                raise InputError("Provide 'image_base64' or 'image_url'.")
        else:
            form = await request.form()
            payload = form.get("payload")
            if payload:
                data = _json.loads(str(payload))
                report = agent.analyze(
                    base64_str=data.get("image_base64"),
                    product=data.get("product") or {},
                )
            else:
                upload = form.get("file")
                product_json = form.get("product_json")
                if upload is None or not product_json:
                    raise InputError("Provide (file + product_json) or JSON {image_base64, product}.")
                raw = await upload.read()
                img = decode_image_bytes(raw)
                product = load_product_json(str(product_json))
                report = agent.analyze(img, product)
    except InputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc
    return JSONResponse(report)


@app.get("/")
def ui_index() -> FileResponse:
    """Real web UI - image upload + product form + live report."""
    return FileResponse(project_root() / "app" / "static" / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)


# Routes se baad mounts (taake /analyze, /health priority rakhein)
_samples_dir = project_root() / "examples" / "samples"
if not _samples_dir.is_dir():
    # The samples are synthetic and are also used by the browser UI. Generate
    # them on first start so a fresh clone behaves like the documented setup.
    try:
        from examples.make_samples import build_all

        build_all(_samples_dir)
    except Exception:  # noqa: BLE001
        pass
if _samples_dir.is_dir():
    app.mount("/samples", StaticFiles(directory=str(_samples_dir)), name="samples")

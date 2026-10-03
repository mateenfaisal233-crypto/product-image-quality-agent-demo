"""Live demo UI - Streamlit (spec Section 10: runnable interface).

Streamlit Community Cloud ke liye entry point:
  streamlit run streamlit_app.py

FastAPI version (cli.py serve) alag hai - yeh usi agent ko wrap karta hai.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Product Image Quality Agent",
    page_icon="\U0001f4e6",
    layout="wide",
)

# Streamlit Secrets -> environment (Groq VLM notes ke liye). Absent ho to VLM skip.
try:
    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, str(_value))
except Exception:  # noqa: BLE001
    pass

import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "examples" / "samples"
EXAMPLES = sorted(p.name for p in SAMPLES.glob("*.jpg"))
DEFAULT_EXAMPLE = "doliprane_ok.jpg" if "doliprane_ok.jpg" in EXAMPLES else (EXAMPLES[0] if EXAMPLES else None)
DEFAULT_JSON = (SAMPLES / "product_doliprane.json").read_text(encoding="utf-8") if (SAMPLES / "product_doliprane.json").exists() else "{}"

STATUS_ICON = {
    "VALID": "\U0001f7e2",
    "VALID_BUT_LOW_QUALITY": "\U0001f7e1",
    "LOW_IMAGE_QUALITY": "\U0001f7e0",
    "ANOMALY": "\U0001f534",
    "WRONG_PRODUCT": "\U0001f534",
    "WRONG_BRAND": "\U0001f534",
    "WRONG_VARIANT": "\U0001f534",
    "WRONG_CATEGORY": "\U0001f534",
}

SCORE_LABELS = [
    ("product_match", "Product match"),
    ("image_quality", "Image quality"),
    ("sharpness", "Sharpness"),
    ("resolution", "Resolution"),
    ("visibility", "Visibility"),
    ("composition", "Composition"),
    ("background", "Background"),
    ("lighting", "Lighting"),
    ("text_readability", "Text readability"),
]


@st.cache_resource(show_spinner=False)
def get_agent():
    from app.agent import ProductQualityAgent

    return ProductQualityAgent()


def render_report(res: dict) -> None:
    status = str(res.get("status", "?"))
    st.markdown(f"### {STATUS_ICON.get(status, '\u26aa')} {status}")

    c1, c2 = st.columns(2)
    c1.metric("Overall score", f"{res.get('scores', {}).get('overall', 0.0):.1f} / 100")
    c2.metric("Product ID", str(res.get("product_id", "-")))
    st.caption(
        f"Anomaly: {res.get('is_anomaly')} \u00b7 "
        f"Correct product: {res.get('is_correct_product')}"
    )

    scores = res.get("scores", {})
    st.markdown("**Scores**")
    for key, label in SCORE_LABELS:
        val = float(scores.get(key, 0.0))
        st.progress(min(max(val, 0.0), 100.0) / 100.0, text=f"{label}: {val:.1f}")

    issues = res.get("issues") or []
    if issues:
        st.markdown("**Issues**")
        for item in issues:
            st.markdown(f"- {item}")

    if res.get("explanation"):
        st.markdown("**Explanation**")
        st.info(res["explanation"])

    if res.get("detected_product"):
        st.markdown("**Detected product**")
        st.json(res["detected_product"])

    with st.expander("Raw JSON"):
        st.json(res)


st.title("Product Image Anomaly Detection & Quality Agent")
st.caption(
    "Upload a product image with its product JSON \u2192 status, 9 scores, "
    "issues and a plain explanation. Analysis takes ~5 seconds "
    "(first run downloads the CLIP model, ~1 minute)."
)

left, right = st.columns([5, 7], gap="large")

with left:
    st.subheader("Input")
    uploaded = st.file_uploader("Product image", type=["jpg", "jpeg", "png", "webp"])
    example = st.selectbox("Or pick an example", EXAMPLES, index=EXAMPLES.index(DEFAULT_EXAMPLE) if DEFAULT_EXAMPLE else 0)
    product_json = st.text_area("Product JSON (partial fields OK)", value=DEFAULT_JSON, height=240)

    input_image = None
    if uploaded is not None:
        buf = np.frombuffer(uploaded.getvalue(), np.uint8)
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if decoded is None:
            st.error("Could not read this image file.")
        else:
            input_image = decoded
            st.image(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), caption=uploaded.name, width="stretch")
    elif example:
        sample_path = SAMPLES / example
        if sample_path.exists():
            input_image = str(sample_path)
            st.image(str(sample_path), caption=f"Example: {example}", width="stretch")

    analyze_clicked = st.button("Analyze", type="primary", width="stretch")

with right:
    st.subheader("Report")

    if analyze_clicked:
        if input_image is None:
            st.warning("Upload an image or pick an example first.")
        else:
            try:
                product = json.loads(product_json)
            except json.JSONDecodeError as exc:
                st.error(f"Invalid product JSON: {exc}")
                product = None
            if product is not None:
                with st.spinner("Analyzing image\u2026"):
                    try:
                        st.session_state["result"] = get_agent().analyze(input_image, product)
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Analysis failed: {exc}")

    result = st.session_state.get("result")
    if result is None:
        st.info("Click **Analyze** to generate the report.")
    else:
        render_report(result)

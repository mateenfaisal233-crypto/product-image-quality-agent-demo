"""Live demo UI - Streamlit (spec Section 10: runnable interface).

Streamlit Community Cloud ke liye entry point:
  streamlit run streamlit_app.py

FastAPI version (cli.py serve) alag hai - yeh usi agent ko wrap karta hai.
"""

from __future__ import annotations

import html as html_lib
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

CSS = """
<style>
:root{
  --ink:#0f172a; --muted:#64748b; --line:#e5e9f0; --bg:#f5f7fb;
  --green:#16a34a; --amber:#d97706; --orange:#ea580c; --red:#dc2626; --blue:#2563eb;
}
[data-testid="stAppViewContainer"]{background:var(--bg);}
[data-testid="stAppViewContainer"] .main{background:var(--bg);}
.block-container{padding-top:1.6rem;max-width:1180px;padding-bottom:2.5rem;}

.hero{background:linear-gradient(135deg,#0f172a 0%,#1e3a8a 55%,#2563eb 100%);
  color:#fff;border-radius:16px;padding:1.5rem 1.9rem;margin:.2rem 0 1.2rem;
  box-shadow:0 6px 18px rgba(30,58,138,.20);}
.hero .eyebrow{font-size:.7rem;letter-spacing:.18em;opacity:.75;font-weight:700;}
.hero h1{font-size:1.5rem;margin:.4rem 0 .45rem;font-weight:750;line-height:1.25;}
.hero p{margin:0;font-size:.9rem;opacity:.85;max-width:820px;line-height:1.5;}

[data-testid="stVerticalBlockBorderWrapper"]{
  background:#fff !important;border-color:var(--line) !important;border-radius:16px !important;
  box-shadow:0 1px 4px rgba(15,23,42,.06);
  padding:1.2rem 1.35rem !important;}

.panel-title{font-size:.72rem;font-weight:800;letter-spacing:.14em;color:var(--muted);
  text-transform:uppercase;margin:.1rem 0 .8rem;}
.sect{font-size:.7rem;font-weight:800;letter-spacing:.14em;color:var(--muted);
  text-transform:uppercase;margin:1.15rem 0 .5rem;}

[data-testid="stFileUploaderDropzone"]{border-radius:12px !important;border-color:#d7e0ee !important;}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--blue) !important;}
div[data-baseweb="select"]>div{border-radius:10px !important;border-color:#d7e0ee !important;background:#fff;}
div[data-baseweb="textarea"]>div{border-radius:10px !important;border-color:#d7e0ee !important;background:#fff;}
[data-testid="stImage"] img{border-radius:12px;border:1px solid var(--line);}
div[data-testid="stButton"]>button{border-radius:10px;font-weight:700;height:2.7rem;
  background:var(--blue);border:none;letter-spacing:.02em;}
div[data-testid="stButton"]>button:hover{background:#1d4ed8;border:none;color:#fff;}
div[data-testid="stFormSubmitButton"]>button{border-radius:10px;font-weight:800;height:3rem;
  font-size:1.02rem;background:var(--blue);border:none;letter-spacing:.02em;width:100%;}
div[data-testid="stFormSubmitButton"]>button:hover{background:#1d4ed8;border:none;color:#fff;}
.stCaption{color:var(--muted);}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:12px;background:#fafbfd;}

.report-head{display:flex;justify-content:space-between;align-items:center;gap:1.2rem;flex-wrap:wrap;}
.pill{display:inline-block;padding:.5rem 1.25rem;border-radius:999px;font-weight:800;
  font-size:1.02rem;color:#fff;letter-spacing:.03em;box-shadow:0 2px 6px rgba(15,23,42,.15);}
.pill-green{background:var(--green);} .pill-amber{background:var(--amber);}
.pill-orange{background:var(--orange);} .pill-red{background:var(--red);}
.chips{margin-top:.75rem;display:flex;gap:.5rem;flex-wrap:wrap;}
.chip{background:#eef4ff;color:#334155;border:1px solid #dbe6fb;border-radius:8px;
  padding:.3rem .65rem;font-size:.8rem;font-weight:600;}

.ring{width:138px;height:138px;border-radius:50%;
  background:conic-gradient(var(--c) calc(var(--p) * 1%), #e8edf5 0);
  display:flex;align-items:center;justify-content:center;flex-shrink:0;
  box-shadow:inset 0 0 0 1px rgba(15,23,42,.04);}
.ring-inner{width:104px;height:104px;background:#fff;border-radius:50%;
  display:flex;flex-direction:column;align-items:center;justify-content:center;}
.ring-num{font-size:1.75rem;font-weight:800;color:var(--ink);line-height:1;}
.ring-sub{font-size:.68rem;color:var(--muted);font-weight:700;letter-spacing:.08em;margin-top:.25rem;}
.ring-cap{font-size:.62rem;color:var(--muted);font-weight:800;letter-spacing:.16em;
  text-align:center;margin-top:.5rem;}

.bars .row{margin-bottom:.7rem;}
.row-top{display:flex;justify-content:space-between;font-size:.84rem;color:#475569;
  margin-bottom:.3rem;font-weight:600;}
.row-top b{color:var(--ink);font-weight:800;}
.track{height:8px;background:#e8edf5;border-radius:99px;overflow:hidden;}
.fill{height:100%;border-radius:99px;transition:width .4s ease;}

.issue{display:flex;gap:.6rem;align-items:flex-start;background:#f8fafc;
  border:1px solid #edf1f7;border-radius:10px;padding:.6rem .85rem;
  font-size:.87rem;color:#334155;margin-bottom:.45rem;line-height:1.45;}
.issue span{flex-shrink:0;}

.explain{background:#f0f7ff;border:1px solid #d7e6fd;border-left:4px solid var(--blue);
  border-radius:10px;padding:.85rem 1.05rem;font-size:.92rem;color:#1e293b;line-height:1.6;}

.empty{border:1.5px dashed #cbd5e1;border-radius:14px;padding:2.6rem 1.2rem;
  text-align:center;color:var(--muted);background:#fbfcfe;font-size:.9rem;}
.empty b{color:#334155;display:block;margin-bottom:.35rem;font-size:1.02rem;}

.footer{margin-top:1.6rem;text-align:center;font-size:.76rem;color:#94a3b8;letter-spacing:.03em;}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
<div class="hero">
  <div class="eyebrow">PRODUCT IMAGE ANOMALY DETECTION</div>
  <h1>Product Image Anomaly Detection &amp; Quality Agent</h1>
  <p>Upload a picture of your product &rarr; the agent tells you whether it is
  the right product and whether the photo is good enough for a catalog: clear
  status, score out of 100, list of issues and a plain-English explanation.
  No technical knowledge needed &middot; analysis takes ~5 seconds (first run
  downloads the model, ~1 min).</p>
</div>
""",
    unsafe_allow_html=True,
)

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "examples" / "samples"
EXAMPLES = sorted(p.name for p in SAMPLES.glob("*.jpg"))
DEFAULT_EXAMPLE = "doliprane_ok.jpg" if "doliprane_ok.jpg" in EXAMPLES else (EXAMPLES[0] if EXAMPLES else None)
DEFAULT_JSON = (SAMPLES / "product_doliprane.json").read_text(encoding="utf-8") if (SAMPLES / "product_doliprane.json").exists() else "{}"

EXAMPLE_LABELS = {
    "doliprane_ok.jpg": "Correct product (Doliprane)",
    "doliprane_blurry.jpg": "Blurry photo",
    "maalox.jpg": "Wrong product (Maalox)",
    "doliprane_500.jpg": "Wrong dosage (500 mg)",
    "doliprane_badbrand.jpg": "Wrong brand",
    "shampoo.jpg": "Wrong category (shampoo)",
    "doliprane_composition.jpg": "Wrong composition text",
    "doliprane_lowres.jpg": "Low resolution",
    "doliprane_clutter.jpg": "Cluttered background",
}

try:
    PRODUCT_DEFAULTS: dict = json.loads(DEFAULT_JSON)
except json.JSONDecodeError:
    PRODUCT_DEFAULTS = {}

STATUS_STYLE = {
    "VALID": "green",
    "VALID_BUT_LOW_QUALITY": "amber",
    "LOW_IMAGE_QUALITY": "orange",
    "ANOMALY": "red",
    "WRONG_PRODUCT": "red",
    "WRONG_BRAND": "red",
    "WRONG_VARIANT": "red",
    "WRONG_CATEGORY": "red",
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


def _bar_color(value: float) -> str:
    if value >= 70:
        return "var(--green)"
    if value >= 50:
        return "var(--amber)"
    return "var(--red)"


def _esc(text: object) -> str:
    return html_lib.escape(str(text))


def render_report(res: dict) -> None:
    status = str(res.get("status", "?"))
    pill_cls = STATUS_STYLE.get(status, "orange")
    scores = res.get("scores", {})
    overall = float(scores.get("overall", 0.0))

    def _ring(value: float) -> str:
        color = "#16a34a" if value >= 70 else "#d97706" if value >= 50 else "#dc2626"
        return (
            f'<div><div class="ring" style="--p:{max(0.0, min(100.0, value)):.1f};--c:{color}">'
            f'<div class="ring-inner"><div class="ring-num">{value:.1f}</div>'
            f'<div class="ring-sub">/ 100</div></div></div>'
            f'<div class="ring-cap">OVERALL SCORE</div></div>'
        )

    st.markdown(
        f"""
<div class="report-head">
  <div>
    <div class="pill pill-{pill_cls}">{_esc(status)}</div>
    <div class="chips">
      <span class="chip">Anomaly: {"Yes" if res.get("is_anomaly") else "No"}</span>
      <span class="chip">Correct product: {"Yes" if res.get("is_correct_product") else "No"}</span>
      <span class="chip">Product ID: {_esc(res.get("product_id", "-"))}</span>
    </div>
  </div>
  {_ring(overall)}
</div>
""",
        unsafe_allow_html=True,
    )

    rows = []
    for key, label in SCORE_LABELS:
        val = float(scores.get(key, 0.0))
        val = max(0.0, min(100.0, val))
        rows.append(
            f'<div class="row"><div class="row-top"><span>{label}</span><b>{val:.1f}</b></div>'
            f'<div class="track"><div class="fill" style="width:{val:.1f}%;background:{_bar_color(val)}"></div></div></div>'
        )
    st.markdown('<div class="sect">Scores</div><div class="bars">' + "".join(rows) + "</div>", unsafe_allow_html=True)

    issues = res.get("issues") or []
    if issues:
        items = "".join(
            f'<div class="issue"><span>\u26a0\ufe0f</span><div>{_esc(item)}</div></div>' for item in issues
        )
        st.markdown(f'<div class="sect">Issues ({len(issues)})</div>{items}', unsafe_allow_html=True)

    if res.get("explanation"):
        st.markdown(
            '<div class="sect">Explanation</div>'
            f'<div class="explain">{_esc(res["explanation"])}</div>',
            unsafe_allow_html=True,
        )

    with st.expander("Detected product & raw JSON"):
        if res.get("detected_product"):
            st.json(res["detected_product"])
        st.json(res)


left, right = st.columns([5, 7], gap="large")

with left, st.container(border=True):
    st.markdown('<div class="panel-title">Step 1 &middot; Choose a picture</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload your own photo", type=["jpg", "jpeg", "png", "webp"])
    example = st.selectbox(
        "Or try one of these examples",
        EXAMPLES,
        index=EXAMPLES.index(DEFAULT_EXAMPLE) if DEFAULT_EXAMPLE else 0,
        format_func=lambda f: EXAMPLE_LABELS.get(f, f),
    )

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
            st.image(str(sample_path), caption=EXAMPLE_LABELS.get(example, example), width="stretch")

    st.markdown('<div class="panel-title">Step 2 &middot; Product details</div>', unsafe_allow_html=True)
    st.caption("Pre-filled for the examples - change these fields for your own photos.")

    with st.form("product_form"):
        p_name = st.text_input("Product name", str(PRODUCT_DEFAULTS.get("name", "")))
        p_brand = st.text_input("Brand", str(PRODUCT_DEFAULTS.get("brand", "")))
        c1, c2 = st.columns(2)
        with c1:
            p_cat = st.text_input("Category", str(PRODUCT_DEFAULTS.get("category", "")))
        with c2:
            p_dos = st.text_input("Dosage / strength", str(PRODUCT_DEFAULTS.get("dosage", "")))
        p_id = st.text_input("Product ID (optional)", str(PRODUCT_DEFAULTS.get("id", "")))
        submitted = st.form_submit_button("Analyze", use_container_width=True)

    with st.expander("Advanced: paste product JSON (optional)"):
        raw_json = st.text_area(
            "If this box is filled, it replaces the form above",
            value="",
            height=130,
            placeholder=DEFAULT_JSON.replace("\n", " "),
        )

with right, st.container(border=True):
    st.markdown('<div class="panel-title">Report</div>', unsafe_allow_html=True)

    if submitted:
        if input_image is None:
            st.warning("Choose a picture first (upload or example).")
        else:
            product = None
            if raw_json.strip():
                try:
                    loaded = json.loads(raw_json)
                    if isinstance(loaded, dict):
                        product = loaded
                    else:
                        st.error('Product JSON must be an object like {"name": "..."}.')
                except json.JSONDecodeError as exc:
                    st.error(f"Invalid product JSON: {exc}")
            else:
                product = {
                    key: value
                    for key, value in {
                        "id": p_id.strip(),
                        "name": p_name.strip(),
                        "brand": p_brand.strip(),
                        "category": p_cat.strip(),
                        "dosage": p_dos.strip(),
                    }.items()
                    if value
                }
                if not product:
                    product = None
                    st.warning("Fill at least one product field (or use the JSON box).")
            if isinstance(product, dict):
                with st.spinner("Analyzing image\u2026"):
                    try:
                        st.session_state["result"] = get_agent().analyze(input_image, product)
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Analysis failed: {exc}")

    result = st.session_state.get("result")
    if result is None:
        st.markdown(
            """
<div class="empty">
  <b>No report yet</b>
  Choose a picture on the left, check the product details, then press <b>Analyze</b>.
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        render_report(result)

st.markdown(
    '<div class="footer">Spec-compliant &middot; 43 tests passing &middot; '
    "OpenCV + CLIP + OCR &middot; Groq VLM validation notes</div>",
    unsafe_allow_html=True,
)

# Product Image Anomaly Detection & Quality Agent

AI agent jo ek **product image** + **product JSON** leta hai aur batata hai:

1. Kya image wohi product hai? (correctness)
2. Kya image mein visual anomalies hain? (quality / visibility / composition / background)
3. Kya image professional product catalog ke liye suitable hai?
4. 0-100 ka detailed quality score + wajuhaat (explanation)

> Spec: `Doc_agent.pdf` — "Product Image Anomaly Detection & Quality Agent"
> **Critical rule:** product correctness aur image quality **independently** evaluate hoti hain.
> **Technical rule:** yeh **LLM-only system NAHI hai** — bunyadi kaam Computer Vision
> (OpenCV + CLIP embeddings + OCR) karta hai. LLM/VLM sirf optional validation layer hai.

---

## Quick Start (Windows) — one double-click

1. Extract the ZIP.
2. Double-click **`Start App.bat`** — first time it installs everything
   (5–10 min, needs Python 3.10+ with "Add Python to PATH"); after that your
   browser opens the app automatically.
3. Click **Load example image** → **Analyze**.

No programming knowledge needed. Plain-English instructions: `START HERE.txt`.
Online version (no install): https://appuct-image-quality-agent-demo-dv7rtfhcmfszn5awblvyue.streamlit.app/

---

## Installation

Poora project + venv ek hi folder mein hai — folder delete = sab delete.

```powershell
# 1) venv (agar naya machine ho)
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip

# 2) torch (CPU) — alag index se
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3) baqi packages
.venv\Scripts\python -m pip install -r requirements.txt
```

Model weights (CLIP) bhi project ke andar hi download hoti hain:
`auto-agent/.modelcache/` (HF_HOME set rehta hai automatically).

---

## Usage

### CLI

```powershell
# samples banao (synthetic test images)
.venv\Scripts\python examples\make_samples.py --out examples\samples

# analyze
.venv\Scripts\python cli.py --image examples\samples\doliprane_ok.jpg --json examples\samples\product_doliprane.json
```

### API (FastAPI)

```powershell
.venv\Scripts\python cli.py serve --port 8000
```

```http
POST /analyze          (multipart: file + product_json)
POST /analyze          (JSON: {"image_base64": "...", "product": {...}})
POST /analyze          (JSON: {"image_url": "https://...", "product": {...}})
GET  /health
GET  /config
```

### Python

```python
from app.agent import ProductQualityAgent

agent = ProductQualityAgent()
report = agent.analyze("photo.jpg", {"id": "123", "name": "Doliprane 1000 mg Comprimé", "brand": "Sanofi", "category": "Medicament"})
print(report["status"], report["scores"]["overall"])
```

### Live demo (Streamlit)

```powershell
streamlit run streamlit_app.py     # local par demo UI
```

Cloud deploy (free — Streamlit Community Cloud):
1. GitHub repo push hona chahiye (`streamlit_app.py` root mein).
2. [share.streamlit.io](https://share.streamlit.io) → **Deploy a app** → repo / branch / `streamlit_app.py` → Deploy.
3. App settings → **Secrets** → `GROQ_API_KEY = "..."` (VLM notes ke liye; optional hai — bina key ke app chalta hai, VLM skip ho jata hai).

Notes: pehli analysis CLIP model download karti hai (~1 min, ~600 MB); 12h idle par app sleep ho jati hai (wake par dobara model load). Linux build ke liye `requirements.txt` mein CPU torch index aur `packages.txt` (libgl1) diya hua hai.

---

## JSON Output (spec section 7)

```json
{
  "product_id": "12345",
  "is_correct_product": true,
  "is_anomaly": false,
  "status": "VALID",
  "anomaly_types": [],
  "scores": {
    "product_match": 97, "image_quality": 91, "sharpness": 95,
    "resolution": 90, "visibility": 96, "composition": 88,
    "background": 92, "lighting": 89, "text_readability": 94, "overall": 92
  },
  "detected_product": {"name": "Doliprane 1000 mg", "brand": "Sanofi", "category": "Medicament"},
  "issues": [],
  "explanation": "The image corresponds to the provided product and has good visual quality."
}
```

---

## Status mapping (spec section 6 + examples ke mutabiq)

| Condition | Status |
|---|---|
| Category alag (Medicament vs Shampoo) | `WRONG_CATEGORY` |
| Doosra product (Doliprane vs Maalox) | `WRONG_PRODUCT` |
| Brand alag (Sanofi vs Novartis) | `WRONG_BRAND` |
| Dosage/form/packaging/quantity alag (1000mg vs 500mg, pack count) | `WRONG_VARIANT` |
| Sahi product + koi anomaly nahi + overall ≥ 70 | `VALID` |
| Sahi product + quality anomaly (BLUR, LOW_RESOLUTION...) — spec section 9: blurry ⇒ yehi | `VALID_BUT_LOW_QUALITY` |
| Sahi product + corrupted image, ya overall < 50 | `LOW_IMAGE_QUALITY` |
| Sahi product + sirf composition/background/visibility anomalies | `ANOMALY` |

Priority: `WRONG_*` > quality/anomaly > `VALID`. **Bina wajuha "bad" nahi kehte** —
har non-VALID status mein `issues[]` + `explanation` hota hai.

---

## Scoring (spec section 5)

`S_overall = Σ w_i × S_i` (weights normalize, total = 1.0)

| Score | Default weight | Kahan se aata hai |
|---|---|---|
| product_match | 0.25 | CLIP cosine similarity (image ↔ product text) + OCR token match |
| image_quality | 0.15 | sharpness/resolution/lighting/contrast/noise ka blend |
| sharpness | 0.10 | Laplacian variance |
| resolution | 0.08 | min side px (400–1000 scale) |
| visibility | 0.12 | product segmentation area + cut-off |
| composition | 0.10 | centering, margins, empty space, object count |
| background | 0.08 | Canny edge density product ke bahar |
| lighting | 0.07 | exposure mean (ideal 128) |
| text_readability | 0.05 | OCR confidence |

Weights `config.yaml → scoring.weights` mein **configurable** hain (auto-normalize).

---

## Checks (spec section 4)

| Section | Anomaly codes |
|---|---|
| 4.1 Quality | `BLUR`, `EXCESSIVE_BLUR`, `LOW_RESOLUTION`, `OVEREXPOSURE`, `UNDEREXPOSURE`, `POOR_CONTRAST`, `NOISE`, `PIXELATION`, `COMPRESSION_ARTIFACTS`, `DISTORTION`, `CORRUPTED_IMAGE` |
| 4.2 Visibility | `PRODUCT_NOT_VISIBLE`, `PRODUCT_TOO_SMALL`, `PRODUCT_CUT_OFF` |
| 4.3 Composition | `BAD_COMPOSITION`, `EXCESSIVE_EMPTY_SPACE`, `MULTIPLE_PRODUCTS` |
| 4.4 Background | `DISTRACTING_BACKGROUND`, `BACKGROUND_CLUTTER` |
| 4.5 Text | `TEXT_UNREADABLE` (+ JSON consistency correctness mein) |

---

## VLM validation layer (optional — client: Groq API)

Spec section 8 kehti hai LLM/VLM sirf **additional validation layer** hai — system
bina VLM ke bhi poori tarah kaam karta hai (OpenCV + CLIP + OCR se hi status decide
hota hai). VLM sirf extra notes deta hai.

```yaml
# config.yaml
vlm:
  enabled: true
  provider: groq                  # groq | gemini
  model: qwen/qwen3.8-27b         # Groq vision model
  timeout_sec: 10
  api_key_env: GROQ_API_KEY
```

```text
# .env file (root mein, git mein nahi jaati)
GROQ_API_KEY=gsk_...
```

- **Groq (default):** `qwen/qwen3.8-27b` — ~1-2s, client key `.env` mein hoti hai
- **Gemini:** `provider: gemini` + `GEMINI_API_KEY` (fallback option)
- VLM background thread mein chalta hai (max 10s budget) — analysis kabhi rukti nahi
- Fail/timeout ho to analysis phir bhi chalti hai (best-effort notes)
- VLM ke notes `issues[]` mein `VLM validation: ...` ke roop mein hote hain —
  **status/scores nahi badalte.**

---

## Tests (spec section 10 — sab categories farz hain)

```powershell
.venv\Scripts\python -m pytest tests -v
```

| Test file | Category |
|---|---|
| `test_correct_product.py` | correct products ✅ |
| `test_wrong_product.py` | wrong products |
| `test_wrong_variant.py` | wrong variants (dosage) |
| `test_wrong_category.py` | wrong categories |
| `test_wrong_brand.py` | wrong brands |
| `test_blurry.py` | blurry images (spec 9: → VALID_BUT_LOW_QUALITY) |
| `test_composition.py` | poor composition + cluttered background |
| `test_lowres.py` | low-resolution images |
| `test_api.py` | FastAPI endpoints (file / base64 / URL inputs) |
| `test_inputs_and_scoring.py` | input validation + scoring rules |
| `test_edge_cases.py` | URL input, minimal JSON, corrupted/tiny files |
| `test_vlm.py` | VLM layer (Groq) — notes, status unchanged |
| `test_ui.py` | Web UI + sample files |

Test images khud generate hoti hain (PIL synthetic) — client material pe depend nahi.

---

## Spec compliance checklist (Doc_agent.pdf → implementation)

| Spec section | Requirement | Status |
|---|---|---|
| 1 | Image + product JSON → correctness / anomalies / suitability / score | ✅ |
| 2.1 | Image input: file / URL / base64 | ✅ agent + API (multipart, `image_base64`, `image_url`) |
| 2.2 | Partial product JSON chale | ✅ (missing fields skip hote hain) |
| 3.1–3.4 | WRONG_PRODUCT / WRONG_BRAND / WRONG_VARIANT / WRONG_CATEGORY, including quantity/packaging evidence | ✅ tests |
| 4.1 | Quality anomalies (blur, low res, noise, exposure, contrast, distortion, corruption...) | ✅ `quality.py` |
| 4.2 | Visibility anomalies (too small, cut off, not visible) | ✅ `visibility.py` |
| 4.3 | Composition (edge, empty space, framing, multiple products) | ✅ `composition.py` |
| 4.4 | Background (distracting, clutter, unwanted objects) | ✅ `background.py` |
| 4.5 | Text readability + metadata consistency | ✅ `text_ocr.py` + `correctness.py` |
| 5 | 9 scores 0–100 + weighted overall, documented + configurable | ✅ README weights table, `config.yaml` |
| 6 | Statuses + har non-VALID ko wajuha | ✅ `classify.py`, `report.py` |
| 7 | Exact JSON schema (sab 3 examples jaisa) | ✅ `report.py` + schema tests |
| 8 | LLM-only NAHI; CV bunyad; VLM sirf validation layer | ✅ OpenCV + CLIP + OCR; Groq notes-only |
| 9 | Correctness aur quality independent (blur ⇒ VBLQ, wrong ⇒ WRONG_*) | ✅ `test_blurry`, `test_wrong_*` |
| 10 | Source, deps, README, API/UI, examples in/out, sab required tests | ✅ 43 tests |
| 11 | "Is this the correct product?" + "Is this a good image?" dono alag | ✅ |

---

## Project structure

```
auto-agent/
├── app/
│   ├── agent.py            # pipeline orchestrator
│   ├── main.py             # FastAPI
│   ├── input_loader.py     # file / URL / base64 + product JSON
│   ├── checks/
│   │   ├── quality.py      # 4.1 quality anomalies (OpenCV)
│   │   ├── visibility.py   # 4.2 product segmentation + visibility
│   │   ├── composition.py  # 4.3 composition
│   │   ├── background.py   # 4.4 background
│   │   ├── text_ocr.py     # 4.5 OCR (rapidocr)
│   │   ├── correctness.py  # 3 correctness (CLIP + OCR)
│   │   └── vlm.py          # optional Groq validation layer
│   ├── scoring.py          # 5 scores + weighted overall
│   ├── classify.py         # 6 final status
│   └── report.py           # 7 JSON output schema
├── examples/               # sample inputs + generator
├── tests/                  # required test categories
├── config.yaml             # weights + thresholds (sab configurable)
├── cli.py
├── requirements.txt
└── .venv/                  # packages (folder delete = sab delete)
└── .modelcache/            # CLIP weights (folder delete = sab delete)
```

---

## Client machine notes (testing: "my machine")

1. Python 3.10+ chahiye, internet sirf **pehli baar** (packages + CLIP download) ke liye —
   us ke baad offline chalta hai.
2. `pip install torch --index-url https://download.pytorch.org/whl/cpu` (GPU ki zaroorat nahi).
3. VLM layer chahiye to `GROQ_API_KEY` set karein, warna `vlm.enabled: false` rakhein.
4. Sab kuch is ek folder mein hai — delete karna ho to folder hata dein, kuch aur nahi.

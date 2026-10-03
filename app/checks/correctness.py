"""Section 3 - Product-image correctness.

Order (spec ke examples ke mutabiq):
  1. WRONG_CATEGORY  - image ki category hi alag hai (Shampoo vs Medicament)
  2. WRONG_PRODUCT   - doosra product hai (Doliprane vs Maalox)
  3. WRONG_BRAND     - wahi product, ghalat brand
  4. WRONG_VARIANT   - wahi product+brand, ghalat dosage/form/packaging

Evidence: OCR text matching (bunyadi) + CLIP image embeddings (jab available).
LLM/VLM yahan sirf optional validation layer hai (config.vlm).
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from functools import lru_cache

import numpy as np

from .text_ocr import normalize_token, ocr_text, ocr_tokens

# ---------------------------------------------------------------- category DB

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "medicament": [
        "comprime", "comprimes", "medicament", "medicaments", "gelule",
        "sirop", "capsule", "antidouleur", "antipiretique", "boite",
        "mg", "posologie", "gel", "pommade", "spray", "pipette",
    ],
    "shampoo": ["shampoo", "shampooing", "shampoing", "after shampoo"],
    "cosmetic": ["creme", "cosmetique", "cosmetic", "lotion", "serum", "savon"],
    "food": ["aliment", "food", "biscuit", "cafe", "the", "chocolat", "jus"],
    "electronics": ["batterie", "chargeur", "cable", "earbuds", "casque"],
}

# Expected category string -> dictionary key
CATEGORY_ALIASES = {
    "medicament": "medicament",
    "medicaments": "medicament",
    "medicine": "medicament",
    "drug": "medicament",
    "shampoo": "shampoo",
    "shampooing": "shampoo",
    "shampoing": "shampoo",
    "cosmetic": "cosmetic",
    "cosmetique": "cosmetic",
    "food": "food",
    "aliment": "food",
    "electronics": "electronics",
    "electronique": "electronics",
}

NAME_STOPWORDS = {
    "mg", "ml", "g", "comprime", "comprimes", "gelule", "capsule", "sirop",
    "boite", "de", "les", "la", "le", "un", "une", "et", "pour", "with",
    "the", "a", "an", "of", "x", "c", "num", "pack",
}


# ---------------------------------------------------------------- CLIP backend

@lru_cache(maxsize=2)
def _load_clip(model_name: str):
    from transformers import CLIPModel, CLIPProcessor
    import torch

    model = CLIPModel.from_pretrained(model_name)
    model.eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor, torch


class ClipBackend:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._models = None
        self.available = False
        self.error: str | None = None
        try:
            self._models = _load_clip(model_name)
            self.available = True
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)

    def similarity(self, img_bgr: np.ndarray, text: str) -> float | None:
        """Cosine similarity image <-> text (0..1 ke qareeb)."""
        if not self.available:
            return None
        model, processor, torch = self._models
        try:
            from PIL import Image

            rgb = img_bgr[:, :, ::-1]
            image = Image.fromarray(np.ascontiguousarray(rgb))
            inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
            with torch.no_grad():
                out = model(**inputs)
            image_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            text_emb = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            return float((image_emb @ text_emb.T).item())
        except Exception:  # noqa: BLE001
            return None

    def classify_category(self, img_bgr: np.ndarray, categories: list[str]) -> str | None:
        if not self.available or not categories:
            return None
        model, processor, torch = self._models
        try:
            from PIL import Image

            rgb = img_bgr[:, :, ::-1]
            image = Image.fromarray(np.ascontiguousarray(rgb))
            prompts = [f"a product photo of {c}" for c in categories]
            inputs = processor(text=prompts, images=[image], return_tensors="pt", padding=True)
            with torch.no_grad():
                out = model(**inputs)
            image_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            text_emb = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            probs = (image_emb @ text_emb.T).softmax(dim=-1).squeeze(0)
            idx = int(probs.argmax())
            if float(probs[idx]) < 0.16:
                return None
            return categories[idx]
        except Exception:  # noqa: BLE001
            return None


# ---------------------------------------------------------------- helpers

def detect_category_from_text(tokens: list[str]) -> str | None:
    best_key, best_hits = None, 0
    for key, words in CATEGORY_KEYWORDS.items():
        hits = sum(1 for w in words if w in tokens)
        if hits > best_hits:
            best_key, best_hits = key, hits
    return best_key


def expected_category_key(category: str | None) -> str | None:
    if not category:
        return None
    return CATEGORY_ALIASES.get(normalize_token(str(category)))


def _significant_name_tokens(name: str) -> list[str]:
    """Name ke meaningful tokens (numbers aur form words alag)."""
    toks = []
    for raw in re.split(r"[^\w]+", str(name or "")):
        tok = normalize_token(raw)
        if not tok or tok in NAME_STOPWORDS:
            continue
        toks.append(tok)
    return toks


def _expected_strengths(product: dict) -> set[str]:
    """JSON se strength numbers (dosage/name se): '1000', '500'..."""
    strengths: set[str] = set()
    for field in ("dosage", "strength", "name"):
        value = str(product.get(field) or "")
        for num in re.findall(r"\d+", value):
            if int(num) >= 10:
                strengths.add(num)
    return strengths


def _clip_score_to_points(sim: float | None) -> float | None:
    if sim is None:
        return None
    return float(np.clip((sim - 0.10) / (0.45 - 0.10) * 100.0, 0.0, 100.0))


def _fuzzy_in(word: str, tokens: set[str], cutoff: float = 0.8) -> bool:
    """OCR typos/partial reads ke liye fuzzy match (e.g. 'OLIPRAN' ~ 'doliprane')."""
    if not word:
        return False
    if word in tokens:
        return True
    for tok in tokens:
        if len(tok) < 4:
            continue
        if SequenceMatcher(None, word, tok).ratio() >= cutoff:
            return True
    return False


def _foreign_brand_candidates(ocr_results: list[dict], exclude: set[str], name_tokens: list[str]) -> list[str]:
    """OCR se aise bade uppercase/shuru mein capital words jo expected product
    ke na hon - yehi WRONG_BRAND ka solid evidence hai (spec 3.2)."""
    candidates: list[str] = []
    name_norm = set(name_tokens) | set(exclude)
    stop = NAME_STOPWORDS | {normalize_token(w) for w in CATEGORY_KEYWORDS_ALL}
    for entry in ocr_results:
        for raw in re.split(r"[^\w']+", entry["text"]):
            raw = raw.strip()
            if len(raw) < 4 or not raw[0].isupper() and not raw.isupper():
                continue
            tok = normalize_token(raw)
            if not tok or len(tok) < 4 or tok in stop or tok.isdigit():
                continue
            if any(_fuzzy_in(tok, {n}, cutoff=0.75) for n in name_norm):
                continue
            if tok not in candidates:
                candidates.append(tok)
    return candidates


CATEGORY_KEYWORDS_ALL = [w for words in CATEGORY_KEYWORDS.values() for w in words]


# ---------------------------------------------------------------- main check

def evaluate_correctness(
    img: np.ndarray,
    product: dict,
    ocr_results: list[dict],
    clip: ClipBackend | None,
    th: dict,
    *,
    allow_visual_mismatch: bool = True,
) -> dict:
    issues: list[str] = []
    tokens = ocr_tokens(ocr_results)
    text = ocr_text(ocr_results).lower()
    tokens_set = set(tokens)

    expected_name = str(product.get("name") or "")
    expected_brand = str(product.get("brand") or product.get("laboratory") or "")
    expected_category = str(product.get("category") or "")

    name_tokens = _significant_name_tokens(expected_name)
    strengths = _expected_strengths(product)

    # --- category detection ---
    # CLIP category classification is intentionally not used as primary
    # evidence. Generic/degraded product photos are often mapped to an
    # unrelated category (for example, a medicine box -> electronics). OCR
    # provides stronger evidence when packaging text is readable; when it is
    # not, the safe result is uncertainty rather than WRONG_CATEGORY.
    detected_category = detect_category_from_text(tokens)
    expected_key = expected_category_key(expected_category)

    wrong_status: str | None = None

    # 1) category mismatch (sirf jab dono confidently known hon)
    if detected_category and expected_key and detected_category != expected_key:
        wrong_status = "WRONG_CATEGORY"
        issues.append(
            f"Expected category '{expected_category}' but the image appears to be '{detected_category}'."
        )

    # 2) product name matching (fuzzy - OCR partial reads ke liye)
    core_token = name_tokens[0] if name_tokens else ""
    name_found = bool(core_token) and _fuzzy_in(core_token, tokens_set)
    any_ocr = bool(tokens)

    clip_sim = clip.similarity(img, expected_name or expected_brand or "product") if (clip and clip.available) else None
    clip_points = _clip_score_to_points(clip_sim)

    if wrong_status is None and any_ocr and core_token and not name_found:
        wrong_status = "WRONG_PRODUCT"
        detected_guess = _guess_detected_name(tokens, expected_brand, expected_category)
        issues.append(
            f"The image does not correspond to the provided product "
            f"(expected '{expected_name or core_token}', image shows '{detected_guess or 'a different product'}')."
        )
    elif wrong_status is None and allow_visual_mismatch and not any_ocr and clip_sim is not None:
        if clip_sim < th["clip_min_similarity"]:
            wrong_status = "WRONG_PRODUCT"
            issues.append("Visual similarity to the expected product is too low.")

    # 3) brand mismatch: name match + readable text + brand missing +
    #    OCR mein koi doosra (foreign) brand word bhi hona chahiye -
    #    sirf "Sanofi" na parhna shahadat nahi (blur/size ki wajah se miss ho sakta hai)
    brand_token = normalize_token(expected_brand.split()[0]) if expected_brand else ""
    brand_found = bool(brand_token) and (brand_token in tokens_set or any(brand_token in t for t in tokens_set))
    foreign = _foreign_brand_candidates(
        ocr_results,
        exclude=({brand_token} if brand_token else set()) | set(name_tokens),
        name_tokens=name_tokens,
    )
    if (
        wrong_status is None
        and any_ocr
        and name_found
        and brand_token
        and not brand_found
        and foreign
    ):
        wrong_status = "WRONG_BRAND"
        issues.append(
            f"Expected brand '{expected_brand}' but the image shows '{foreign[0].title()}'."
        )

    # 4) variant: dosage / strength / form
    if wrong_status is None and name_found and any_ocr:
        ocr_numbers = set(re.findall(r"\d+", text))
        if strengths and not (strengths & ocr_numbers) and ocr_numbers:
            wrong_status = "WRONG_VARIANT"
            expected_str = "/".join(sorted(strengths))
            issues.append(
                f"Expected dosage/strength ({expected_str}) not found on the image "
                f"(image shows: {', '.join(sorted(ocr_numbers))})."
            )
        else:
            form = normalize_token(str(product.get("form") or ""))
            form_aliases = {
                "comprime": {"comprime", "comprimes", "tablette", "tablet"},
                "gelule": {"gelule", "capsule"},
                "sirop": {"sirop", "syrup"},
                "creme": {"creme", "cream", "pommade"},
            }
            if form and form in form_aliases:
                allowed = form_aliases[form]
                other_forms = {"comprime", "comprimes", "gelule", "sirop", "creme", "capsule", "tablette"}
                found_form = tokens_set & other_forms
                if found_form and not (found_form & allowed):
                    wrong_status = "WRONG_VARIANT"
                    issues.append(
                        f"Expected pharmaceutical form '{product.get('form')}' but the image shows '{sorted(found_form)[0]}'."
                    )

    # --- product_match score ---
    ocr_match_ratio = 0.0
    if name_tokens:
        hits = sum(1 for t in name_tokens if _fuzzy_in(t, tokens_set))
        ocr_match_ratio = hits / len(name_tokens)

    if clip_points is not None:
        product_match = 0.45 * clip_points + 0.55 * ocr_match_ratio * 100.0
    elif any_ocr:
        product_match = ocr_match_ratio * 100.0
    else:
        product_match = 50.0  # uncertain - benefit of doubt

    if wrong_status == "WRONG_PRODUCT":
        product_match = min(product_match, 20.0)
    elif wrong_status is not None:
        product_match = min(product_match, 55.0)

    if not any_ocr and clip is None:
        issues.append("Correctness verification is limited (no OCR text and no CLIP model available).")

    # --- detected product ---
    detected_name = _guess_detected_name(tokens, expected_brand, expected_category)
    if wrong_status is None:
        detected_name = expected_name or detected_name or "Unknown"
    detected_brand = expected_brand if (brand_found and wrong_status != "WRONG_BRAND") else ("Unknown" if wrong_status == "WRONG_BRAND" else (expected_brand if brand_found else "Unknown"))
    detected_cat = detected_category or (expected_category if wrong_status is None else "Unknown")
    detected_cat_title = detected_cat[:1].upper() + detected_cat[1:] if detected_cat else "Unknown"

    return {
        "is_correct": wrong_status is None,
        "wrong_status": wrong_status,
        "product_match": round(float(np.clip(product_match, 0.0, 100.0)), 1),
        "clip_similarity": round(clip_sim, 4) if clip_sim is not None else None,
        "detected_product": {
            "name": detected_name or "Unknown",
            "brand": detected_brand or "Unknown",
            "category": detected_cat_title or "Unknown",
        },
        "issues": issues,
        "ocr_text": text,
    }


def _guess_detected_name(tokens: list[str], expected_brand: str, expected_category: str) -> str:
    """OCR tokens se jo product nazar aa raha hai uska naam (spec example:
    detected_product.name = 'Maalox')."""
    brand_toks = {normalize_token(t) for t in expected_brand.split()} if expected_brand else set()
    cat_words = set()
    for words in CATEGORY_KEYWORDS.values():
        cat_words.update(words)
    skip = NAME_STOPWORDS | brand_toks | cat_words
    words = [t for t in tokens if t not in skip and not t.isdigit() and len(t) >= 3]
    if not words:
        return ""
    title = " ".join(w[:1].upper() + w[1:] for w in words[:3])
    return title

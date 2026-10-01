"""Synthetic sample inputs banata hai (Spec Section 10: example inputs +
test cases). PIL se draw hoti hain - koi internet ya client material nahi chahiye.

Istemal:
  python examples/make_samples.py --out examples/samples
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIRS = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
]


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_DIRS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _draw_centered(draw: ImageDraw.ImageDraw, xy_center: tuple[int, int], text: str, font, fill: str) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    w, h = right - left, bottom - top
    x = xy_center[0] - w // 2
    y = xy_center[1] - h // 2
    draw.text((x, y), text, font=font, fill=fill)


def draw_product_box(
    canvas: tuple[int, int] = (900, 900),
    box_size: tuple[int, int] = (520, 640),
    position: tuple[int, int] | None = None,
    box_color: str = "#EAF3FC",
    border_color: str = "#155A96",
    title: str = "DOLIPRANE",
    subtitle: str = "1000 mg",
    line3: str = "Comprimés",
    brand: str = "Sanofi",
) -> Image.Image:
    """Ek product box wali synthetic tasveer - white background."""
    img = Image.new("RGB", canvas, "#FFFFFF")
    draw = ImageDraw.Draw(img)

    if position is None:
        x = (canvas[0] - box_size[0]) // 2
        y = (canvas[1] - box_size[1]) // 2
    else:
        x, y = position

    draw.rectangle([x, y, x + box_size[0], y + box_size[1]], fill=box_color, outline=border_color, width=6)
    # header strip
    draw.rectangle([x, y, x + box_size[0], y + 90], fill=border_color)

    cx = x + box_size[0] // 2
    _draw_centered(draw, (cx, y + 45), title, get_font(44), "#FFFFFF")
    _draw_centered(draw, (cx, y + 160), subtitle, get_font(56), "#0E2A3F")
    _draw_centered(draw, (cx, y + 250), line3, get_font(34), "#20303F")
    _draw_centered(draw, (cx, y + box_size[1] - 60), brand, get_font(30), border_color)

    # thora sa design (packaging stripes)
    draw.rectangle([x + 30, y + 320, x + box_size[0] - 30, y + 330], fill=border_color)
    draw.rectangle([x + 30, y + 350, x + box_size[0] - 60, y + 358], fill="#7FB2DF")
    return img


def draw_bottle(canvas: tuple[int, int] = (900, 900)) -> Image.Image:
    """Shampoo bottle (wrong category test)."""
    img = Image.new("RGB", canvas, "#FFFFFF")
    draw = ImageDraw.Draw(img)
    x, y, w, h = 330, 150, 240, 620
    draw.rounded_rectangle([x, y, x + w, y + h], radius=60, fill="#F6D9EE", outline="#B03A8A", width=6)
    draw.rectangle([x + 60, y - 60, x + w - 60, y + 10], fill="#B03A8A")
    _draw_centered(draw, (x + w // 2, y + 220), "SHAMPOOING", get_font(34), "#6E1E56")
    _draw_centered(draw, (x + w // 2, y + 300), "PROVANT", get_font(44), "#6E1E56")
    _draw_centered(draw, (x + w // 2, y + 420), "400 ml", get_font(30), "#6E1E56")
    return img


def add_clutter(img: Image.Image, seed: int = 7) -> Image.Image:
    """Background mein distractions (wrong objects / clutter test).
    Product (center box) ko chhootha jata hai - sirf background distract hota hai."""
    rnd = random.Random(seed)
    draw = ImageDraw.Draw(img)
    w, h = img.size
    colors = ["#E74C3C", "#F39C12", "#27AE60", "#8E44AD", "#2C3E50", "#16A085"]
    # product box = center (190..710, 130..770) - us ke bahar hi shapes
    zones = [  # (x_min, x_max, y_min, y_max)
        (5, 180, 5, h - 5),        # left strip
        (720, w - 5, 5, h - 5),    # right strip
        (185, 715, 5, 120),        # top strip
        (185, 715, 780, h - 5),    # bottom strip
    ]
    for _ in range(14):
        zx0, zx1, zy0, zy1 = rnd.choice(zones)
        cw = rnd.randint(40, min(110, max(45, zx1 - zx0 - 5)))
        ch = rnd.randint(40, min(110, max(45, zy1 - zy0 - 5)))
        if zx1 - zx0 - cw < 3 or zy1 - zy0 - ch < 3:
            continue
        cx = rnd.randint(zx0, zx1 - cw)
        cy = rnd.randint(zy0, zy1 - ch)
        color = rnd.choice(colors)
        if rnd.random() < 0.5:
            draw.rectangle([cx, cy, cx + cw, cy + ch], fill=color)
        else:
            draw.ellipse([cx, cy, cx + cw, cy + ch], fill=color)
    # thora sa texture noise bhi background par
    for _ in range(300):
        x = rnd.randint(0, w - 1)
        y = rnd.randint(0, h - 1)
        if 185 <= x <= 715 and 120 <= y <= 780:
            continue
        draw.point((x, y), fill=rnd.choice(colors))
    return img


PRODUCT_JSON = {
    "id": "12345",
    "name": "Doliprane 1000 mg Comprimé",
    "category": "Medicament",
    "brand": "Sanofi",
    "laboratory": "Sanofi",
    "dosage": "1000 mg",
    "form": "Comprimé",
    "packaging": "Boîte de 8 comprimés",
}


def build_all(out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    def save(img: Image.Image, name: str) -> Path:
        path = out / name
        img.save(path, quality=95)
        paths[name] = path
        return path

    # 1) correct product - clean
    save(draw_product_box(), "doliprane_ok.jpg")

    # 2) blurry (spec test: blurry images)
    blur = draw_product_box().filter(ImageFilter.GaussianBlur(4.0))
    save(blur, "doliprane_blurry.jpg")

    # 3) low resolution (spec test: low-resolution images)
    lowres = draw_product_box().resize((260, 260), Image.LANCZOS)
    save(lowres, "doliprane_lowres.jpg")

    # 4) poor composition (product corner mein, bahut khali jagah)
    comp = draw_product_box(box_size=(190, 240), position=(25, 30))
    save(comp, "doliprane_composition.jpg")

    # 5) wrong product (Maalox instead of Doliprane)
    maalox = draw_product_box(
        box_color="#FDF0E3",
        border_color="#C06014",
        title="MAALOX",
        subtitle="100 mg",
        line3="Comprimés",
        brand="Reckitt",
    )
    save(maalox, "maalox.jpg")

    # 6) wrong variant (500 mg instead of 1000 mg)
    v500 = draw_product_box(subtitle="500 mg")
    save(v500, "doliprane_500.jpg")

    # 7) wrong category (shampoo instead of medicament)
    save(draw_bottle(), "shampoo.jpg")

    # 8) wrong brand (Novartis instead of Sanofi)
    badbrand = draw_product_box(brand="Novartis")
    save(badbrand, "doliprane_badbrand.jpg")

    # 9) cluttered background
    save(add_clutter(draw_product_box()), "doliprane_clutter.jpg")

    # product JSON (example input)
    json_path = out / "product_doliprane.json"
    json_path.write_text(json.dumps(PRODUCT_JSON, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["product_doliprane.json"] = json_path

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic sample inputs")
    parser.add_argument("--out", default=str(Path(__file__).parent / "samples"))
    args = parser.parse_args()
    paths = build_all(args.out)
    for name, path in paths.items():
        print(f"  {name}  ->  {path}")


if __name__ == "__main__":
    main()

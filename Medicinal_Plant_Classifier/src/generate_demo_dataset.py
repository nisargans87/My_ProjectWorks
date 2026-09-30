"""
generate_demo_dataset.py
-------------------------
Creates a small SYNTHETIC leaf-image dataset so the entire project
(EDA -> training -> Streamlit app) can be run end-to-end immediately,
without waiting to download a real dataset.

Each "species" gets a distinct leaf shape, base color, vein pattern and
texture noise so a CNN has real (if easy) signal to learn from.

IMPORTANT: This is a placeholder dataset for demonstration and testing
only. For a real project, replace `data/raw/<class_name>/*.jpg` with an
actual photographed dataset such as the "Indian Medicinal Leaf Image
Dataset" (Mendeley Data) or a Kaggle medicinal-leaf dataset, keeping the
same folder-per-class structure. Everything downstream (analysis,
training, app) works unchanged once real images are dropped in.

Usage:
    python src/generate_demo_dataset.py --per_class 60 --img_size 224
"""

import argparse
import os
import random
import math
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

# 8 common Indian medicinal plants used across placement / academic
# mini-projects. Feel free to rename / extend -- the rest of the code
# reads class names directly from the folder structure.
CLASSES = {
    "Tulsi_Basil":     {"color": (46, 125, 50),  "shape": "oval",    "veins": "pinnate"},
    "Neem":            {"color": (76, 175, 80),  "shape": "serrated","veins": "pinnate"},
    "Aloe_Vera":       {"color": (129, 199, 132),"shape": "spiky",   "veins": "parallel"},
    "Betel":           {"color": (27, 94, 32),   "shape": "heart",   "veins": "palmate"},
    "Mint":            {"color": (102, 187, 106),"shape": "round",   "veins": "reticulate"},
    "Curry_Leaf":      {"color": (56, 142, 60),  "shape": "lance",   "veins": "pinnate"},
    "Hibiscus":        {"color": (67, 160, 71),  "shape": "serrated","veins": "palmate"},
    "Guava":           {"color": (85, 139, 47),  "shape": "oval",    "veins": "pinnate"},
}


def _leaf_polygon(shape, w, h, rng):
    """Return a list of (x, y) points approximating a leaf outline."""
    cx, cy = w / 2, h / 2
    pts = []
    n = 60
    for i in range(n):
        theta = 2 * math.pi * i / n
        base_r = min(w, h) * 0.38

        if shape == "oval":
            r = base_r * (1.0 - 0.15 * math.cos(theta) ** 2)
        elif shape == "heart":
            r = base_r * (1.0 + 0.35 * math.sin(theta / 2))
            if math.pi * 0.8 < theta < math.pi * 1.2:
                r *= 0.6
        elif shape == "lance":
            r = base_r * (0.7 + 0.5 * abs(math.cos(theta)))
        elif shape == "round":
            r = base_r * (1.0 + 0.05 * math.sin(4 * theta))
        elif shape == "spiky":
            r = base_r * (0.75 + 0.35 * (i % 4 == 0))
        elif shape == "serrated":
            r = base_r * (0.95 + 0.08 * math.sin(10 * theta))
        else:
            r = base_r

        r *= rng.uniform(0.96, 1.04)
        x = cx + r * math.cos(theta) * 0.85
        y = cy + r * math.sin(theta)
        pts.append((x, y))
    return pts


def _draw_veins(draw, w, h, pattern, color, rng):
    cx, cy = w / 2, h / 2
    dark = tuple(max(0, c - 60) for c in color)
    if pattern == "pinnate":
        draw.line([(cx, h * 0.12), (cx, h * 0.88)], fill=dark, width=2)
        for t in np.linspace(0.15, 0.85, 7):
            y = h * t
            spread = (w * 0.32) * math.sin(t * math.pi)
            draw.line([(cx, y), (cx - spread, y - h * 0.05)], fill=dark, width=1)
            draw.line([(cx, y), (cx + spread, y - h * 0.05)], fill=dark, width=1)
    elif pattern == "palmate":
        for ang in range(-60, 61, 30):
            rad = math.radians(ang - 90)
            x2 = cx + math.cos(rad) * w * 0.35
            y2 = cy + math.sin(rad) * h * 0.35 + h * 0.1
            draw.line([(cx, cy + h * 0.15), (x2, y2)], fill=dark, width=1)
    elif pattern == "parallel":
        for x in np.linspace(w * 0.3, w * 0.7, 5):
            draw.line([(x, h * 0.1), (x, h * 0.9)], fill=dark, width=1)
    else:  # reticulate
        rng_local = random.Random(0)
        for _ in range(10):
            x1, y1 = rng_local.uniform(w * 0.3, w * 0.7), rng_local.uniform(h * 0.2, h * 0.8)
            x2, y2 = x1 + rng_local.uniform(-20, 20), y1 + rng_local.uniform(-20, 20)
            draw.line([(x1, y1), (x2, y2)], fill=dark, width=1)


def make_leaf_image(class_name, img_size, seed):
    rng = random.Random(seed)
    spec = CLASSES[class_name]
    w = h = img_size

    # Background: soft neutral surface (like a table / notebook backdrop)
    bg_shade = rng.randint(225, 245)
    img = Image.new("RGB", (w, h), (bg_shade, bg_shade - 5, bg_shade - 15))
    draw = ImageDraw.Draw(img)

    # Slight background texture
    for _ in range(30):
        x, y = rng.randint(0, w), rng.randint(0, h)
        r = rng.randint(1, 3)
        shade = bg_shade + rng.randint(-8, 8)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(shade, shade - 5, shade - 15))

    # Leaf color with per-sample jitter (simulates lighting / individual variation)
    base = spec["color"]
    jitter = lambda c: max(0, min(255, c + rng.randint(-18, 18)))
    color = tuple(jitter(c) for c in base)

    poly = _leaf_polygon(spec["shape"], w, h, rng)
    draw.polygon(poly, fill=color, outline=tuple(max(0, c - 40) for c in color))

    _draw_veins(draw, w, h, spec["veins"], color, rng)

    # Random rotation + slight blur to mimic real photo variability
    angle = rng.uniform(-25, 25)
    img = img.rotate(angle, fillcolor=(bg_shade, bg_shade - 5, bg_shade - 15))
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.8)))

    return img


def build_dataset(out_dir, per_class, img_size, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)
    total = 0
    for class_name in CLASSES:
        class_dir = os.path.join(out_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(per_class):
            img = make_leaf_image(class_name, img_size, seed=rng.randint(0, 10_000_000))
            img.save(os.path.join(class_dir, f"{class_name}_{i:03d}.jpg"), quality=90)
            total += 1
        print(f"  {class_name}: {per_class} images")
    print(f"Done. Wrote {total} synthetic images to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic demo leaf dataset.")
    parser.add_argument("--out_dir", default="data/raw", help="Output directory (class-per-folder)")
    parser.add_argument("--per_class", type=int, default=60, help="Images per class")
    parser.add_argument("--img_size", type=int, default=224, help="Square image size in pixels")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Generating synthetic dataset: {len(CLASSES)} classes x {args.per_class} images...")
    build_dataset(args.out_dir, args.per_class, args.img_size, args.seed)

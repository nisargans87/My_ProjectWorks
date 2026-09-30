"""
data_analysis.py
-----------------
Exploratory Data Analysis (EDA) for the medicinal plant leaf dataset.

Scans `data/raw/<class_name>/*.jpg`, builds a Pandas DataFrame describing
every image (class, resolution, file size, aspect ratio, avg RGB), then
produces Matplotlib figures summarizing the dataset:

    1. Class distribution bar chart
    2. Image file-size distribution
    3. Image width/height scatter (resolution consistency check)
    4. Average brightness per class (box plot)
    5. A sample-image grid, one row per class

All figures are written to `outputs/eda/`. The DataFrame itself is also
saved as `outputs/eda/dataset_manifest.csv` so it can be reused by other
scripts or inspected in Excel/Pandas directly.

Usage:
    python src/data_analysis.py --data_dir data/raw --out_dir outputs/eda
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

VALID_EXTS = {".jpg", ".jpeg", ".png"}


def build_manifest(data_dir: str) -> pd.DataFrame:
    """Walk the class-per-folder dataset and build a metadata DataFrame."""
    rows = []
    data_path = Path(data_dir)
    class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])

    if not class_dirs:
        raise FileNotFoundError(
            f"No class sub-folders found under '{data_dir}'. Expected "
            f"structure: {data_dir}/<class_name>/<image files>"
        )

    for class_dir in class_dirs:
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() not in VALID_EXTS:
                continue
            try:
                with Image.open(f) as im:
                    w, h = im.size
                    arr = np.asarray(im.convert("RGB").resize((32, 32)), dtype=np.float32)
                    mean_r, mean_g, mean_b = arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()
                rows.append({
                    "filepath": str(f),
                    "class": class_dir.name,
                    "width": w,
                    "height": h,
                    "aspect_ratio": round(w / h, 3),
                    "filesize_kb": round(f.stat().st_size / 1024, 1),
                    "mean_r": round(mean_r, 1),
                    "mean_g": round(mean_g, 1),
                    "mean_b": round(mean_b, 1),
                    "brightness": round((mean_r + mean_g + mean_b) / 3, 1),
                })
            except Exception as e:
                print(f"  [skip] {f}: {e}")

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame):
    print("\n=== Dataset Summary ===")
    print(f"Total images : {len(df)}")
    print(f"Classes ({df['class'].nunique()}): {', '.join(sorted(df['class'].unique()))}")
    print("\nImages per class:")
    print(df["class"].value_counts().to_string())
    print("\nResolution stats:")
    print(df[["width", "height", "filesize_kb"]].describe().round(1).to_string())

    counts = df["class"].value_counts()
    imbalance_ratio = counts.max() / counts.min()
    print(f"\nClass imbalance ratio (max/min count): {imbalance_ratio:.2f}")
    if imbalance_ratio > 1.5:
        print("  -> Noticeable imbalance: consider class_weight or augmentation for minority classes.")
    else:
        print("  -> Classes are reasonably balanced.")


def plot_class_distribution(df, out_dir):
    counts = df["class"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(counts.index, counts.values, color="#4c8c4a")
    ax.set_xlabel("Number of images")
    ax.set_title("Images per Medicinal Plant Class")
    for bar, val in zip(bars, counts.values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2, str(val), va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "class_distribution.png"), dpi=140)
    plt.close(fig)


def plot_filesize_distribution(df, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(df["filesize_kb"], bins=25, color="#6aa96a", edgecolor="white")
    ax.set_xlabel("File size (KB)")
    ax.set_ylabel("Count")
    ax.set_title("Image File Size Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "filesize_distribution.png"), dpi=140)
    plt.close(fig)


def plot_resolution_scatter(df, out_dir):
    fig, ax = plt.subplots(figsize=(6, 6))
    for cls, sub in df.groupby("class"):
        ax.scatter(sub["width"], sub["height"], label=cls, alpha=0.6, s=18)
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("Image Resolution by Class")
    ax.legend(fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "resolution_scatter.png"), dpi=140)
    plt.close(fig)


def plot_brightness_boxplot(df, out_dir):
    classes = sorted(df["class"].unique())
    data = [df.loc[df["class"] == c, "brightness"] for c in classes]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(data, tick_labels=classes, showmeans=True)
    ax.set_ylabel("Mean brightness (0-255)")
    ax.set_title("Average Image Brightness per Class")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "brightness_boxplot.png"), dpi=140)
    plt.close(fig)


def plot_sample_grid(df, out_dir, per_class=4):
    classes = sorted(df["class"].unique())
    fig, axes = plt.subplots(len(classes), per_class, figsize=(per_class * 2.1, len(classes) * 2.1))
    for i, cls in enumerate(classes):
        sample_paths = df.loc[df["class"] == cls, "filepath"].sample(
            min(per_class, (df["class"] == cls).sum()), random_state=0
        )
        for j, p in enumerate(sample_paths):
            ax = axes[i, j] if len(classes) > 1 else axes[j]
            ax.imshow(Image.open(p))
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(cls, fontsize=9)
        for j in range(len(sample_paths), per_class):
            ax = axes[i, j] if len(classes) > 1 else axes[j]
            ax.axis("off")
    fig.suptitle("Sample Images per Class", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sample_grid.png"), dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="EDA for medicinal plant leaf dataset.")
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--out_dir", default="outputs/eda")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Scanning '{args.data_dir}' ...")
    df = build_manifest(args.data_dir)
    df.to_csv(os.path.join(args.out_dir, "dataset_manifest.csv"), index=False)

    print_summary(df)

    print("\nGenerating plots ...")
    plot_class_distribution(df, args.out_dir)
    plot_filesize_distribution(df, args.out_dir)
    plot_resolution_scatter(df, args.out_dir)
    plot_brightness_boxplot(df, args.out_dir)
    plot_sample_grid(df, args.out_dir)
    print(f"Saved manifest + 5 figures to '{args.out_dir}'")


if __name__ == "__main__":
    main()

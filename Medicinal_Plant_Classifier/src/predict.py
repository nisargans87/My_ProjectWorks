"""
predict.py
----------
Quick command-line inference on a single leaf image, using the model
trained by train.py.

Usage:
    python src/predict.py --image path/to/leaf.jpg
"""

import argparse
import json

import numpy as np
from PIL import Image
from tensorflow import keras


def main():
    parser = argparse.ArgumentParser(description="Predict a medicinal plant species from a leaf image.")
    parser.add_argument("--image", required=True, help="Path to a leaf photo (jpg/png)")
    parser.add_argument("--model_path", default="saved_model/leaf_classifier.keras")
    parser.add_argument("--class_info", default="saved_model/class_indices.json")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    with open(args.class_info) as f:
        meta = json.load(f)
    classes, img_size = meta["classes"], meta["img_size"]

    model = keras.models.load_model(args.model_path)

    img = Image.open(args.image).convert("RGB").resize((img_size, img_size))
    arr = np.array(img, dtype=np.float32)[None, ...]

    preds = model.predict(arr, verbose=0)[0]
    order = np.argsort(preds)[::-1][: args.top_k]

    print(f"\nPredictions for '{args.image}':")
    for rank, idx in enumerate(order, start=1):
        print(f"  {rank}. {classes[idx]:<15s} {preds[idx] * 100:5.2f}%")


if __name__ == "__main__":
    main()

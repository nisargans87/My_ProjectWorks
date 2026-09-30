"""
train.py
--------
Trains the medicinal plant leaf classifier end-to-end:

    1. Loads image paths from `data/raw/<class>/*.jpg`
    2. Stratified 70/15/15 train/val/test split
    3. Builds a tf.data pipeline with on-the-fly augmentation for training
    4. Trains a MobileNetV2 transfer-learning model (see model.py)
    5. Plots accuracy/loss curves
    6. Evaluates on the held-out test set: classification report + confusion matrix
    7. Saves the trained model + class index mapping to `saved_model/`

Usage:
    python src/train.py --data_dir data/raw --epochs 15 --batch_size 16
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

from model import build_transfer_model, compile_model

AUTOTUNE = tf.data.AUTOTUNE


def list_images(data_dir):
    classes = sorted(
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    )
    class_to_idx = {c: i for i, c in enumerate(classes)}
    paths, labels = [], []
    for c in classes:
        for f in sorted(os.listdir(os.path.join(data_dir, c))):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(data_dir, c, f))
                labels.append(class_to_idx[c])
    return np.array(paths), np.array(labels), classes, class_to_idx


def make_dataset(paths, labels, img_size, batch_size, num_classes, augment=False, shuffle=False):
    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, [img_size, img_size])
        label = tf.one_hot(label, num_classes)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=42)
    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)

    if augment:
        augmenter = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.15),
            layers.RandomTranslation(0.1, 0.1),
        ])
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def plot_history(history, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=140)
    plt.close(fig)


def plot_confusion_matrix(cm, classes, out_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Test Set)")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train the medicinal plant leaf classifier.")
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--out_dir", default="outputs/training")
    parser.add_argument("--model_dir", default="saved_model")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fine_tune_epochs", type=int, default=5,
                         help="Extra epochs with top layers of the base model unfrozen.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pretrained_weights", default="imagenet",
                         help="'imagenet' (default, needs internet) or 'none' for offline testing.")
    args = parser.parse_args()
    weights_arg = None if args.pretrained_weights.lower() == "none" else args.pretrained_weights

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    print("Loading file list ...")
    paths, labels, classes, class_to_idx = list_images(args.data_dir)
    num_classes = len(classes)
    print(f"Found {len(paths)} images across {num_classes} classes: {classes}")

    # Stratified 70/15/15 split
    train_p, temp_p, train_y, temp_y = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=42
    )
    val_p, test_p, val_y, test_y = train_test_split(
        temp_p, temp_y, test_size=0.50, stratify=temp_y, random_state=42
    )
    print(f"Train: {len(train_p)}  Val: {len(val_p)}  Test: {len(test_p)}")

    train_ds = make_dataset(train_p, train_y, args.img_size, args.batch_size, num_classes,
                             augment=True, shuffle=True)
    val_ds = make_dataset(val_p, val_y, args.img_size, args.batch_size, num_classes)
    test_ds = make_dataset(test_p, test_y, args.img_size, args.batch_size, num_classes)

    class_weights_arr = compute_class_weight("balanced", classes=np.unique(train_y), y=train_y)
    class_weight = {i: w for i, w in enumerate(class_weights_arr)}
    print("Class weights:", {classes[i]: round(w, 2) for i, w in class_weight.items()})

    print("Building model (MobileNetV2 transfer learning, frozen base) ...")
    model = build_transfer_model(num_classes, img_size=args.img_size, fine_tune_at=None,
                                  weights=weights_arg)
    compile_model(model, lr=args.lr)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(
            os.path.join(args.model_dir, "best_model.keras"),
            monitor="val_accuracy", save_best_only=True,
        ),
    ]

    print("\n=== Stage 1: training classification head (base frozen) ===")
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs,
        class_weight=class_weight, callbacks=callbacks,
    )

    if args.fine_tune_epochs > 0:
        print("\n=== Stage 2: fine-tuning top layers of MobileNetV2 ===")
        model = build_transfer_model(num_classes, img_size=args.img_size, fine_tune_at=100,
                                      weights=weights_arg)
        model.load_weights(os.path.join(args.model_dir, "best_model.keras"))
        compile_model(model, lr=args.lr / 10)
        history_ft = model.fit(
            train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs,
            class_weight=class_weight, callbacks=callbacks,
        )
        for k in history.history:
            history.history[k] += history_ft.history[k]

    plot_history(history, args.out_dir)

    print("\n=== Evaluating on held-out test set ===")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc * 100:.2f}%   Test loss: {test_loss:.4f}")

    y_true, y_pred = [], []
    for x, y in test_ds:
        preds = model.predict(x, verbose=0)
        y_true.extend(np.argmax(y.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    report = classification_report(y_true, y_pred, target_names=classes, digits=3)
    print("\nClassification report:\n", report)
    with open(os.path.join(args.out_dir, "classification_report.txt"), "w") as f:
        f.write(f"Test accuracy: {test_acc * 100:.2f}%\n\n")
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, classes, args.out_dir)

    final_path = os.path.join(args.model_dir, "leaf_classifier.keras")
    model.save(final_path)
    with open(os.path.join(args.model_dir, "class_indices.json"), "w") as f:
        json.dump({"classes": classes, "img_size": args.img_size}, f, indent=2)

    print(f"\nSaved final model to {final_path}")
    print(f"Saved class index mapping to {os.path.join(args.model_dir, 'class_indices.json')}")
    print(f"Saved plots + report to {args.out_dir}")


if __name__ == "__main__":
    main()

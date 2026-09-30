"""
model.py
--------
CNN architecture for medicinal plant leaf classification.

Uses transfer learning on MobileNetV2 (pretrained on ImageNet) as a
feature extractor, with a small custom classification head on top.
This is the standard, resource-friendly approach for leaf/plant
classification projects and is what lets a modest dataset reach good
accuracy without training a CNN from scratch.

A `build_simple_cnn` fallback (plain CNN, no transfer learning) is also
provided in case you want to compare against training from scratch.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_transfer_model(num_classes: int, img_size: int = 224, fine_tune_at: int | None = None,
                          weights: str | None = "imagenet"):
    """
    Build a MobileNetV2-based transfer learning classifier.

    Args:
        num_classes: number of leaf species to classify.
        img_size: input image side length (square).
        fine_tune_at: if given, unfreeze base-model layers from this
            index onward for fine-tuning (call again after initial
            training). Leave None to keep the base fully frozen.
        weights: "imagenet" (default, requires internet access on first
            run to download pretrained weights) or None to train the
            same architecture from random initialization (useful for
            offline testing, but needs far more data/epochs to converge).
    """
    base_model = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights=weights,
    )

    if fine_tune_at is None:
        base_model.trainable = False
    else:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False if fine_tune_at is None else None)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="medicinal_leaf_classifier")
    return model


def build_simple_cnn(num_classes: int, img_size: int = 224):
    """A plain CNN trained from scratch -- useful as a baseline comparison."""
    model = keras.Sequential([
        keras.Input(shape=(img_size, img_size, 3)),
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ], name="simple_leaf_cnn")
    return model


def compile_model(model, lr: float = 1e-3):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

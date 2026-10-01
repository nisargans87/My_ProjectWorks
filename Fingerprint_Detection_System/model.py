import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

IMG_SIZE = 128

DATASET_DIR = "dataset/SOCOFing"

images = []
labels = []

valid_extensions = (".bmp", ".png", ".jpg", ".jpeg")


def get_label(filename):
    """
    Example:
    1__M_Left_index_finger_CR.BMP

    Label becomes:
    Left_index
    """

    name = os.path.basename(filename)

    parts = name.split("_")

    if len(parts) < 5:
        return None

    hand = parts[2]
    finger = parts[3]

    return hand + "_" + finger


print("Scanning dataset...")

for root, dirs, files in os.walk(DATASET_DIR):

    for file in files:

        if file.lower().endswith(valid_extensions):

            path = os.path.join(root, file)

            label = get_label(file)

            if label is None:
                continue

            image = cv2.imread(path)

            if image is None:
                continue

            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

            image = image.astype("float32") / 255.0

            images.append(image)

            labels.append(label)

print("Images Loaded :", len(images))
print("Labels Loaded :", len(labels))

X = np.array(images)

encoder = LabelEncoder()

y = encoder.fit_transform(labels)

num_classes = len(np.unique(y))

print("Classes :", encoder.classes_)

y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
print("\nBuilding CNN Model...")

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Flatten(),

    Dense(256, activation="relu"),
    Dropout(0.5),

    Dense(128, activation="relu"),
    Dropout(0.3),

    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
print("\nStarting Training...\n")

history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

print("\nEvaluating Model...\n")

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print(f"\nTest Accuracy : {accuracy*100:.2f}%")
print(f"Test Loss     : {loss:.4f}")

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save trained model
model.save("model/fingerprint_classifier.h5")

# Save class names
with open("model/class_names.txt", "w") as f:
    for cls in encoder.classes_:
        f.write(cls + "\n")

print("\nModel saved successfully.")

print("Saved files:")
print("model/fingerprint_classifier.h5")
print("model/class_names.txt")
import random

print("\nTesting the trained model...")

# Select a random test image
index = random.randint(0, len(X_test) - 1)

sample = X_test[index]

prediction = model.predict(
    np.expand_dims(sample, axis=0),
    verbose=0
)

predicted_class = encoder.inverse_transform(
    [np.argmax(prediction)]
)[0]

actual_class = encoder.inverse_transform(
    [np.argmax(y_test[index])]
)[0]

print("-------------------------------------")
print("Actual Class    :", actual_class)
print("Predicted Class :", predicted_class)
print("Confidence      : {:.2f}%".format(np.max(prediction) * 100))
print("-------------------------------------")

print("\nTraining Completed Successfully.")
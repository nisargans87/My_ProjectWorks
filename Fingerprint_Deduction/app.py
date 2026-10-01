import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from database import init_db, add_detection, get_all_detections
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
MODEL_PATH = "model/fingerprint_classifier.h5"
CLASS_NAMES_PATH = "model/class_names.txt"
IMG_SIZE = (128, 128)
model = None
class_names = []
def load_model_if_available():
    """Lazily load the trained model. Returns True if a model is ready."""
    global model, class_names
    if model is not None:
        return True
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_NAMES_PATH):
        return False
    from tensorflow.keras.models import load_model
    model = load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH) as f:
        class_names = [line.strip() for line in f if line.strip()]
    return True
def predict_fingerprint(image_path):
    from tensorflow.keras.preprocessing import image as keras_image

    img = keras_image.load_img(image_path, target_size=IMG_SIZE)
    arr = keras_image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    predictions = model.predict(arr)[0]
    best_idx = int(np.argmax(predictions))
    return class_names[best_idx], float(predictions[best_idx])


@app.route("/", methods=["GET", "POST"])
def index():
    model_ready = load_model_if_available()

    if request.method == "POST":
        if not model_ready:
            return render_template(
                "index.html",
                model_ready=False,
                error="No trained model found yet. Run train.py first (see README).",
            )

        name = request.form.get("name", "").strip() or "Unknown"
        file = request.files.get("fingerprint")

        if not file or file.filename == "":
            return render_template("index.html", model_ready=True, error="Please choose an image file.")

        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        predicted_group, confidence = predict_fingerprint(save_path)
        record_id = add_detection(name, save_path, predicted_group, confidence)

        return render_template(
            "result.html",
            name=name,
            predicted_group=predicted_group,
            confidence=round(confidence * 100, 2),
            image_path=save_path,
            record_id=record_id,
        )

    return render_template("index.html", model_ready=model_ready, error=None)


@app.route("/history")
def history():
    records = get_all_detections()
    return render_template("history.html", records=records)
import sqlite3

@app.route("/clear_history", methods=["POST"])
def clear_history():
    conn = sqlite3.connect("fingerprint_group.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM detections")
    conn.commit()
    conn.close()
    return redirect(url_for("history"))
if __name__ == "__main__":
    init_db()
    app.run(debug=True)
@app.route("/clear_history", methods=["POST"])
def clear_history():
    import sqlite3

    conn = sqlite3.connect("fingerprint_group.db")
    cursor = conn.cursor()

    cursor.execute("DELETE FROM detections")

    conn.commit()
    conn.close()

    return redirect(url_for("history"))
@app.route("/delete/<int:id>", methods=["POST"])
def delete_record(id):
    import sqlite3

    conn = sqlite3.connect("fingerprint_group.db")
    cursor = conn.cursor()

    cursor.execute("DELETE FROM detections WHERE id=?", (id,))

    conn.commit()
    conn.close()

    return redirect(url_for("history"))
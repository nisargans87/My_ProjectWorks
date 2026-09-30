"""
app.py
------
Interactive Streamlit UI for the Medicinal Plant Leaf Classifier.

Run with:
    streamlit run app.py

Tabs:
    1. Classify a Leaf   - upload a photo, get a prediction + confidence chart
    2. Dataset Explorer  - browse class distribution and sample images
    3. Model Info        - architecture, training curves, confusion matrix

The app degrades gracefully: if the model hasn't been trained yet, the
Classify tab shows setup instructions instead of crashing, and the other
tabs still work off the dataset / EDA outputs.
"""

import json
import os

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Medicinal Plant Leaf Classifier", page_icon="🌿", layout="wide")

MODEL_PATH = "saved_model/leaf_classifier.keras"
CLASS_INFO_PATH = "saved_model/class_indices.json"
DATA_DIR = "data/raw"
EDA_DIR = "outputs/eda"
TRAINING_DIR = "outputs/training"

# Short, general-knowledge blurbs about each demo species (written from
# common knowledge, not copied from any single source). Extend this as
# you add real classes / a real dataset.
PLANT_INFO = {
    "Tulsi_Basil": {
        "common_name": "Tulsi (Holy Basil)",
        "scientific_name": "Ocimum tenuiflorum",
        "notes": "Widely grown in Indian households; leaves are commonly used in herbal teas "
                 "and traditional remedies for coughs and colds.",
    },
    "Neem": {
        "common_name": "Neem",
        "scientific_name": "Azadirachta indica",
        "notes": "A fast-growing tree valued in traditional medicine for skin care and as a "
                 "natural pesticide in agriculture.",
    },
    "Aloe_Vera": {
        "common_name": "Aloe Vera",
        "scientific_name": "Aloe barbadensis miller",
        "notes": "Succulent with thick, gel-filled leaves; the gel is popularly applied to "
                 "minor burns and used in skincare products.",
    },
    "Betel": {
        "common_name": "Betel Leaf",
        "scientific_name": "Piper betle",
        "notes": "Heart-shaped glossy leaves, traditionally chewed with areca nut in parts of "
                 "South and Southeast Asia, and used in some folk remedies.",
    },
    "Mint": {
        "common_name": "Mint (Pudina)",
        "scientific_name": "Mentha spp.",
        "notes": "Aromatic herb used widely in cooking, chutneys, and teas; commonly associated "
                 "with digestive comfort.",
    },
    "Curry_Leaf": {
        "common_name": "Curry Leaf",
        "scientific_name": "Murraya koenigii",
        "notes": "Aromatic leaves used extensively as a tempering spice in South Indian cooking.",
    },
    "Hibiscus": {
        "common_name": "Hibiscus",
        "scientific_name": "Hibiscus rosa-sinensis",
        "notes": "Ornamental flowering plant whose leaves and flowers are used in some hair-care "
                 "and herbal-tea preparations.",
    },
    "Guava": {
        "common_name": "Guava",
        "scientific_name": "Psidium guajava",
        "notes": "Common fruit tree; the leaves are used in some traditional preparations for "
                 "digestive wellness.",
    },
}


@st.cache_resource
def load_model():
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_INFO_PATH) as f:
        meta = json.load(f)
    return model, meta["classes"], meta["img_size"]


@st.cache_data
def load_manifest():
    path = os.path.join(EDA_DIR, "dataset_manifest.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def predict(model, img: Image.Image, img_size: int, classes):
    img_resized = img.convert("RGB").resize((img_size, img_size))
    arr = np.array(img_resized, dtype=np.float32)[None, ...]
    preds = model.predict(arr, verbose=0)[0]
    order = np.argsort(preds)[::-1]
    return [(classes[i], float(preds[i])) for i in order]


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
st.sidebar.title("🌿 Leaf Classifier")
st.sidebar.markdown(
    "Deep-learning classifier for medicinal plant leaves, built with "
    "**TensorFlow/Keras** (MobileNetV2 transfer learning) and analyzed "
    "with **Pandas** + **Matplotlib**."
)
manifest = load_manifest()
if manifest is not None:
    st.sidebar.metric("Total images", len(manifest))
    st.sidebar.metric("Classes", manifest["class"].nunique())
st.sidebar.markdown("---")
st.sidebar.caption(
    "Tip: run `python src/generate_demo_dataset.py` then `python src/train.py` "
    "to get a working model before using the Classify tab."
)

tab1, tab2, tab3 = st.tabs(["🔍 Classify a Leaf", "📊 Dataset Explorer", "🧠 Model Info"])

# ----------------------------------------------------------------------
# Tab 1: Classify
# ----------------------------------------------------------------------
with tab1:
    st.header("Classify a Medicinal Plant Leaf")

    if not os.path.exists(MODEL_PATH):
        st.warning(
            "No trained model found yet.\n\n"
            "Run these first:\n"
            "```bash\n"
            "python src/generate_demo_dataset.py   # or drop in your real dataset\n"
            "python src/train.py\n"
            "```"
        )
    else:
        model, classes, img_size = load_model()
        col_left, col_right = st.columns([1, 1.2])

        with col_left:
            uploaded = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])
            use_sample = st.checkbox("...or use a random sample from the dataset instead")
            top_k = st.slider("Show top-K predictions", 1, min(8, len(classes)), min(5, len(classes)))

            image = None
            if uploaded is not None:
                image = Image.open(uploaded)
            elif use_sample and manifest is not None:
                sample_row = manifest.sample(1).iloc[0]
                image = Image.open(sample_row["filepath"])
                st.caption(f"Sampled true label: **{sample_row['class']}**")

            if image is not None:
                st.image(image, caption="Input image", width='stretch')

        with col_right:
            if image is not None:
                with st.spinner("Running inference..."):
                    results = predict(model, image, img_size, classes)[:top_k]

                best_class, best_conf = results[0]
                info = PLANT_INFO.get(best_class, {})
                st.success(f"**Prediction: {info.get('common_name', best_class)}**  "
                           f"({best_conf * 100:.1f}% confidence)")
                if info:
                    st.markdown(f"*{info['scientific_name']}* — {info['notes']}")

                st.subheader("Confidence breakdown")
                chart_df = pd.DataFrame(results, columns=["class", "confidence"])
                chart_df["confidence"] = (chart_df["confidence"] * 100).round(2)
                st.bar_chart(chart_df.set_index("class"))
            else:
                st.info("Upload an image (or tick the sample-image box) to see a prediction here.")

# ----------------------------------------------------------------------
# Tab 2: Dataset Explorer
# ----------------------------------------------------------------------
with tab2:
    st.header("Dataset Explorer")

    if manifest is None:
        st.warning(
            "No dataset manifest found. Run:\n\n"
            "```bash\npython src/data_analysis.py\n```\n\nafter generating/placing a dataset."
        )
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total images", len(manifest))
        c2.metric("Classes", manifest["class"].nunique())
        c3.metric("Avg file size (KB)", f"{manifest['filesize_kb'].mean():.1f}")

        st.subheader("Images per class")
        st.bar_chart(manifest["class"].value_counts())

        st.subheader("Browse sample images")
        chosen_class = st.selectbox("Pick a class", sorted(manifest["class"].unique()))
        subset = manifest[manifest["class"] == chosen_class].sample(
            min(8, (manifest["class"] == chosen_class).sum()), random_state=1
        )
        cols = st.columns(4)
        for i, (_, row) in enumerate(subset.iterrows()):
            with cols[i % 4]:
                st.image(row["filepath"], width='stretch')

        info = PLANT_INFO.get(chosen_class)
        if info:
            st.markdown(f"**{info['common_name']}** (*{info['scientific_name']}*) — {info['notes']}")

# ----------------------------------------------------------------------
# Tab 3: Model Info
# ----------------------------------------------------------------------
with tab3:
    st.header("Model Info & Training Results")

    if os.path.exists(MODEL_PATH):
        model, classes, img_size = load_model()
        st.write(f"**Input size:** {img_size}x{img_size}x3  |  **Classes:** {len(classes)}  |  "
                 f"**Trainable params:** {sum(np.prod(v.shape) for v in model.trainable_weights):,}")
        with st.expander("Full architecture summary"):
            lines = []
            model.summary(print_fn=lambda x: lines.append(x))
            st.code("\n".join(lines))
    else:
        st.info("Train a model first (`python src/train.py`) to see architecture details here.")

    curves_path = os.path.join(TRAINING_DIR, "training_curves.png")
    cm_path = os.path.join(TRAINING_DIR, "confusion_matrix.png")
    report_path = os.path.join(TRAINING_DIR, "classification_report.txt")

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(curves_path):
            st.subheader("Training curves")
            st.image(curves_path, width='stretch')
    with col2:
        if os.path.exists(cm_path):
            st.subheader("Confusion matrix (test set)")
            st.image(cm_path, width='stretch')

    if os.path.exists(report_path):
        with st.expander("Full classification report"):
            st.text(open(report_path).read())

# Medicinal Plant Leaves Classification Using Deep Learning 🌿

A complete, runnable project matching the resume bullets:

> Used Python with Pandas and Matplotlib to analyze and visualize medicinal plant data;
> Developed a deep learning classifier achieving 82% accuracy in identifying medicinal
> plant species.

It includes data analysis (Pandas/Matplotlib), a CNN classifier (TensorFlow/Keras,
MobileNetV2 transfer learning), and an **interactive Streamlit web app** for classifying
leaf photos in real time.

---

## Project structure

```
medicinal-plant-classifier/
├── app.py                        # Interactive Streamlit UI
├── requirements.txt
├── data/
│   └── raw/<class_name>/*.jpg    # One folder per species (dataset lives here)
├── src/
│   ├── generate_demo_dataset.py  # Creates a synthetic demo dataset (for instant testing)
│   ├── data_analysis.py          # Pandas + Matplotlib EDA
│   ├── model.py                  # CNN architecture (MobileNetV2 transfer learning)
│   ├── train.py                  # Full training pipeline
│   └── predict.py                # CLI single-image inference
├── saved_model/                  # Trained model + class mapping land here after training
└── outputs/
    ├── eda/                      # EDA plots + dataset_manifest.csv
    └── training/                 # Training curves, confusion matrix, classification report
```

## Quickstart (works immediately, no dataset needed)

The project ships with a **synthetic demo dataset generator** so you can run the entire
pipeline right away, before you've collected/downloaded real leaf photos.

```bash
pip install -r requirements.txt

# 1. Generate a small synthetic 8-class leaf dataset (Tulsi, Neem, Aloe Vera, Betel,
#    Mint, Curry Leaf, Hibiscus, Guava) for testing the pipeline end-to-end
python src/generate_demo_dataset.py --per_class 60

# 2. Run the EDA (Pandas + Matplotlib) — produces charts + dataset_manifest.csv
python src/data_analysis.py

# 3. Train the CNN (downloads ImageNet weights the first time — needs internet)
python src/train.py --epochs 15 --fine_tune_epochs 5

# 4. Launch the interactive web app
streamlit run app.py
```

The Streamlit app opens in your browser with three tabs:

- **🔍 Classify a Leaf** — upload a photo (or sample one from the dataset), see the
  predicted species with a confidence bar chart and a short plant info blurb.
- **📊 Dataset Explorer** — class distribution chart, dataset stats, and a browsable
  image gallery per class.
- **🧠 Model Info** — architecture summary, training curves, and confusion matrix.

## Using your own (real) dataset

Replace the synthetic images with actual photographs, keeping the same structure:

```
data/raw/
├── Tulsi/          *.jpg
├── Neem/           *.jpg
├── Aloe_Vera/      *.jpg
└── ... (your species)
```

Good sources for real medicinal-leaf datasets: the **"Indian Medicinal Leaf Image
Dataset"** on Mendeley Data, or similar medicinal-plant datasets on Kaggle. Once your
images are in place, steps 2–4 above work unchanged — the code reads class names
directly from the folder structure, so you can use any number of species.

**About the 82% accuracy figure:** transfer learning with MobileNetV2 is a strong,
standard baseline for leaf classification and commonly lands in the 80-90%+ range on
real photographed datasets, depending on image quality, number of classes, and dataset
size — consistent with the 82% claimed in the resume bullet. The synthetic demo dataset
included here is only for verifying the pipeline runs correctly (its shapes are very
easy to separate, so it will train much faster than real leaf photos); your real
accuracy number should come from training on an actual photographed dataset.

## How each resume line maps to the code

| Resume line | Where it lives |
|---|---|
| "Used Python with Pandas and Matplotlib to analyze and visualize medicinal plant data" | `src/data_analysis.py` — builds a Pandas DataFrame of every image (class, resolution, file size, brightness) and produces 5 Matplotlib figures |
| "Developed a deep learning classifier achieving 82% accuracy" | `src/model.py` + `src/train.py` — MobileNetV2 transfer-learning CNN with augmentation, class-weighting, and two-stage fine-tuning; test accuracy is printed and saved to `outputs/training/classification_report.txt` after training |
| "Make UI interactive" | `app.py` — Streamlit app with live image upload, prediction, confidence charts, and a dataset browser |

## Tips for extending this for placement interviews

- Swap `MobileNetV2` in `model.py` for `EfficientNetB0` or `ResNet50` and compare
  results — good talking point on architecture trade-offs.
- Add Grad-CAM visualization to the Streamlit app to show *why* the model predicted a
  class (highlights which leaf regions influenced the decision) — a strong interview
  demo.
- Track experiments with different image sizes / augmentation strengths and report
  the effect on validation accuracy — shows you understand the tuning process, not just
  the final number.

## Requirements

See `requirements.txt`. Core stack: TensorFlow/Keras, Streamlit, Pandas, Matplotlib,
scikit-learn, Pillow.

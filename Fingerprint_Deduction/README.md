# Blood Group Detection (Fingerprint-based ML)

Predicts blood group (A+, A-, B+, B-, AB+, AB-, O+, O-) from a
fingerprint image using a CNN, and logs every prediction to a
SQLite database.

> Note: this predicts from fingerprint ridge patterns using a
> trained ML model — it is a college-project-style demo, not a
> medical diagnostic tool. A real blood group can only be
> confirmed by an actual lab test.

## Project structure

```
blood_group_detection/
├── app.py              Flask web app (upload -> predict -> store -> history)
├── model.py             CNN definition + training script
├── database.py          SQLite helper functions
├── requirements.txt
├── dataset/              <- you add this: one subfolder per class
│   ├── A+/  A-/  B+/  B-/  AB+/  AB-/  O+/  O-/
├── model/                created after training (saved .h5 + class list)
├── static/uploads/       uploaded images get saved here
└── templates/            HTML pages
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

## 1. Get a dataset

You need fingerprint images sorted into 8 folders under `dataset/`,
one per blood group, e.g.:

```
dataset/A+/fingerprint001.jpg
dataset/A+/fingerprint002.jpg
dataset/O-/fingerprint045.jpg
...
```

Search **"fingerprint based blood group detection dataset"** on
Kaggle — a ready-made labeled dataset exists there under that
name. Download and extract it into `dataset/` matching the
structure above.

## 2. Train the model

```bash
python model.py
```

This trains a CNN on your dataset and saves:
- `model/blood_group_model.h5` — the trained model
- `model/class_names.txt` — the class order used for predictions

Adjust `epochs` in `model.py`'s `train()` call if you want longer/shorter training.

## 3. Run the web app

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser:
- Upload a fingerprint image + enter a name → get an instant predicted blood group with confidence score
- Every prediction is saved to `blood_group.db` (SQLite)
- Visit `/history` to see all past detections pulled from the database

## Notes for your report/viva

- **Model**: 3-block CNN (Conv2D + MaxPooling) → Dense → Softmax, 8-class classification
- **Database**: SQLite, table `detections` (id, name, image_path, predicted_group, confidence, created_at)
- **Framework**: Flask for the web interface
- If asked why fingerprints can predict blood group: this project is based on published research suggesting statistical correlation between fingerprint ridge patterns and ABO blood groups — the model learns those patterns from labeled training data, it does not do any biological/chemical analysis.

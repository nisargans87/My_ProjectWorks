# Fingerprint Classification Web App

This project is a Flask-based fingerprint classification application that uses a Convolutional Neural Network (CNN) trained on fingerprint images to predict a class such as `F_Left`, `F_Right`, `M_Left`, or `M_Right`. It provides a simple web interface for uploading a fingerprint image, viewing the prediction result, and storing history in a local SQLite database.

This repository is intended for academic/demo use and is not a production-grade biometric identity system.

## Overview

The application workflow is:

1. User uploads a fingerprint image through the web interface.
2. The app preprocesses the image and passes it to a trained TensorFlow CNN.
3. The model predicts the fingerprint class from the trained dataset.
4. The result is displayed to the user with a confidence score.
5. Prediction details are saved to a SQLite database and shown on the history page.

## Technical Stack

- Python 3.x
- Flask for the web application
- TensorFlow / Keras for the CNN model
- OpenCV for image loading and resizing
- NumPy for image array processing
- scikit-learn for label encoding and train/test splitting
- SQLite for local prediction history storage
- HTML/CSS/JavaScript for the frontend UI

## Project Structure

```text
Fingerprint_Deduction/
├── app.py                  # Flask application; handles upload, prediction, and routing
├── model.py                # CNN model architecture and training pipeline
├── database.py             # SQLite functions for storing and fetching detections
├── requirements.txt        # Python dependencies
├── dataset/
│   └── SOCOFing/           # Dataset used for training the model
├── model/
│   ├── fingerprint_classifier.h5   # Trained model file
│   └── class_names.txt             # Label names used by the model
├── static/
│   ├── script.js           # Frontend interactions
│   ├── style.css           # Page styling
│   └── uploads/            # Uploaded fingerprint images are saved here
├── templates/
│   ├── index.html          # Upload page
│   ├── result.html         # Prediction result page
│   └── history.html        # History and database display
├── fingerprint_group.db    # SQLite database generated at runtime
├── README.md               # Project documentation
└── .gitignore              # Optional environment and generated file exclusions
```

## Dataset

The model is trained using the SOCOFing dataset stored under:

```text
dataset/SOCOFing/
```

The training script reads image files from the dataset, extracts labels from filenames such as:

```text
1__M_Left_index_finger_CR.BMP
```

and converts them into labels like:

- `M_Left`
- `M_Right`
- `F_Left`
- `F_Right`

This project therefore classifies fingerprint samples according to the SOCOFing label scheme rather than performing identity verification or real biometric authentication.

## Setup and Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd Fingerprint_Deduction
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Model Training

Before launching the app, ensure the trained model exists in the `model/` directory.

If you need to retrain the model:

```bash
python model.py
```

This script:

- loads the fingerprint images from `dataset/SOCOFing`
- resizes and normalizes the images
- encodes class labels
- builds a CNN model
- trains the network
- saves:
  - `model/fingerprint_classifier.h5`
  - `model/class_names.txt`

## Run the Application

Start the Flask web server:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

### Features in the web app

- Upload fingerprint image
- Enter a user name
- Predict the fingerprint class
- Display confidence score
- Save the result to SQLite
- View prediction history at `/history`

## How the Model Works

The project uses a CNN architecture with:

- convolutional layers
- max-pooling layers
- flattening layer
- dense layers
- dropout regularization
- softmax output for multi-class classification

This architecture is suitable for image-based classification tasks and is implemented in `model.py`.

## Database

The application uses SQLite and creates the database automatically when the app starts.

The main `detections` table includes fields such as:

- `id`
- `name`
- `image_path`
- `predicted_group`
- `confidence`
- `created_at`

The database file is:

```text
fingerprint_group.db
```

## Limitations

This project is a research/demo implementation and has several important limitations:

- It is not a secure biometric authentication system.
- It does not perform identity matching against a stored enrollment database.
- It does not include liveness detection, anti-spoofing checks, or presentation attack detection.
- Accuracy depends heavily on the quality, consistency, and balance of the dataset.
- The model is trained on a specific dataset (SOCOFing) and may not generalize well to other environments or devices.
- The app stores results locally in SQLite, which is not suitable for production-grade multi-user systems.
- Predictions are statistical and should not be treated as definitive real-world biometric judgments.
- Image noise, lighting differences, background clutter, or low-quality scans can affect performance.

## Important Note

This project is best understood as an educational machine-learning exercise for fingerprint image classification. It should not be used for security-sensitive or legally critical fingerprint-based identification without a proper biometric system design, validation pipeline, and domain-specific evaluation.

## Future Improvements

Possible enhancements include:

- adding preprocessing pipelines for better image normalization
- experimenting with transfer learning (e.g., MobileNet, ResNet)
- improving dataset quality and class balance
- adding model evaluation metrics and confusion matrix reports
- upgrading the database to a more scalable backend
- adding authentication or role-based user management
- cleaning and modularizing the Flask routes and training code

## License

This project is provided as-is for educational and research purposes. Please check the repository license if one is added later.

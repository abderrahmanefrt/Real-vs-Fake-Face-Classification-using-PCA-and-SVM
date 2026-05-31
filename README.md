# Face Anti-Spoofing using PCA and SVM-RBF

## Overview

This project implements a complete Face Anti-Spoofing (Liveness Detection) pipeline using Machine Learning techniques.

The objective is to classify facial images as:

- Real (Genuine User)
- Fake (Spoofing Attack)

The system combines image preprocessing, dimensionality reduction through Principal Component Analysis (PCA), and classification using a Support Vector Machine (SVM) with an RBF kernel.



## Features

- Automatic image loading and preprocessing
- Face extraction and normalization
- Contrast enhancement (CLAHE)
- PCA-based dimensionality reduction
- SVM-RBF classification
- Validation and performance evaluation
- Confusion matrix generation
- PCA statistical analysis
- Eigenvalue and explained variance visualization
- Individual and variable contribution analysis
- Interactive testing on external images
- Support for RGB, Depth, and RGB+Depth modalities



## Project Pipeline

```text
Input Images
      │
      ▼
Preprocessing
(Resize, Grayscale, CLAHE)
      │
      ▼
Feature Matrix Construction
      │
      ▼
Standardization (Z-score)
      │
      ▼
Principal Component Analysis (PCA)
      │
      ▼
Feature Projection
      │
      ▼
SVM (RBF Kernel)
      │
      ▼
Real / Fake Prediction
```


## Technologies Used

- Python
- OpenCV
- NumPy
- Pandas
- Scikit-Learn
- Matplotlib
- Seaborn
- Joblib



## Machine Learning Methods

### Principal Component Analysis (PCA)

PCA is used to:

- Reduce dimensionality
- Remove redundant information
- Preserve the most informative facial features
- Improve classifier performance

### Support Vector Machine (SVM-RBF)

The classifier is trained using an RBF kernel to:

- Separate genuine and spoofed faces
- Handle nonlinear decision boundaries
- Improve classification accuracy


## Dataset Structure

```text
dataset/
│
├── train_img/
│   ├── color/
│   └── depth/
│
├── test_img/
│   ├── color/
│   └── depth/
│
└── labels/
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/face-anti-spoofing-pca-svm.git
cd face-anti-spoofing-pca-svm
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Notebook

Open:

```bash
final_examples.ipynb
```

Then execute all cells sequentially.

---

## Results

The notebook provides:

- Accuracy Score
- Balanced Accuracy
- F1 Score
- Classification Report
- Confusion Matrix
- PCA Variance Analysis
- Individual Projection Plots
- Variable Contribution Analysis

---

## Example Applications

- Face Authentication Systems
- Biometric Access Control
- Mobile Device Security
- Online Identity Verification
- Anti-Spoofing Research



## Future Improvements

- Deep Learning models (CNNs)
- Vision Transformers (ViTs)
- Real-time webcam detection
- Mobile deployment
- Multi-modal fusion
- Explainable AI techniques


## Author

Abdou Ferhat

Computer Vision & Machine Learning Student

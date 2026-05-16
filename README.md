# Diabetic Retinopathy Classification Using CNN on APTOS 2019

This repository contains a TensorFlow/Keras implementation of a Convolutional Neural Network (CNN) for multiclass diabetic retinopathy classification using the APTOS 2019 fundus image dataset.

The code is organized in a research-friendly structure so that preprocessing, model training, evaluation, visualization, and configuration are separated into clean Python files.

## Project Overview

Diabetic retinopathy is a diabetes-related retinal disease that can lead to vision impairment if not detected early. This project applies deep learning and computer vision methods to classify retinal fundus images into diabetic retinopathy severity grades.

## Repository Structure

```text
diabetic-retinopathy-cnn-aptos2019/
│
├── README.md
├── requirements.txt
├── .gitignore
├── run_experiment.py
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualization.py
│
├── figures/
│   └── .gitkeep
│
├── results/
│   └── .gitkeep
│
└── models/
    └── .gitkeep
```

## Dataset

This project is designed for the APTOS 2019 Blindness Detection dataset.

Expected dataset structure:

```text
Aptos2019_dataset/
│
├── train.csv
└── train_images/
    ├── image_1.png
    ├── image_2.png
    └── ...
```

The CSV file should contain:

```text
id_code,diagnosis
```

where `id_code` is the image filename without `.png`, and `diagnosis` is the class label.

## Diabetic Retinopathy Classes

The APTOS 2019 dataset commonly uses five severity grades:

| Label | Class |
|---|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

## Main Features

- Fundus image loading and preprocessing
- Image resizing to 224 × 224
- Pixel normalization
- Stratified train/validation/test split
- CNN-based multiclass classification
- Data augmentation
- Accuracy and loss curves
- Confusion matrix
- ROC-AUC curve using one-vs-rest strategy
- Precision, recall, F1-score, accuracy, QWK
- Multiclass sensitivity and specificity
- Saved results and trained model

## Installation

Create a Python environment and install the requirements:

```bash
pip install -r requirements.txt
```

## How to Run

Update the dataset path in `src/config.py`:

```python
DATA_DIR = "./Aptos2019_dataset"
```

Then run:

```bash
python run_experiment.py
```

## Output Files

After running the experiment, the following outputs are saved:

```text
results/
├── classification_report.txt
├── metrics.json
├── confusion_matrix.csv
├── training_history.csv
│
figures/
├── confusion_matrix.png
├── roc_auc_curve.png
├── accuracy_curve.png
└── loss_curve.png
│
models/
└── cnn_aptos2019_model.keras
```

## Model Architecture

The CNN model contains:

- Four convolutional blocks
- Max-pooling layers
- Flatten layer
- Dense layer
- Dropout regularization
- Softmax classification layer

## Important Notes

Do not upload the full APTOS dataset to GitHub. Upload only the source code, README, requirements file, and sample result figures if needed.

The original local Windows path has been replaced with a relative dataset path so that the project is reusable across systems.

## Author

Farah Jabeen  
PhD Candidate  
Department of Electrical and Electronic Engineering Science  
University of Johannesburg, South Africa

## Research Areas

- Artificial Intelligence
- Machine Learning
- Deep Learning
- Computer Vision
- Biomedical Image Analysis
- Retinal Disease Diagnosis

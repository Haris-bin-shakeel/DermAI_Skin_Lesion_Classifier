# DermAI — Skin Lesion Classification with CNNs

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-API-red?style=flat-square&logo=keras)
![Colab](https://img.shields.io/badge/Google%20Colab-Ready-yellow?style=flat-square&logo=googlecolab)
![Dataset](https://img.shields.io/badge/Dataset-HAM10000-lightgrey?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

A practical end-to-end CNN pipeline for multi-class skin lesion classification using the **HAM10000** dermatology dataset. Built with TensorFlow/Keras and designed for reproducibility, clean experimentation, and easy adaptation to similar medical imaging tasks.

> ⚠️ **Disclaimer**: This project is for educational and research purposes only. It is not intended for clinical use.

---

## Overview

This project demonstrates a complete image classification workflow applied to dermatoscopic images — from raw data loading and preprocessing through model training, evaluation, and inference. The goal is to produce a well-structured, reproducible baseline that can be extended for real-world or client-facing computer vision tasks.

The workflow and engineering structure used here can also be adapted to related computer vision tasks such as gaze estimation, medical imaging, and image-based classification systems.

**Problem**: Classify skin lesion images into one of three medically relevant categories.  
**Approach**: Custom CNN trained on 128×128 RGB images using TensorFlow/Keras.  
**Platform**: Fully runnable on Google Colab (no GPU required for small-scale experiments).

---

## Key Features

- Real-world medical imaging dataset (HAM10000, 10,000+ annotated images)
- Full pipeline: data loading → preprocessing → training → evaluation → inference
- Custom CNN architecture built from scratch with TensorFlow/Keras
- Stratified train/test split for reliable evaluation
- Early stopping to prevent overfitting
- Accuracy and loss curve visualization
- Manual inference with custom image input
- Clean, modular notebook structure — easy to follow and modify
- Google Colab compatible — no local setup needed

---

## Dataset

**Source**: [HAM10000 on Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)  
**Description**: A curated collection of dermatoscopic images annotated by medical experts, containing 10,000+ labeled lesion images across multiple classes.

### Classes Used

| Label | Description | Type |
|-------|-------------|------|
| `nv` | Melanocytic Nevi | Benign (mole) |
| `mel` | Melanoma | Malignant |
| `bcc` | Basal Cell Carcinoma | Malignant |

> Dataset is not included in this repo due to size. Download from the Kaggle link above and place it in the `data/` directory as described in the [Quick Start](#quick-start) section.

---

## Project Structure

```
DermAI_Skin_Lesion_Classifier/
│
├── notebooks/
│   └── DermAI_Skin_Lesion_Classifier.ipynb   # Main pipeline notebook
│
├── src/
│   ├── preprocess.py      # Image loading, resizing, normalization
│   ├── model.py           # CNN architecture definition
│   └── evaluate.py        # Metrics, plots, confusion matrix
│
├── inference/
│   └── predict.py         # Run predictions on new images
│
├── outputs/
│   ├── plots/             # Training curves, confusion matrix
│   └── saved_model/       # Trained model weights (.h5 or SavedModel)
│
├── data/                  # Place HAM10000 dataset here (not tracked by git)
│   ├── HAM10000_images_part1/
│   ├── HAM10000_images_part2/
│   └── HAM10000_metadata.csv
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Preprocessing Pipeline

Raw images are processed through a consistent pipeline before training:

1. **Metadata loading** — Read labels and image paths from `HAM10000_metadata.csv`
2. **Class filtering** — Select `nv`, `mel`, and `bcc` classes only
3. **Label encoding** — Map string labels to integer indices
4. **Image resizing** — Standardize all images to `128 × 128` pixels
5. **Normalization** — Scale pixel values to `[0, 1]`
6. **Train/test split** — Stratified 80/20 split to preserve class distribution
7. **tf.data pipeline** — Efficient batching and prefetching for training

---

## Model Architecture

A custom CNN built with TensorFlow's Sequential API:

```
Input: (128, 128, 3)
  │
  ├── Conv2D(32, 3×3, ReLU) → MaxPooling2D(2×2)
  ├── Conv2D(64, 3×3, ReLU) → MaxPooling2D(2×2)
  ├── Conv2D(128, 3×3, ReLU) → MaxPooling2D(2×2)
  │
  ├── Flatten()
  ├── Dense(128, ReLU)
  ├── Dropout(0.5)
  │
  └── Dense(3, Softmax)  →  Output: class probabilities
```

**Loss function**: Sparse Categorical Crossentropy  
**Optimizer**: Adam  
**Output**: Softmax probabilities over 3 classes

---

## Training Pipeline

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=30,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
)
```

- Early stopping monitors validation loss and restores best weights automatically
- Training and validation metrics are logged per epoch
- Plots are saved to `outputs/plots/`

---

## Evaluation

After training, the model is evaluated with:

- **Accuracy** on held-out test set
- **Loss curves** — Training vs validation (to inspect overfitting)
- **Manual inference** — Predict on individual images and compare to ground truth

> Note: Advanced metrics (ROC AUC, precision/recall per class, confusion matrix) are included in the [Future Improvements](#future-improvements) roadmap and partially implemented in `src/evaluate.py`.

---

## Inference / Demo

To run a prediction on a new image:

```python
from src.preprocess import load_and_preprocess_image
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('outputs/saved_model/dermai_model.h5')

img = load_and_preprocess_image('path/to/your/image.jpg')  # returns (1, 128, 128, 3)
pred = model.predict(img)
class_labels = ['nv', 'mel', 'bcc']
print(f"Predicted class: {class_labels[np.argmax(pred)]}")
```

Or run the full notebook interactively in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Haris-bin-shakeel/DermAI_Skin_Lesion_Classifier/blob/main/notebooks/DermAI_Skin_Lesion_Classifier.ipynb)
---

## Quick Start

### Prerequisites

Python 3.8+ and the following packages:

```bash
pip install -r requirements.txt
```

**`requirements.txt`**:
```
tensorflow>=2.10
numpy
pandas
matplotlib
seaborn
scikit-learn
Pillow
```

### Running the Project

```bash
# 1. Clone the repository
git clone https://github.com/Haris-bin-shakeel/DermAI_Skin_Lesion_Classifier.git
cd DermAI_Skin_Lesion_Classifier

# 2. Download the HAM10000 dataset from Kaggle and place it in data/

# 3. Launch the notebook
jupyter notebook notebooks/DermAI_Skin_Lesion_Classifier.ipynb
```

Or open directly in Google Colab using the badge above.

---

## Reproducibility

To reproduce results:
- Set random seeds for NumPy and TensorFlow at the top of the notebook (`np.random.seed(42)`, `tf.random.set_seed(42)`)
- Use the same stratified split via `sklearn.model_selection.train_test_split` with `random_state=42`
- Dataset version: HAM10000 from Kaggle (linked above)

---

## Future Improvements

The following enhancements are planned or partially implemented:

- [ ] **Data Augmentation** — Random flips, rotations, and zoom to improve generalization
- [ ] **Class imbalance handling** — Class weights or oversampling (SMOTE-style)
- [ ] **Transfer learning** — EfficientNetB0 or MobileNetV2 as feature extractor
- [ ] **Advanced metrics** — ROC AUC, per-class precision/recall, confusion matrix
- [ ] **Hyperparameter tuning** — Learning rate schedules, batch size experiments
- [ ] **Gradio demo** — Simple web UI for image upload and prediction

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| TensorFlow / Keras | Model building and training |
| NumPy / Pandas | Data manipulation |
| Matplotlib / Seaborn | Visualization |
| scikit-learn | Evaluation and splitting |
| Google Colab | Cloud GPU training |

---

## About

Built by Muhammad Haris as part of a practical ML engineering portfolio focused on computer vision and deep learning applications.

Feel free to open an issue or reach out if you'd like to collaborate or have questions about the implementation.

## Connect

- GitHub: https://github.com/Haris-bin-shakeel
- LinkedIn: https://www.linkedin.com/in/haris-shakeel-aa1186330/
- Email: harisshakeel0981@gmail.com

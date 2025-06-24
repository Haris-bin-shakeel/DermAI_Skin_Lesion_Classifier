# DermAI — Skin Lesion Classification Using CNNs

A deep learning project aimed at automated skin lesion classification using the **HAM10000** dataset. This project demonstrates the practical application of image classification techniques with Convolutional Neural Networks (CNNs), marking a key milestone in my machine learning journey, guided by the **Deep Learning Specialization by Andrew Ng**.

---

## Project Overview

Early detection of skin cancer is critical. This project leverages deep learning to build a CNN-based classifier capable of distinguishing between common skin lesions using dermatoscopic images. While primarily educational, the project reflects **industry-standard best practices** in data handling, model development, and evaluation.

---

## Key Features

✅ Utilizes real-world medical imagery (HAM10000 dataset)  
✅ End-to-end machine learning pipeline for image classification  
✅ Custom CNN model built from scratch with TensorFlow/Keras  
✅ Focused classification on **three** medically significant lesion types  
✅ Model evaluation with accuracy metrics and visualizations  
✅ Clean code structure, ideal for experimentation and learning

---

## Dataset Details

**Dataset**: [HAM10000 - Human Against Machine with 10000 training images](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)  
A curated dermatology dataset containing dermatoscopic images of pigmented skin lesions, annotated by medical experts.

### Classes Used in This Project:

| Class Label | Description                          |
|-------------|--------------------------------------|
| `nv`        | Benign Melanocytic Nevi (Mole)       |
| `mel`       | Melanoma (Skin Cancer)               |
| `bcc`       | Basal Cell Carcinoma                 |

⚠️ **Note**: Dataset is not included in this repository due to size constraints. Please download from the Kaggle link above.

---

##  Project Pipeline

### Data Loading & Exploration
- Load metadata and image paths
- Visualize class distributions

### Data Filtering & Preprocessing
- Filter for `nv`, `mel`, and `bcc` classes
- Encode labels and normalize images
- Resize all images to 128x128
- Create efficient data pipelines with `tf.data`

### Model Development
- Custom CNN built using TensorFlow/Keras
- Layers: Conv2D → MaxPooling2D → Dropout → Dense
- Activation functions: ReLU and Softmax
- Loss: Sparse Categorical Crossentropy

### Model Training
- Stratified train-test split
- Early stopping to prevent overfitting
- Realtime training monitoring via validation accuracy/loss

### Model Evaluation
- Accuracy/loss curve visualization
- Manual prediction testing with custom image input
- Ground truth comparison from metadata

---

## 🧠 Model Architecture

```python
model = Sequential([
    Input(shape=(128, 128, 3)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])
```
---
### Future Work & Roadmap

* ✔️ Data Augmentation to improve generalization
* ✔️ Address class imbalance using class weights or oversampling
* ✔️ Implement transfer learning with pretrained CNNs (e.g., EfficientNet, MobileNet)
* ✔️ Hyperparameter tuning with callbacks and learning rate schedules
* ✔️ Evaluate with advanced metrics like ROC AUC, precision, recall

---
### Tech Stack

* Python
* TensorFlow / Keras
* NumPy, Pandas
* Matplotlib, Seaborn
* scikit-learn

---

### ⚠️Disclaimer

This project is intended for educational purposes only.

It is not suitable for clinical diagnosis or deployment in medical settings without extensive validation and regulatory approvals.

---

### Quick Start Guide

#### Prerequisites

Ensure Python 3.x is installed and run the following to install dependencies:

```bash
pip install -r requirements.txt
-Run the Project
Bash

# Clone the repository
git clone [https://github.com/yourusername/DermAI_Skin_Lesion_Classifier.git](https://github.com/yourusername/DermAI_Skin_Lesion_Classifier.git)

cd DermAI_Skin_Lesion_Classifier

# Launch the notebook
jupyter notebook DermAI_Skin_Lesion_Classifier.ipynb

```
---

### About Me

I am an aspiring machine learning engineer, currently progressing. This project reflects my practical understanding of CNNs and my commitment to applying AI to real-world challenges.



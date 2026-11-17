# Gaza Structural Damage Assessment AI
### Deep Learning (MobileNetV2) & Explainable AI (Grad-CAM) for Building Integrity Triage

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B%20%2F%202.21-orange.svg)](https://tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production%20Ready-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/PyTest-14%2F14%20Passing-brightgreen.svg)](https://docs.pytest.org/)
[![Architecture](https://img.shields.io/badge/Clean%20Architecture-Modular%20Pipeline-blueviolet.svg)](#system-architecture)

---
## Project explanatory video


https://github.com/user-attachments/assets/ad3c6dcb-90a9-492a-ab9d-a8c2fe67aa57







##  Executive Summary

This project delivers an end-to-end computer vision and explainable AI system designed to assess, classify, and triage building structural damage in the Gaza Strip. The pipeline categorizes building images into three actionable tiers:

1. **Safe (Low Hazard)**: Structurally sound; no load-bearing failure visible.
2. **Repair (Moderate Damage)**: Localized facade, wall, or cladding damage with intact primary structural frames (columns and slabs).
3. **Demolish (Critical Hazard)**: Severe structural failure, column buckling, or pancake slab collapse requiring immediate evacuation.

Built with a **Clean Architecture** approach, the repository features leak-free stratified data partitioning, cost-sensitive transfer learning, gradient-weighted class activation mapping (Grad-CAM), and a modern web dashboard.

---

##  System Architecture & Engineering Design

The repository follows clean software design principles with strict separation of concerns:

```
Building_Damage_Detection/
├── config/                  # Centralized configuration & environment loader
│   ├── __init__.py
│   └── settings.py
├── src/                     # Core business logic & model pipelines
│   ├── api/                 # Production FastAPI service & security middleware
│   │   ├── __init__.py
│   │   └── server.py
│   ├── models/              # Neural network definitions & checkpoint managers
│   │   ├── __init__.py
│   │   └── mobilenet.py
│   ├── pipeline/            # Image loading, color-space conversion & normalization
│   │   ├── __init__.py
│   │   └── preprocessor.py
│   └── xai/                 # Explainable AI (Grad-CAM gradient visualizers)
│       ├── __init__.py
│       └── gradcam.py
├── scripts/                 # Standalone, reproducible workflow scripts
│   ├── split_dataset.py     # Deterministic stratified partitioner
│   ├── train.py             # Two-stage cost-sensitive training pipeline
│   ├── evaluate.py          # Independent holdout test set evaluator
│   └── predict_cli.py       # Command-line prediction tool
├── tests/                   # Automated unit & integration test suite (PyTest)
│   ├── test_api.py
│   ├── test_gradcam.py
│   ├── test_model.py
│   └── test_preprocessor.py
├── static/                  # Modern web application assets
│   ├── index.html
│   ├── app.js
│   ├── style.css
│   └── samples/             # Benchmark validation images
├── weights/                 # Model checkpoints (.keras / .h5)
├── assets/                  # Visual documentation & confusion matrices
├── .env.example             # Safe environment variable template
├── .gitignore               # Multi-layer git ignore rules
└── requirements.txt         # Project dependencies
```

---

##  AI Methodology & Performance

### 1. Zero-Leakage Dataset Partitioning
The ground truth dataset (177 unique images) was partitioned into strictly disjoint sets using stratified sampling:
- **Training Set (70%)**: 123 images
- **Validation Set (15%)**: 27 images
- **Independent Test Set (15%)**: 27 images (completely unseen during training & hyperparameter tuning)

*Zero-leakage verified at the binary MD5 hash level across all partitions.*

### 2. Cost-Sensitive Transfer Learning
- **Backbone**: MobileNetV2 pretrained on ImageNet (lightweight, edge-deployable).
- **Class Weights**: Applied balanced class weights with an extra penalty factor on the `Demolish` class to reflect life-safety priorities.
- **Augmentation**: Rotation, shear, zoom, width/height shifts, and brightness modulation.
- **Two-Stage Training**:
  - *Stage 1*: Train the classification head while keeping the base feature extractor frozen.
  - *Stage 2*: Fine-tune the top 25 convolutional layers with a low learning rate (`2e-5`) and learning rate decay.

### 3. Holdout Test Evaluation Benchmark

| Metric | Safe | Repair | Demolish | Overall |
|:---|:---:|:---:|:---:|:---:|
| **Precision** | 77.8% | 66.7% | 83.3% | **76.8% (Weighted)** |
| **Recall** | 87.5% | 50.0% | **90.9%** | **77.8%** |
| **F1-Score** | 82.4% | 57.1% | **87.0%** | **76.8%** |
| **False-Safe Rate** | - | - | **0.0%** | **0 Critical Failures** |

> **Safety Achievement**: The model achieved a **90.91% recall on critical collapse cases (Demolish)** with **zero instances of a demolished building misclassified as safe**.

---

##  Explainable AI (Grad-CAM)

In structural engineering, black-box predictions are unacceptable. The system integrates Grad-CAM to compute visual attribution heatmaps:

1. Computes the gradient of the predicted score $y^c$ with respect to the last convolutional feature maps $A^k$.
2. Performs global average pooling over the gradients to compute neuron importance weights $\alpha_k^c$.
3. Combines maps with a rectified linear unit (ReLU) to isolate features positively contributing to the damage classification.

---

##  Quickstart Guide

### 1. Prerequisites & Environment Setup
Clone the repository and install requirements:
```bash
git clone https://github.com/your-username/Gaza-Building-Damage-AI.git
cd Gaza-Building-Damage-AI

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate   # On Linux/macOS: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables Configuration
Copy the sample environment configuration:
```bash
cp .env.example .env
```

### 3. Run the Web Application
Launch the production FastAPI server:
```bash
python api.py
```
Open your browser at **http://localhost:8000** to interact with the modern dashboard.

### 4. Run CLI Inference
```bash
python predict.py test_image.jpg
```

### 5. Execute Automated Tests
Run the complete PyTest test suite:
```bash
pytest tests -v
```

## License & Ethical Usage

This project is developed for humanitarian aid, civil structural engineering, and disaster management applications.

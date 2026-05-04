# Deep Learning-Based Vehicle Make & Model Classification

**Authors:** Krish Iyer, Aadit Chetan, Erica Phann  
**Affiliation:** University of North Carolina at Charlotte

---

## Overview

Fine-grained vehicle classification system using EfficientNet-B3 and the Stanford Cars Dataset (196 classes, 16,185 images). Achieves projected top-1 accuracy of 88–92% and top-5 accuracy >97%.

---

## Project Structure

```
vehicle_classifier/
├── data/
│   └── dataset.py          # Dataset class & DataLoaders (Krish Iyer)
├── models/
│   └── model.py            # EfficientNet-B3 & ResNet-50 backbones (Aadit Chetan)
├── utils/
│   ├── preprocess.py       # Preprocessing pipeline (Krish Iyer)
│   ├── train.py            # Training loop & scheduler (Aadit Chetan)
│   └── evaluate.py         # Metrics, Grad-CAM, confusion matrix (Erica Phann)
├── outputs/
│   ├── checkpoints/        # Saved model weights
│   ├── logs/               # TensorBoard logs
│   └── visualizations/     # Grad-CAM, confusion matrices
├── main.py                 # Entry point
├── config.py               # All hyperparameters & paths
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

### Download the Stanford Cars Dataset

1. Download from: https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset
2. Extract and place as:
```
vehicle_classifier/
└── stanford_cars/
    ├── cars_train/
    ├── cars_test/
    └── devkit/
        ├── cars_train_annos.mat
        ├── cars_test_annos_withlabels.mat
        └── cars_meta.mat
```

---

## Usage

### Train

```bash
# Train EfficientNet-B3 (primary model)
python main.py --mode train --backbone efficientnet

# Train ResNet-50 (baseline)
python main.py --mode train --backbone resnet50
```

### Evaluate

```bash
python main.py --mode evaluate --backbone efficientnet --checkpoint outputs/checkpoints/best_efficientnet.pth
```

### Predict on a single image

```bash
python main.py --mode predict --image path/to/car.jpg --checkpoint outputs/checkpoints/best_efficientnet.pth
```

---

## Configuration

All hyperparameters are in `config.py`. Key settings:

| Parameter | Value |
|---|---|
| Backbone | EfficientNet-B3 |
| Input size | 300×300 |
| Warmup epochs | 5 |
| Fine-tune epochs | 30 |
| Warmup LR | 1e-3 |
| Fine-tune LR | 1e-4 |
| Optimizer | Adam |
| Scheduler | Cosine Annealing |
| Early stopping patience | 5 |

---

## Individual Responsibilities

| Member | Role | Files |
|---|---|---|
| Krish Iyer | Data Preprocessing | `data/dataset.py`, `utils/preprocess.py` |
| Aadit Chetan | Model Architecture & Training | `models/model.py`, `utils/train.py` |
| Erica Phann | Evaluation & Reporting | `utils/evaluate.py` |

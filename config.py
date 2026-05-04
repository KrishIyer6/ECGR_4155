"""
config.py — Central configuration for all hyperparameters, paths, and settings.
All team members should import from here rather than hardcoding values.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple


# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "stanford_cars")
VMMRDB_DIR = os.path.join(ROOT_DIR, "vmmrdb")          # optional secondary dataset

OUTPUT_DIR    = os.path.join(ROOT_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR        = os.path.join(OUTPUT_DIR, "logs")
VIZ_DIR        = os.path.join(OUTPUT_DIR, "visualizations")

# Stanford Cars specific paths
TRAIN_IMG_DIR  = os.path.join(DATA_DIR, "cars_train")
TEST_IMG_DIR   = os.path.join(DATA_DIR, "cars_test")
TRAIN_ANNOS    = os.path.join(DATA_DIR, "devkit", "cars_train_annos.mat")
TEST_ANNOS     = os.path.join(DATA_DIR, "devkit", "cars_test_annos_withlabels.mat")
META_PATH      = os.path.join(DATA_DIR, "devkit", "cars_meta.mat")


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

NUM_CLASSES  = 196
TRAIN_SPLIT  = 0.80   # from official train partition
VAL_SPLIT    = 0.10
TEST_SPLIT   = 0.10   # remainder of official train partition used as held-out test


# ──────────────────────────────────────────────
# Preprocessing  (Krish Iyer)
# ──────────────────────────────────────────────

IMAGE_SIZE: Tuple[int, int] = (300, 300)   # EfficientNet-B3 native resolution

# ImageNet normalization constants
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD:  List[float] = [0.229, 0.224, 0.225]

USE_BBOX_CROP = True   # crop to bounding box before resize

# Augmentation parameters
AUG_HFLIP_PROB    = 0.5
AUG_ROTATION_DEG  = 15
AUG_COLOR_JITTER  = dict(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
AUG_RANDOM_ERASE  = dict(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3))


# ──────────────────────────────────────────────
# Model  (Aadit Chetan)
# ──────────────────────────────────────────────

BACKBONE = "efficientnet_b3"   # options: "efficientnet_b3" | "resnet50"
PRETRAINED = True
DROPOUT_RATE = 0.3             # dropout before final FC layer


# ──────────────────────────────────────────────
# Training  (Aadit Chetan)
# ──────────────────────────────────────────────

BATCH_SIZE    = 64
NUM_WORKERS   = 8
PIN_MEMORY    = True
RANDOM_SEED   = 42

# Two-stage training
WARMUP_EPOCHS     = 5
WARMUP_LR         = 1e-3
FINETUNE_EPOCHS   = 30
FINETUNE_LR       = 1e-4

# Scheduler
SCHEDULER         = "cosine"         # "cosine" | "step"
COSINE_T_MAX      = FINETUNE_EPOCHS  # period for CosineAnnealingLR
COSINE_ETA_MIN    = 1e-6

# Optimizer
OPTIMIZER         = "adam"
WEIGHT_DECAY      = 1e-4

# Early stopping
EARLY_STOP_PATIENCE = 5
EARLY_STOP_METRIC   = "val_loss"     # "val_loss" | "val_acc"


# ──────────────────────────────────────────────
# Evaluation  (Erica Phann)
# ──────────────────────────────────────────────

TOP_K_ACCURACIES = [1, 5]    # compute top-1 and top-5

# Grad-CAM
GRADCAM_LAYER    = "features"        # attribute name on EfficientNet; auto-resolved in code
NUM_GRADCAM_IMGS = 16                # images to visualize

# Confusion matrix — limit to top-N classes to keep plot readable
CONF_MATRIX_TOP_N = 40


# ──────────────────────────────────────────────
# Checkpoint naming
# ──────────────────────────────────────────────

def checkpoint_name(backbone: str, tag: str = "best") -> str:
    return os.path.join(CHECKPOINT_DIR, f"{tag}_{backbone}.pth")

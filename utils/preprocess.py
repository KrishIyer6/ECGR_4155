"""
utils/preprocess.py — Data preprocessing pipeline.
Responsible: Krish Iyer

Covers:
  - Bounding-box crop
  - Resize to EfficientNet-B3 input resolution (300×300)
  - ImageNet normalization
  - Training augmentation (flip, rotation, color jitter, random erasing)
  - Validation / test transforms (no augmentation)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

logger = logging.getLogger(__name__)


def crop_to_bbox(
    image: Image.Image,
    bbox: Optional[Tuple[int, int, int, int]],
    padding: float = 0.05,
) -> Image.Image:
    """
    Crop a PIL image to its annotated bounding box with optional padding.

    Args:
        image:   PIL.Image in any mode.
        bbox:    (x1, y1, x2, y2) in pixel coordinates, or None to skip crop.
        padding: Fractional padding added around the box (relative to box size).

    Returns:
        Cropped PIL.Image (or original image if bbox is None).
    """
    if bbox is None:
        return image

    x1, y1, x2, y2 = bbox
    w_img, h_img = image.size

    # Add proportional padding
    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w_img, x2 + pad_x)
    y2 = min(h_img, y2 + pad_y)

    return image.crop((x1, y1, x2, y2))



def build_train_transform() -> T.Compose:
    """
    Augmentation pipeline applied during training.

    Pipeline:
        1. Resize to IMAGE_SIZE
        2. Random horizontal flip
        3. Random rotation ± AUG_ROTATION_DEG degrees
        4. Color jitter (brightness / contrast / saturation / hue)
        5. Convert to tensor
        6. Normalize with ImageNet statistics
        7. Random erasing (after tensor conversion)
    """
    return T.Compose([
        T.Resize(cfg.IMAGE_SIZE),
        T.RandomHorizontalFlip(p=cfg.AUG_HFLIP_PROB),
        T.RandomRotation(degrees=cfg.AUG_ROTATION_DEG),
        T.ColorJitter(**cfg.AUG_COLOR_JITTER),
        T.ToTensor(),
        T.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD),
        T.RandomErasing(**cfg.AUG_RANDOM_ERASE),
    ])


def build_eval_transform() -> T.Compose:
    """
    Deterministic pipeline for validation and test sets.
    Only resize + normalize — no augmentation.
    """
    return T.Compose([
        T.Resize(cfg.IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=cfg.IMAGENET_MEAN, std=cfg.IMAGENET_STD),
    ])



def denormalize(tensor):
    """
    Reverse ImageNet normalization for visualization.

    Args:
        tensor: (C, H, W) float tensor, normalized.

    Returns:
        (C, H, W) float tensor in [0, 1].
    """
    import torch
    mean = torch.tensor(cfg.IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device)
    std  = torch.tensor(cfg.IMAGENET_STD,  dtype=tensor.dtype, device=tensor.device)
    return tensor * std[:, None, None] + mean[:, None, None]



def check_class_distribution(labels: list, class_names: list) -> dict:
    """
    Compute per-class sample counts and flag classes with fewer than
    ``min_samples`` examples.

    Args:
        labels:       List of integer class indices (0-based).
        class_names:  List of string class names indexed by class index.

    Returns:
        Dictionary mapping class name → sample count.
    """
    from collections import Counter
    counts = Counter(labels)
    dist = {class_names[k]: v for k, v in sorted(counts.items())}

    min_count = min(counts.values())
    max_count = max(counts.values())
    logger.info(
        "Class distribution — min: %d, max: %d, total classes: %d",
        min_count, max_count, len(counts),
    )

    rare_classes = [class_names[k] for k, v in counts.items() if v < 10]
    if rare_classes:
        logger.warning("Classes with <10 samples: %s", rare_classes)

    return dist


def verify_images(image_paths: list) -> Tuple[list, list]:
    """
    Attempt to open every image in ``image_paths`` and separate valid from
    corrupted files.

    Args:
        image_paths: List of file path strings.

    Returns:
        (valid_paths, corrupt_paths) — two lists of strings.
    """
    valid, corrupt = [], []
    for path in image_paths:
        try:
            with Image.open(path) as img:
                img.verify()
            valid.append(path)
        except Exception as exc:
            logger.warning("Corrupt image %s — %s", path, exc)
            corrupt.append(path)

    logger.info(
        "Image verification: %d valid, %d corrupted", len(valid), len(corrupt)
    )
    return valid, corrupt

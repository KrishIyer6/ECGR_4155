"""
data/dataset.py — Stanford Cars Dataset loader & DataLoader factory.
Responsible: Krish Iyer

Handles:
  - Parsing .mat annotation files from the Stanford Cars devkit
  - Optional bounding-box crop per image
  - 80 / 10 / 10 split of the official training partition
  - DataLoader construction with appropriate transforms
  - Optional VMMRdb loader for cross-dataset evaluation
"""

from __future__ import annotations

import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from utils.preprocess import (
    crop_to_bbox,
    build_train_transform,
    build_eval_transform,
    check_class_distribution,
    verify_images,
)

logger = logging.getLogger(__name__)


def load_stanford_annotations(
    annos_path: str,
    img_dir: str,
) -> Tuple[List[str], List[int], List[Optional[Tuple[int, int, int, int]]], List[str]]:
    """
    Parse a Stanford Cars .mat annotation file.

    Returns:
        image_paths : absolute path to each image
        labels      : 0-based class index for each image
        bboxes      : (x1, y1, x2, y2) or None if not present
        class_names : list of 196 class name strings (index = class id)
    """
    annos = sio.loadmat(annos_path, squeeze_me=True)
    meta  = sio.loadmat(cfg.META_PATH,  squeeze_me=True)

    class_names: List[str] = [str(n) for n in meta["class_names"].flat]

    image_paths: List[str] = []
    labels:      List[int] = []
    bboxes:      List[Optional[Tuple[int, int, int, int]]] = []

    for anno in annos["annotations"].flat:
        fname = str(anno["fname"])
        image_paths.append(os.path.join(img_dir, fname))

        # Labels in the .mat file are 1-indexed
        labels.append(int(anno["class"]) - 1)

        try:
            bbox = (
                int(anno["bbox_x1"]),
                int(anno["bbox_y1"]),
                int(anno["bbox_x2"]),
                int(anno["bbox_y2"]),
            )
        except (KeyError, TypeError):
            bbox = None
        bboxes.append(bbox)

    logger.info(
        "Loaded %d annotations from %s", len(image_paths), annos_path
    )
    return image_paths, labels, bboxes, class_names



class StanfordCarsDataset(Dataset):
    """
    PyTorch Dataset for the Stanford Cars benchmark.

    Args:
        image_paths : list of absolute image file paths.
        labels      : list of integer class indices (0-based).
        bboxes      : list of (x1, y1, x2, y2) tuples or None per image.
        transform   : torchvision transform to apply after optional crop.
        use_bbox    : whether to crop to bounding box before transform.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        bboxes: List[Optional[Tuple[int, int, int, int]]],
        transform: Optional[T.Compose] = None,
        use_bbox: bool = cfg.USE_BBOX_CROP,
    ) -> None:
        assert len(image_paths) == len(labels) == len(bboxes), \
            "Mismatched lengths for image_paths, labels, and bboxes."

        self.image_paths = image_paths
        self.labels      = labels
        self.bboxes      = bboxes
        self.transform   = transform
        self.use_bbox    = use_bbox

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        label    = self.labels[idx]
        bbox     = self.bboxes[idx] if self.use_bbox else None

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            logger.error("Failed to load image %s: %s", img_path, exc)
            # Return a black image so training doesn't crash on a single bad file
            image = Image.new("RGB", cfg.IMAGE_SIZE)

        if bbox is not None:
            image = crop_to_bbox(image, bbox)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def build_dataloaders(
    run_eda: bool = False,
    verify: bool = False,
) -> Dict[str, DataLoader]:
    """
    Build train, val, and test DataLoaders from the Stanford Cars dataset.

    The official *training* partition (8,144 images) is split 80/10/10 into
    train / val / test subsets using a fixed random seed for reproducibility.
    The official *test* partition is kept for cross-dataset evaluation only.

    Args:
        run_eda : if True, log class distribution statistics.
        verify  : if True, scan all images for corruption before training.

    Returns:
        dict with keys "train", "val", "test" mapping to DataLoader instances.
    """
    # Parse annotations
    image_paths, labels, bboxes, class_names = load_stanford_annotations(
        cfg.TRAIN_ANNOS, cfg.TRAIN_IMG_DIR
    )

    if verify:
        valid_paths, _ = verify_images(image_paths)
        valid_set = set(valid_paths)
        filtered = [
            (p, l, b) for p, l, b in zip(image_paths, labels, bboxes)
            if p in valid_set
        ]
        image_paths, labels, bboxes = map(list, zip(*filtered))

    if run_eda:
        check_class_distribution(labels, class_names)

    full_dataset = StanfordCarsDataset(
        image_paths, labels, bboxes,
        transform=build_eval_transform(),
    )

    # Reproducible split
    generator = torch.Generator().manual_seed(cfg.RANDOM_SEED)
    n_total   = len(full_dataset)
    n_train   = int(n_total * cfg.TRAIN_SPLIT)
    n_val     = int(n_total * cfg.VAL_SPLIT)
    n_test    = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    # Override transform on train split with augmentation
    # We subclass to apply a different transform per split.
    train_ds = _SplitDataset(full_dataset, train_ds.indices, build_train_transform())
    val_ds   = _SplitDataset(full_dataset, val_ds.indices,   build_eval_transform())
    test_ds  = _SplitDataset(full_dataset, test_ds.indices,  build_eval_transform())

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d", n_train, n_val, n_test
    )

    def _loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=cfg.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=cfg.NUM_WORKERS,
            pin_memory=cfg.PIN_MEMORY,
        )

    return {
        "train": _loader(train_ds, shuffle=True),
        "val":   _loader(val_ds,   shuffle=False),
        "test":  _loader(test_ds,  shuffle=False),
        "class_names": class_names,  # type: ignore[dict-item]
    }


def build_official_test_loader() -> Tuple[DataLoader, List[str]]:
    """
    Build a DataLoader from the *official* Stanford Cars test set with ground-truth
    labels (uses cars_test_annos_withlabels.mat).
    Used for final benchmark reporting.
    """
    image_paths, labels, bboxes, class_names = load_stanford_annotations(
        cfg.TEST_ANNOS, cfg.TEST_IMG_DIR
    )
    dataset = StanfordCarsDataset(
        image_paths, labels, bboxes,
        transform=build_eval_transform(),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    return loader, class_names


# ──────────────────────────────────────────────────────────────────────────────
# Internal helper: split-aware dataset wrapper
# ──────────────────────────────────────────────────────────────────────────────

class _SplitDataset(Dataset):
    """Wraps a base StanfordCarsDataset with a specific index subset and transform."""

    def __init__(
        self,
        base: StanfordCarsDataset,
        indices: List[int],
        transform: T.Compose,
    ) -> None:
        self.base      = base
        self.indices   = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        # Load raw image (before transform)
        img_path = self.base.image_paths[real_idx]
        label    = self.base.labels[real_idx]
        bbox     = self.base.bboxes[real_idx] if self.base.use_bbox else None

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", cfg.IMAGE_SIZE)

        if bbox is not None:
            image = crop_to_bbox(image, bbox)

        return self.transform(image), label

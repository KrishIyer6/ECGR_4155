"""
utils/train.py — Two-stage training loop with early stopping & TensorBoard logging.
Responsible: Aadit Chetan

Stage 1 (warm-up): train head only for WARMUP_EPOCHS at WARMUP_LR.
Stage 2 (fine-tune): unfreeze all layers, train for FINETUNE_EPOCHS at FINETUNE_LR with CosineAnnealingLR and early stopping.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from models.model import VehicleClassifier, freeze_backbone, unfreeze_backbone

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _set_seed(seed: int = cfg.RANDOM_SEED) -> None:
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)
    return device


# ──────────────────────────────────────────────────────────────────────────────
# Single epoch pass
# ──────────────────────────────────────────────────────────────────────────────

def _run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device:    torch.device,
    phase:     str,
    scaler:    torch.amp.GradScaler | None = None,
) -> Tuple[float, float]:
    """
    Run one epoch.

    Args:
        optimizer : pass None for eval phase (no gradient updates).

    Returns:
        (avg_loss, top1_accuracy) for the epoch.
    """
    is_train = optimizer is not None
    model.train(is_train)

    running_loss    = 0.0
    correct_top1    = 0
    total_samples   = 0

    pbar = tqdm(loader, desc=f"{phase}", leave=False, dynamic_ncols=True)

    with torch.set_grad_enabled(is_train):
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                logits = model(images)
                loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            # Metrics
            bs = images.size(0)
            running_loss  += loss.item() * bs
            total_samples += bs

            preds = logits.argmax(dim=1)
            correct_top1 += (preds == labels).sum().item()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / total_samples
    accuracy = correct_top1 / total_samples
    return avg_loss, accuracy


# ──────────────────────────────────────────────────────────────────────────────
# Early stopping
# ──────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stop training when the monitored metric has not improved for
    ``patience`` consecutive epochs.
    """

    def __init__(self, patience: int = cfg.EARLY_STOP_PATIENCE, mode: str = "min") -> None:
        self.patience  = patience
        self.mode      = mode
        self.counter   = 0
        self.best      = float("inf") if mode == "min" else float("-inf")
        self.triggered = False

    def step(self, value: float) -> bool:
        """Returns True if training should stop."""
        improved = (self.mode == "min" and value < self.best) or \
                   (self.mode == "max" and value > self.best)

        if improved:
            self.best    = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
                logger.info(
                    "Early stopping triggered — no improvement for %d epochs.", self.patience
                )
        return self.triggered


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model: nn.Module, path: str, metadata: dict | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {"state_dict": model.state_dict()}
    if metadata:
        state.update(metadata)
    torch.save(state, path)
    logger.info("Checkpoint saved -> %s", path)


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> dict:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["state_dict"])
    logger.info("Checkpoint loaded <- %s", path)
    return {k: v for k, v in state.items() if k != "state_dict"}


# ──────────────────────────────────────────────────────────────────────────────
# Main training function  (Aadit Chetan)
# ──────────────────────────────────────────────────────────────────────────────

def train(
    model:       VehicleClassifier,
    dataloaders: Dict[str, DataLoader],
    backbone_tag: str = "efficientnet",
) -> VehicleClassifier:
    """
    Full two-stage training procedure.

    Stage 1 — warm-up (head only):
        LR = WARMUP_LR  |  epochs = WARMUP_EPOCHS  |  no LR schedule

    Stage 2 — fine-tuning (all layers):
        LR = FINETUNE_LR  |  epochs = FINETUNE_EPOCHS
        CosineAnnealingLR  |  early stopping on val_loss

    Args:
        model        : VehicleClassifier instance.
        dataloaders  : dict with "train" and "val" DataLoader entries.
        backbone_tag : string tag used for checkpoint filenames.

    Returns:
        The model loaded with the best checkpoint weights.
    """
    _set_seed()
    device = _get_device()
    model  = model.to(device)

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(cfg.LOG_DIR, backbone_tag))

    criterion = nn.CrossEntropyLoss()

    best_ckpt = cfg.checkpoint_name(backbone_tag, tag="best")
    history   = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # ── Stage 1: warm-up ─────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Stage 1 — Warm-up (%d epochs, LR=%.0e)", cfg.WARMUP_EPOCHS, cfg.WARMUP_LR)
    logger.info("=" * 60)

    freeze_backbone(model)
    optimizer_warmup = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.WARMUP_LR,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    for epoch in range(1, cfg.WARMUP_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = _run_epoch(
            model, dataloaders["train"], criterion, optimizer_warmup, device, "Train", scaler
        )
        val_loss, val_acc = _run_epoch(
            model, dataloaders["val"], criterion, None, device, "Val", None
        )

        elapsed = time.time() - t0
        logger.info(
            "[Warmup %2d/%d] train_loss=%.4f  train_acc=%.4f  "
            "val_loss=%.4f  val_acc=%.4f  (%.1fs)",
            epoch, cfg.WARMUP_EPOCHS, train_loss, train_acc, val_loss, val_acc, elapsed,
        )

        writer.add_scalars("Loss",     {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc,  "val": val_acc},  epoch)

    save_checkpoint(model, cfg.checkpoint_name(backbone_tag, "warmup"),
                    {"stage": "warmup", "epochs": cfg.WARMUP_EPOCHS})

    # ── Stage 2: full fine-tuning ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(
        "Stage 2 — Fine-tuning (%d epochs, LR=%.0e)", cfg.FINETUNE_EPOCHS, cfg.FINETUNE_LR
    )
    logger.info("=" * 60)

    unfreeze_backbone(model)
    optimizer_ft = Adam(
        model.parameters(),
        lr=cfg.FINETUNE_LR,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(
        optimizer_ft,
        T_max=cfg.COSINE_T_MAX,
        eta_min=cfg.COSINE_ETA_MIN,
    )
    early_stop = EarlyStopping(patience=cfg.EARLY_STOP_PATIENCE, mode="min")

    best_val_loss = float("inf")

    for epoch in range(1, cfg.FINETUNE_EPOCHS + 1):
        global_epoch = cfg.WARMUP_EPOCHS + epoch
        t0 = time.time()

        train_loss, train_acc = _run_epoch(
            model, dataloaders["train"], criterion, optimizer_ft, device, "Train", scaler
        )
        val_loss, val_acc = _run_epoch(
            model, dataloaders["val"], criterion, None, device, "Val", None
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        elapsed    = time.time() - t0

        logger.info(
            "[FT %2d/%d] train_loss=%.4f  train_acc=%.4f  "
            "val_loss=%.4f  val_acc=%.4f  lr=%.2e  (%.1fs)",
            epoch, cfg.FINETUNE_EPOCHS,
            train_loss, train_acc, val_loss, val_acc, current_lr, elapsed,
        )

        writer.add_scalars("Loss",     {"train": train_loss, "val": val_loss}, global_epoch)
        writer.add_scalars("Accuracy", {"train": train_acc,  "val": val_acc},  global_epoch)
        writer.add_scalar("LR", current_lr, global_epoch)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, best_ckpt,
                {"epoch": global_epoch, "val_loss": val_loss, "val_acc": val_acc},
            )

        # Early stopping
        if early_stop.step(val_loss):
            logger.info("Stopping early at fine-tune epoch %d.", epoch)
            break

    writer.close()

    # Reload best weights
    logger.info("Reloading best checkpoint from %s", best_ckpt)
    load_checkpoint(model, best_ckpt, device)

    return model

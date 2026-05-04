"""
utils/evaluate.py — Evaluation suite: accuracy, confusion matrix, Grad-CAM, per-class report.
Responsible: Erica Phann

Provides:
  - compute_accuracy()        : top-1 and top-5 accuracy on any DataLoader
  - classification_report()   : per-class precision / recall / F1 (sklearn)
  - plot_confusion_matrix()   : heatmap for the top-N most common classes
  - gradcam_visualize()       : Grad-CAM saliency maps on sample images
  - evaluate()                : master function that runs the full suite
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")           # headless backend — safe for servers
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report as sk_classification_report

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg
from utils.preprocess import denormalize

logger = logging.getLogger(__name__)



@torch.no_grad()
def compute_accuracy(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
    top_k:  List[int] = cfg.TOP_K_ACCURACIES,
) -> dict:
    """
    Compute top-k accuracies over an entire DataLoader.

    Returns:
        dict mapping f"top{k}" → accuracy (float in [0, 1]).
    """
    model.eval()
    correct  = {k: 0 for k in top_k}
    total    = 0

    all_preds  = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        probs  = F.softmax(logits, dim=1)

        for k in top_k:
            _, top_preds = probs.topk(k, dim=1)
            correct[k] += top_preds.eq(labels.unsqueeze(1)).any(dim=1).sum().item()

        total      += images.size(0)
        all_preds  .extend(logits.argmax(dim=1).cpu().numpy())
        all_labels .extend(labels.cpu().numpy())

    accuracies = {f"top{k}": correct[k] / total for k in top_k}

    for k, acc in accuracies.items():
        logger.info("%s accuracy: %.4f (%.2f%%)", k, acc, acc * 100)

    return accuracies, np.array(all_preds), np.array(all_labels)



def classification_report(
    all_preds:   np.ndarray,
    all_labels:  np.ndarray,
    class_names: List[str],
    save_path:   Optional[str] = None,
) -> str:
    """
    Generate and optionally save a per-class classification report.

    Args:
        all_preds   : 1-D array of predicted class indices.
        all_labels  : 1-D array of ground-truth class indices.
        class_names : list mapping index → class name string.
        save_path   : if provided, write the report text to this file.

    Returns:
        Report string.
    """
    report = sk_classification_report(
        all_labels,
        all_preds,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    logger.info("\n%s", report)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            f.write(report)
        logger.info("Per-class report saved -> %s", save_path)

    return report


def plot_confusion_matrix(
    all_preds:   np.ndarray,
    all_labels:  np.ndarray,
    class_names: List[str],
    top_n:       int  = cfg.CONF_MATRIX_TOP_N,
    save_path:   str  = os.path.join(cfg.VIZ_DIR, "confusion_matrix.png"),
) -> None:
    """
    Plot a confusion matrix restricted to the ``top_n`` most frequent classes
    so the heatmap remains readable.

    Args:
        top_n     : number of most-frequent ground-truth classes to include.
        save_path : file path for the saved PNG.
    """
    from collections import Counter

    # Select top-N most common ground-truth classes
    counts      = Counter(all_labels.tolist())
    top_classes = [cls for cls, _ in counts.most_common(top_n)]
    top_classes.sort()

    mask = np.isin(all_labels, top_classes)
    y_true = all_labels[mask]
    y_pred = all_preds[mask]

    # Remap to a 0-based index within the subset
    idx_map = {orig: new for new, orig in enumerate(top_classes)}
    y_true_r = np.array([idx_map[v] for v in y_true])
    y_pred_r = np.array([idx_map.get(v, -1) for v in y_pred])

    n = len(top_classes)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true_r, y_pred_r):
        if 0 <= p < n:
            cm[t, p] += 1

    # Row-normalize to get recall per class
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm  = cm / row_sums

    labels = [class_names[c] for c in top_classes]
    # Shorten labels for readability
    short_labels = [lbl[:20] for lbl in labels]

    fig_size = max(14, n // 3)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm_norm, ax=ax,
        xticklabels=short_labels, yticklabels=short_labels,
        cmap="Blues", vmin=0, vmax=1,
        annot=(n <= 20), fmt=".2f",
        linewidths=0.5 if n <= 20 else 0,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Ground Truth", fontsize=12)
    ax.set_title(f"Confusion Matrix (top-{n} classes, row-normalized)", fontsize=14)
    plt.xticks(rotation=90, fontsize=6 if n > 20 else 8)
    plt.yticks(rotation=0,  fontsize=6 if n > 20 else 8)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved -> %s", save_path)



class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017).

    Usage:
        cam = GradCAM(model, target_layer)
        heatmap = cam(image_tensor, class_idx)   # (H, W) numpy array in [0, 1]
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model        = model
        self.target_layer = target_layer
        self._gradients:  Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def fwd_hook(_, __, output):
            self._activations = output.detach()

        def bwd_hook(_, __, grad_output):
            self._gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def __call__(
        self, image: torch.Tensor, class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Args:
            image     : (1, C, H, W) tensor on same device as model.
            class_idx : target class (None → argmax of logits).

        Returns:
            (H, W) numpy float32 heatmap in [0, 1].
        """
        self.model.eval()
        image.requires_grad_(True)

        logits = self.model(image)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Global average pool of gradients → weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam     = (weights * self._activations).sum(dim=1).squeeze()  # (h, w)
        cam     = torch.clamp(cam, min=0)

        # Upsample to input spatial size
        cam_np  = cam.cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)

        # Resize to image spatial dims
        import cv2
        h, w = image.shape[2], image.shape[3]
        cam_np = cv2.resize(cam_np, (w, h))
        return cam_np


def gradcam_visualize(
    model:       nn.Module,
    loader:      DataLoader,
    class_names: List[str],
    device:      torch.device,
    n_images:    int = cfg.NUM_GRADCAM_IMGS,
    save_dir:    str = cfg.VIZ_DIR,
) -> None:
    """
    Generate and save Grad-CAM overlays for ``n_images`` samples.

    Saves a single grid PNG to ``save_dir/gradcam_grid.png``.
    """
    try:
        import cv2
    except ImportError:
        logger.warning("opencv-python not installed — skipping Grad-CAM.")
        return

    from models.model import VehicleClassifier
    assert isinstance(model, VehicleClassifier), "Model must be a VehicleClassifier"

    target_layer = model.get_gradcam_target_layer()
    cam          = GradCAM(model, target_layer)

    model.eval()
    images_collected, labels_collected, preds_collected, cams_collected = [], [], [], []

    for imgs, lbls in loader:
        for i in range(imgs.size(0)):
            if len(images_collected) >= n_images:
                break
            img   = imgs[i:i+1].to(device)
            label = lbls[i].item()
            hmap  = cam(img)
            pred  = model(img).argmax(dim=1).item()

            images_collected.append(denormalize(imgs[i]).clamp(0, 1).permute(1, 2, 0).numpy())
            labels_collected.append(label)
            preds_collected.append(pred)
            cams_collected.append(hmap)

        if len(images_collected) >= n_images:
            break

    n     = len(images_collected)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(ncols * 6, nrows * 3))
    axes = np.array(axes).reshape(nrows, ncols * 2)

    for idx, (img_np, lbl, pred, hmap) in enumerate(
        zip(images_collected, labels_collected, preds_collected, cams_collected)
    ):
        row, col = divmod(idx, ncols)
        ax_img = axes[row, col * 2]
        ax_cam = axes[row, col * 2 + 1]

        ax_img.imshow(img_np)
        color = "green" if lbl == pred else "red"
        ax_img.set_title(
            f"GT: {class_names[lbl][:18]}\nPred: {class_names[pred][:18]}",
            fontsize=7, color=color,
        )
        ax_img.axis("off")

        overlay = img_np.copy()
        hmap_colored = plt.cm.jet(hmap)[:, :, :3]
        overlay = 0.6 * img_np + 0.4 * hmap_colored
        overlay = np.clip(overlay, 0, 1)
        ax_cam.imshow(overlay)
        ax_cam.set_title("Grad-CAM", fontsize=7)
        ax_cam.axis("off")

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col * 2].axis("off")
        axes[row, col * 2 + 1].axis("off")

    plt.suptitle("Grad-CAM Visualizations", fontsize=14, y=1.01)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "gradcam_grid.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Grad-CAM grid saved -> %s", out_path)



def evaluate(
    model:        nn.Module,
    test_loader:  DataLoader,
    class_names:  List[str],
    device:       torch.device,
    backbone_tag: str = "efficientnet",
    run_gradcam:  bool = True,
) -> dict:
    """
    Full evaluation suite.

    Computes:
      1. Top-1 and top-5 accuracy
      2. Per-class classification report (saved to outputs/visualizations/)
      3. Confusion matrix heatmap
      4. Grad-CAM visualizations (optional)

    Args:
        model        : trained VehicleClassifier.
        test_loader  : DataLoader for the evaluation set.
        class_names  : list of 196 class name strings.
        device       : torch.device.
        backbone_tag : string tag for output filenames.
        run_gradcam  : whether to produce Grad-CAM visualizations.

    Returns:
        dict with keys "top1", "top5" (float accuracies).
    """
    logger.info("=" * 60)
    logger.info("Running evaluation suite — backbone: %s", backbone_tag)
    logger.info("=" * 60)

    # 1. Accuracy
    accuracies, all_preds, all_labels = compute_accuracy(model, test_loader, device)

    # 2. Per-class report
    report_path = os.path.join(cfg.VIZ_DIR, f"classification_report_{backbone_tag}.txt")
    classification_report(all_preds, all_labels, class_names, save_path=report_path)

    # 3. Confusion matrix
    cm_path = os.path.join(cfg.VIZ_DIR, f"confusion_matrix_{backbone_tag}.png")
    plot_confusion_matrix(all_preds, all_labels, class_names, save_path=cm_path)

    # 4. Grad-CAM
    if run_gradcam:
        gradcam_visualize(
            model, test_loader, class_names, device,
            save_dir=os.path.join(cfg.VIZ_DIR, backbone_tag),
        )

    logger.info(
        "Evaluation complete — top-1: %.2f%%  top-5: %.2f%%",
        accuracies["top1"] * 100,
        accuracies["top5"] * 100,
    )
    return accuracies

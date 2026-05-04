"""
models/model.py — EfficientNet-B3 and ResNet-50 backbone definitions.
Responsible: Aadit Chetan

Provides:
  - VehicleClassifier : unified wrapper for either backbone
  - build_model()     : factory function
  - freeze_backbone() / unfreeze_backbone() : stage-1 / stage-2 helpers
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn as nn
import torchvision.models as tv_models

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

logger = logging.getLogger(__name__)

BackboneType = Literal["efficientnet_b3", "resnet50"]


# ──────────────────────────────────────────────────────────────────────────────
# Model wrapper
# ──────────────────────────────────────────────────────────────────────────────

class VehicleClassifier(nn.Module):
    """
    Fine-grained vehicle classifier built on a pretrained CNN backbone.

    The final classification head of the backbone is replaced with:
        Dropout → Linear(in_features, NUM_CLASSES)

    Args:
        backbone_name : "efficientnet_b3" or "resnet50"
        num_classes   : number of output classes (default: cfg.NUM_CLASSES = 196)
        pretrained    : load ImageNet weights (default: True)
        dropout_rate  : dropout probability before final FC layer
    """

    def __init__(
        self,
        backbone_name: BackboneType = "efficientnet_b3",
        num_classes:   int   = cfg.NUM_CLASSES,
        pretrained:    bool  = cfg.PRETRAINED,
        dropout_rate:  float = cfg.DROPOUT_RATE,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes   = num_classes

        weights_arg = "DEFAULT" if pretrained else None

        if backbone_name == "efficientnet_b3":
            base = tv_models.efficientnet_b3(weights=weights_arg)
            in_features = base.classifier[1].in_features
            base.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )
            self.model = base

        elif backbone_name == "resnet50":
            base = tv_models.resnet50(weights=weights_arg)
            in_features = base.fc.in_features
            base.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes),
            )
            self.model = base

        else:
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. "
                "Choose 'efficientnet_b3' or 'resnet50'."
            )

        logger.info(
            "Built %s — %d output classes, pretrained=%s, dropout=%.2f",
            backbone_name, num_classes, pretrained, dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ── Grad-CAM target layer ──────────────────────────────────────────────

    def get_gradcam_target_layer(self) -> nn.Module:
        """
        Return the last convolutional layer for Grad-CAM visualizations.
        """
        if self.backbone_name == "efficientnet_b3":
            # Last MBConv block inside features
            return self.model.features[-1]
        elif self.backbone_name == "resnet50":
            return self.model.layer4[-1]
        raise NotImplementedError(self.backbone_name)

    # ── Parameter counts ──────────────────────────────────────────────────

    def count_parameters(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ──────────────────────────────────────────────────────────────────────────────
# Freeze / unfreeze helpers  (Aadit Chetan)
# ──────────────────────────────────────────────────────────────────────────────

def freeze_backbone(model: VehicleClassifier) -> None:
    """
    Stage 1 — freeze all layers except the classification head.
    Only the head's parameters will receive gradient updates.
    """
    for param in model.model.parameters():
        param.requires_grad = False

    # Unfreeze the classification head
    if model.backbone_name == "efficientnet_b3":
        head = model.model.classifier
    else:
        head = model.model.fc

    for param in head.parameters():
        param.requires_grad = True

    n = model.count_parameters()
    logger.info(
        "Backbone frozen — trainable params: %d / %d",
        n["trainable"], n["total"],
    )


def unfreeze_backbone(model: VehicleClassifier) -> None:
    """
    Stage 2 — unfreeze all layers for full fine-tuning.
    """
    for param in model.model.parameters():
        param.requires_grad = True

    n = model.count_parameters()
    logger.info(
        "Backbone unfrozen — trainable params: %d / %d",
        n["trainable"], n["total"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_model(
    backbone: BackboneType = cfg.BACKBONE,
    device: str | torch.device = "cpu",
) -> VehicleClassifier:
    """
    Instantiate and move the model to the specified device.

    Args:
        backbone : backbone identifier string.
        device   : target device ("cpu", "cuda", or a torch.device).

    Returns:
        VehicleClassifier on the target device.
    """
    model = VehicleClassifier(backbone_name=backbone)
    model = model.to(device)
    logger.info("Model moved to device: %s", device)
    return model

"""
main.py — Command-line entry point for training, evaluation, and inference.

Usage:
    python main.py --mode train     --backbone efficientnet
    python main.py --mode train     --backbone resnet50
    python main.py --mode evaluate  --backbone efficientnet --checkpoint outputs/checkpoints/best_efficientnet.pth
    python main.py --mode predict   --backbone efficientnet --checkpoint outputs/checkpoints/best_efficientnet.pth --image car.jpg
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings

# Suppress TensorFlow / TensorBoard warnings before they are imported
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Suppress the PyTorch CUDA capability warning for newer architectures like the 5070 Ti
warnings.filterwarnings("ignore", category=UserWarning, module="torch\\.cuda", message=".*is not compatible with the current PyTorch installation.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*is not compatible with the current PyTorch installation.*")

import torch
from PIL import Image

import config as cfg
from data.dataset import build_dataloaders, build_official_test_loader
from models.model import build_model
from utils.train import train, load_checkpoint
from utils.evaluate import evaluate
from utils.preprocess import build_eval_transform


# ──────────────────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(cfg.OUTPUT_DIR, "run.log"), mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle Make & Model Classifier — UNCC Deep Learning Project"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "predict"],
        default="train",
        help="Operation mode.",
    )
    parser.add_argument(
        "--backbone",
        choices=["efficientnet", "resnet50"],
        default="efficientnet",
        help="Backbone architecture.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint .pth file (required for evaluate / predict).",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single image file (required for predict mode).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions to show in predict mode.",
    )
    parser.add_argument(
        "--no-gradcam",
        action="store_true",
        help="Skip Grad-CAM visualization during evaluation.",
    )
    parser.add_argument(
        "--run-eda",
        action="store_true",
        help="Run exploratory data analysis before training.",
    )
    parser.add_argument(
        "--verify-images",
        action="store_true",
        help="Scan dataset for corrupt images before training.",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Device helper
# ──────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Mode handlers
# ──────────────────────────────────────────────────────────────────────────────

def run_train(args: argparse.Namespace) -> None:
    backbone_map = {
        "efficientnet": "efficientnet_b3",
        "resnet50":     "resnet50",
    }
    backbone_cfg = backbone_map[args.backbone]

    logger.info("Building dataloaders …")
    loaders = build_dataloaders(run_eda=args.run_eda, verify=args.verify_images)

    device = get_device()
    logger.info("Building model: %s", backbone_cfg)
    model  = build_model(backbone=backbone_cfg, device=device)

    logger.info("Starting training …")
    model  = train(model, loaders, backbone_tag=args.backbone)

    logger.info("Training complete. Running evaluation on held-out test set …")
    run_evaluate(args, model=model, class_names=loaders["class_names"], test_loader=loaders["test"])


def run_evaluate(
    args:        argparse.Namespace,
    model=None,
    class_names: list | None = None,
    test_loader=None,
) -> None:
    backbone_map = {
        "efficientnet": "efficientnet_b3",
        "resnet50":     "resnet50",
    }
    backbone_cfg = backbone_map[args.backbone]
    device       = get_device()

    if model is None:
        if args.checkpoint is None:
            logger.error("--checkpoint is required for evaluate mode.")
            sys.exit(1)
        model = build_model(backbone=backbone_cfg, device=device)
        load_checkpoint(model, args.checkpoint, device)

    if test_loader is None or class_names is None:
        # Fall back to the deterministic 80/10/10 held-out test set since official labels are missing
        from data.dataset import build_dataloaders
        loaders = build_dataloaders()
        if test_loader is None:
            test_loader = loaders["test"]
        if class_names is None:
            class_names = loaders["class_names"]

    evaluate(
        model,
        test_loader,
        class_names,
        device,
        backbone_tag=args.backbone,
        run_gradcam=not args.no_gradcam,
    )


def run_predict(args: argparse.Namespace) -> None:
    if args.image is None:
        logger.error("--image is required for predict mode.")
        sys.exit(1)
    if args.checkpoint is None:
        logger.error("--checkpoint is required for predict mode.")
        sys.exit(1)

    backbone_map = {
        "efficientnet": "efficientnet_b3",
        "resnet50":     "resnet50",
    }
    backbone_cfg = backbone_map[args.backbone]
    device       = get_device()

    model = build_model(backbone=backbone_cfg, device=device)
    meta  = load_checkpoint(model, args.checkpoint, device)
    model.eval()

    # Load class names from the dataset devkit
    import scipy.io as sio
    meta_mat    = sio.loadmat(cfg.META_PATH, squeeze_me=True)
    class_names = [str(n) for n in meta_mat["class_names"].flat]

    transform = build_eval_transform()

    image = Image.open(args.image).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze()

    top_k = min(args.top_k, cfg.NUM_CLASSES)
    top_probs, top_idxs = probs.topk(top_k)

    print("\n" + "=" * 55)
    print(f"  Predictions for: {os.path.basename(args.image)}")
    print("=" * 55)
    for rank, (idx, prob) in enumerate(
        zip(top_idxs.cpu().tolist(), top_probs.cpu().tolist()), start=1
    ):
        print(f"  {rank:2d}. {class_names[idx]:<40s}  {prob * 100:6.2f}%")
    print("=" * 55 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Ensure output directories exist
    for d in [cfg.OUTPUT_DIR, cfg.CHECKPOINT_DIR, cfg.LOG_DIR, cfg.VIZ_DIR]:
        os.makedirs(d, exist_ok=True)

    args = parse_args()
    logger.info("Mode: %s | Backbone: %s", args.mode, args.backbone)

    if args.mode == "train":
        run_train(args)
    elif args.mode == "evaluate":
        run_evaluate(args)
    elif args.mode == "predict":
        run_predict(args)

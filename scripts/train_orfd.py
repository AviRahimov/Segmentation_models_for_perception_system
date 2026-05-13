"""Fine-tune a semantic segmentation model on the ORFD binary freespace dataset.

Supported models
----------------
  segformer-b2   Start from nvidia/segformer-b2-finetuned-ade-512-512
  segformer-b4   Start from nvidia/segformer-b4-finetuned-ade-512-512
  ddrnet         Start from weights/ddrnet_category_512.pth (replace final head)

Usage
-----
    python scripts/train_orfd.py --model segformer-b2
    python scripts/train_orfd.py --model ddrnet --lr 1e-4 --epochs 80

Output
------
Best checkpoint (by validation mIoU) → weights/orfd/<model_name>/best.pth
Last checkpoint                       → weights/orfd/<model_name>/last.pth
Training log                          → weights/orfd/<model_name>/train_log.json
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

# Make sure the src package is importable when running as a script.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from perception.datasets.orfd_torch import ORFDDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_orfd")

IGNORE_INDEX = 255    # augmentation-edge padding — excluded from loss and IoU
NUM_CLASSES = 3       # 0 = non_traversable, 1 = traversable, 2 = sky
N_WARMUP = 5          # epochs of linear LR warmup before cosine decay


def seed_everything(seed: int) -> None:
    """Seed all RNG sources for fully reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker a unique but deterministic NumPy seed."""
    np.random.seed(torch.initial_seed() % 2**32)


# --------------------------------------------------------------------------- #
# Model builders                                                               #
# --------------------------------------------------------------------------- #


def build_segformer(variant: str, device: str, fp16: bool) -> tuple[nn.Module, object]:
    """Return (model, processor) for SegFormer fine-tuning.

    Loads the ADE20K-pretrained backbone and replaces the decode head with a
    fresh NUM_CLASSES-class head (``ignore_mismatched_sizes=True``).
    """
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    hf_ids = {
        "segformer-b2": "nvidia/segformer-b2-finetuned-ade-512-512",
        "segformer-b4": "nvidia/segformer-b4-finetuned-ade-512-512",
    }
    hf_id = hf_ids[variant]
    logger.info("Loading SegFormer base weights from %s ...", hf_id)
    processor = SegformerImageProcessor.from_pretrained(hf_id)
    model = SegformerForSemanticSegmentation.from_pretrained(
        hf_id,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    model = model.to(device)
    return model, processor


def build_ddrnet(device: str, fp16: bool) -> nn.Module:
    """Return DDRNet-39 with 12-class backbone weights + fresh 2-class head.

    Strategy:
        1. Build the full GOOSE-12 architecture (num_classes=12).
        2. Strict-load the GOOSE-12 checkpoint (all 501 entries match).
        3. Replace ``model.final_layer`` with a fresh ``SegmentHead``
           (in_planes=256, inter_planes=256, out_planes=2, scale_factor=8).
        4. Freeze the backbone for the first ``DDRNET_FREEZE_EPOCHS`` epochs
           to let the new head stabilise (done in the training loop).
    """
    import torch.nn as nn
    sys.path.insert(0, str(_ROOT / "src"))
    from perception.models.semantic._vendored.ddrnet39_goose import (
        DDRNet, SegmentHead, ddrnet_39_goose,
    )

    weights_path = _ROOT / "weights" / "ddrnet_category_512.pth"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"DDRNet checkpoint not found at {weights_path}. "
            "Run scripts/download_datasets.py to fetch it."
        )

    logger.info("Loading DDRNet-39 GOOSE-12 weights from %s ...", weights_path)
    model = ddrnet_39_goose(num_classes=12, use_aux_heads=False)
    ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    state_dict = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)

    # Replace the final classification head with a 2-class head.
    # in_planes = highres_planes * layer5_bottleneck_expansion = 128 * 2 = 256
    model.final_layer = SegmentHead(
        in_planes=256,
        inter_planes=256,
        out_planes=NUM_CLASSES,
        scale_factor=8,
    )
    logger.info("DDRNet final_layer replaced with 2-class head.")

    model = model.to(device)
    return model


# --------------------------------------------------------------------------- #
# Loss / metrics                                                               #
# --------------------------------------------------------------------------- #


def compute_miou(
    preds: torch.Tensor,  # (N, H, W) int64 predicted class indices
    labels: torch.Tensor, # (N, H, W) int64 ground truth
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[float, list[float]]:
    """Return (mean_iou, [iou_per_class])."""
    valid = labels != ignore_index
    ious = []
    for c in range(num_classes):
        pred_c  = (preds  == c) & valid
        label_c = (labels == c) & valid
        inter = (pred_c & label_c).sum().item()
        union = (pred_c | label_c).sum().item()
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(inter / union)
    valid_ious = [v for v in ious if not (isinstance(v, float) and v != v)]
    mean = float(np.mean(valid_ious)) if valid_ious else 0.0
    return mean, ious


# --------------------------------------------------------------------------- #
# Forward pass helpers                                                         #
# --------------------------------------------------------------------------- #


def segformer_forward(
    model: nn.Module,
    processor: object,
    images_chw: torch.Tensor,  # (B, 3, H, W) float32, ImageNet-normalised
    device: str,
    fp16: bool,
) -> torch.Tensor:
    """Return (B, 2, H, W) upsampled logits from SegFormer."""
    b, _, h, w = images_chw.shape

    # Re-encode as the HF processor expects: list of HWC uint8 RGB ndarrays.
    # We already have normalised tensors, so we reverse normalisation to get
    # the raw pixel values the processor can accept.
    _MEAN = torch.tensor([0.485, 0.456, 0.406], device=images_chw.device).view(1, 3, 1, 1)
    _STD  = torch.tensor([0.229, 0.224, 0.225], device=images_chw.device).view(1, 3, 1, 1)
    rgb_01 = images_chw * _STD + _MEAN
    rgb_u8 = (rgb_01.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    inputs = processor(images=list(rgb_u8), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits  # (B, 2, H/4, W/4)
    logits = torch.nn.functional.interpolate(
        logits, size=(h, w), mode="bilinear", align_corners=False,
    )
    return logits  # (B, 2, H, W)


def ddrnet_forward(
    model: nn.Module,
    images_chw: torch.Tensor,  # (B, 3, H, W) float32, ImageNet-normalised
    device: str,
    fp16: bool,
) -> torch.Tensor:
    """Return (B, 2, H, W) upsampled logits from DDRNet."""
    b, _, h, w = images_chw.shape
    x = images_chw.to(device)

    # DDRNet's skip connections compute width_output = W // 8 and expect the
    # backbone feature maps to have exactly that width. When W is not divisible
    # by 8 the floor-division mismatches the actual conv-stride output by 1.
    # Pad to the nearest multiple of 8; crop back via interpolate at the end.
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h > 0 or pad_w > 0:
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    logits = model(x)
    # Upsample to the exact original (H, W), which also removes any padding.
    logits = torch.nn.functional.interpolate(
        logits, size=(h, w), mode="bilinear", align_corners=False,
    )
    return logits


# --------------------------------------------------------------------------- #
# Training loop                                                                #
# --------------------------------------------------------------------------- #

DDRNET_FREEZE_EPOCHS = 10  # freeze backbone for first N epochs


def train_one_epoch(
    model: nn.Module,
    processor,
    loader: DataLoader,
    optimizer,
    criterion: nn.Module,
    device: str,
    fp16: bool,
    model_name: str,
) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        # bfloat16 has the same exponent range as float32, so no GradScaler is needed.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=fp16):
            if model_name.startswith("segformer"):
                logits = segformer_forward(model, processor, images, device, fp16=False)
            else:
                logits = ddrnet_forward(model, images, device, fp16=False)
            loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    processor,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    fp16: bool,
    model_name: str,
) -> tuple[float, float]:
    """Return (val_loss, mean_iou)."""
    model.eval()
    total_loss = 0.0
    all_preds:  list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=fp16):
            if model_name.startswith("segformer"):
                logits = segformer_forward(model, processor, images, device, fp16=False)
            else:
                logits = ddrnet_forward(model, images, device, fp16=False)

            loss = criterion(logits, labels)
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    preds_cat  = torch.cat(all_preds,  dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    miou, _ = compute_miou(preds_cat, labels_cat)
    return total_loss / len(loader), miou


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune segmentation model on ORFD")
    p.add_argument("--model",   default="segformer-b2",
                   choices=["segformer-b2", "segformer-b4", "ddrnet"],
                   help="Model to train")
    p.add_argument("--data",    default=str(_ROOT / "datasets" / "orfd"),
                   help="Path to ORFD root (contains training/, validation/)")
    p.add_argument("--epochs",  type=int, default=100)
    p.add_argument("--batch",   type=int, default=8)
    p.add_argument("--lr",      type=float, default=6e-5)
    p.add_argument("--wd",      type=float, default=0.01,  help="AdamW weight decay")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience",type=int, default=10,
                   help="Early-stopping patience (epochs without val mIoU improvement)")
    p.add_argument("--no-fp16", action="store_true", help="Disable mixed-precision training")
    p.add_argument("--seed",    type=int, default=None,
                   help="Global RNG seed for reproducibility (omit for non-deterministic)")
    p.add_argument("--out",     default=None,
                   help="Output directory (default: weights/orfd/<model>/)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16   = not args.no_fp16 and device == "cuda"

    if args.seed is not None:
        seed_everything(args.seed)
        logger.info("Global seed: %d", args.seed)

    out_dir = Path(args.out) if args.out else _ROOT / "weights" / "orfd" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)
    logger.info("Device: %s  fp16: %s", device, fp16)

    # --- Datasets ---
    train_ds = ORFDDataset(args.data, split="training",   augment=True)
    val_ds   = ORFDDataset(args.data, split="validation", augment=False)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        worker_init_fn=_worker_init_fn if args.seed is not None else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    logger.info("Train: %d samples  Val: %d samples", len(train_ds), len(val_ds))

    # --- Model ---
    processor = None
    if args.model.startswith("segformer"):
        model, processor = build_segformer(args.model, device, fp16)
    else:
        model = build_ddrnet(device, fp16)

    # --- Optimiser ---
    # Use different LR for backbone vs head (fine-tuning heuristic).
    if args.model.startswith("segformer"):
        head_params = list(model.decode_head.parameters())
        head_ids = {id(p) for p in head_params}
        backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
        param_groups = [
            {"params": backbone_params, "lr": args.lr * 0.1},
            {"params": head_params,      "lr": args.lr},
        ]
    else:
        # DDRNet: freeze backbone params for first DDRNET_FREEZE_EPOCHS epochs.
        final_layer_ids = {id(p) for p in model.final_layer.parameters()}
        backbone_params = [p for p in model.parameters() if id(p) not in final_layer_ids]
        head_params     = list(model.final_layer.parameters())
        param_groups = [
            {"params": backbone_params, "lr": args.lr * 0.0},  # frozen initially
            {"params": head_params,      "lr": args.lr},
        ]

    optimizer = AdamW(param_groups, weight_decay=args.wd)
    warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=N_WARMUP)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - N_WARMUP), eta_min=1e-7)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched],
                             milestones=[N_WARMUP])
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, label_smoothing=0.1)

    # --- Training loop ---
    best_miou     = 0.0
    patience_left = args.patience
    log_entries: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        # Unfreeze DDRNet backbone after warm-up period.
        if args.model == "ddrnet" and epoch == DDRNET_FREEZE_EPOCHS + 1:
            optimizer.param_groups[0]["lr"] = args.lr * 0.1
            logger.info("Epoch %d: DDRNet backbone unfrozen.", epoch)

        train_loss = train_one_epoch(
            model, processor, train_loader, optimizer, criterion,
            device, fp16, args.model,
        )
        val_loss, val_miou = evaluate(
            model, processor, val_loader, criterion, device, fp16, args.model,
        )
        scheduler.step()

        elapsed = time.perf_counter() - t0
        logger.info(
            "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  val_mIoU=%.4f  (%.1fs)",
            epoch, args.epochs, train_loss, val_loss, val_miou, elapsed,
        )

        entry = dict(epoch=epoch, train_loss=train_loss,
                     val_loss=val_loss, val_miou=val_miou)
        log_entries.append(entry)
        (out_dir / "train_log.json").write_text(json.dumps(log_entries, indent=2))

        # Save last checkpoint.
        _save_checkpoint(model, out_dir / "last.pth", epoch, val_miou)

        # Save best checkpoint.
        if val_miou > best_miou:
            best_miou = val_miou
            patience_left = args.patience
            _save_checkpoint(model, out_dir / "best.pth", epoch, val_miou)
            logger.info("  → new best mIoU %.4f saved.", best_miou)
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("Early stopping: no improvement for %d epochs.", args.patience)
                break

    logger.info("Training done. Best val mIoU: %.4f", best_miou)
    logger.info("Best checkpoint: %s", out_dir / "best.pth")


def _save_checkpoint(model: nn.Module, path: Path, epoch: int, miou: float) -> None:
    torch.save({"net": model.state_dict(), "epoch": epoch, "miou": miou}, str(path))


if __name__ == "__main__":
    main()

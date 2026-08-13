"""Shared ORFD SegFormer training/eval helpers.

Split out of train_orfd.py (the training entrypoint) because
resolution_sweep.py, benchmark_jetson.py, train_qat.py, train_sparse.py, and
compare_models.py all need the same loss/metric/forward-pass/model-builder
functions and were previously importing train_orfd.py sideways via a manual
sys.path insert (`import train_orfd as _t`) -- reusing a script that also
defines its own argparse/main() as a library. Both train_orfd.py and the 5
importers now depend on this module directly instead.
"""
from __future__ import annotations

import logging
import random
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger("train_orfd")

IGNORE_INDEX = 255    # augmentation-edge padding — excluded from loss and IoU
NUM_CLASSES  = 3      # 0 = non_traversable, 1 = traversable, 2 = sky


def seed_everything(seed: int) -> None:
    """Seed all RNG sources for fully reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_segformer(variant: str, device: str, fp16: bool) -> tuple[nn.Module, object]:
    """Return (model, processor) for SegFormer fine-tuning.

    Loads the ADE20K-pretrained backbone and replaces the decode head with a
    fresh NUM_CLASSES-class head (``ignore_mismatched_sizes=True``).
    """
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    hf_ids = {
        "segformer-b0": "nvidia/segformer-b0-finetuned-ade-512-512",
        "segformer-b1": "nvidia/segformer-b1-finetuned-ade-512-512",
        "segformer-b2": "nvidia/segformer-b2-finetuned-ade-512-512",
        "segformer-b3": "nvidia/segformer-b3-finetuned-ade-512-512",
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


def _dice_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
    dice_weight: float = 0.5,
    label_smoothing: float = 0.0,
    class_weights: torch.Tensor | None = None,
    asym_alpha: float | None = None,
    asym_beta: float | None = None,
    asym_class: int = 1,
) -> torch.Tensor:
    """Dice + CrossEntropy combined loss.

    CE stabilises gradients; Dice directly optimises IoU and handles class
    imbalance.  Ignore-index pixels are masked out of the Dice computation.

    class_weights (optional, shape (num_classes,)): up-weights rare classes
    in both terms -- CE via F.cross_entropy's own `weight=`, Dice via a
    weighted mean over the per-class dice scores instead of a plain mean.
    Added for the Gaza-domain fine-grained classes (animal/vehicle/rubble
    are rare in a 241-image set, the same long-tail problem GOOSE has at
    64 classes, just smaller scale) -- unused (None) leaves ORFD's existing
    3-class training numerically unchanged.

    asym_alpha/asym_beta (optional): replace plain Dice with an asymmetric
    Tversky term for `asym_class` only (every other class keeps the exact
    existing Dice formula). Standard Dice for a class is mathematically
    Tversky with alpha=beta=0.5 (TP/(TP+0.5*FP+0.5*FN)); asym_alpha>asym_beta
    penalises false positives on that class more than false negatives --
    added to directly counteract a real, measured failure mode where a
    Gaza-domain fine-tune over-predicts `traversable` (class 1) on large
    rocks/hillsides in non-Gaza footage (a false-positive-heavy error).
    Both args must be given together; leaving them None reproduces today's
    plain-Dice behaviour exactly (the branch below is never entered).
    """
    import torch.nn.functional as F

    ce = F.cross_entropy(logits, labels,
                         ignore_index=ignore_index,
                         label_smoothing=label_smoothing,
                         weight=class_weights)

    # Build valid-pixel mask (ignore 255 pixels).
    valid = (labels != ignore_index)  # (B, H, W) bool
    if valid.sum() == 0:
        return ce

    # Clamp labels so one_hot doesn't blow up on 255.
    labels_safe = labels.clone()
    labels_safe[~valid] = 0

    probs = torch.softmax(logits.float(), dim=1)              # (B, C, H, W)
    labels_oh = torch.zeros_like(probs)                       # (B, C, H, W)
    labels_oh.scatter_(1, labels_safe.unsqueeze(1), 1.0)

    mask = valid.unsqueeze(1).float()                         # (B, 1, H, W)
    probs    = probs    * mask
    labels_oh = labels_oh * mask

    dims = (0, 2, 3)  # average over batch + spatial
    intersection = (probs * labels_oh).sum(dim=dims)
    union        = probs.sum(dim=dims) + labels_oh.sum(dim=dims)
    dice = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)  # (C,) per-class

    if asym_alpha is not None and asym_beta is not None:
        eps = 1e-6
        c = asym_class
        tp_c = intersection[c]
        fp_c = (probs[:, c] * (1.0 - labels_oh[:, c])).sum()
        fn_c = ((1.0 - probs[:, c]) * labels_oh[:, c]).sum()
        tversky_c = (tp_c + eps) / (tp_c + asym_alpha * fp_c + asym_beta * fn_c + eps)
        dice = dice.clone()
        dice[c] = 1.0 - tversky_c

    if class_weights is not None:
        w = class_weights.to(dice.device, dice.dtype)
        dice = (dice * w).sum() / w.sum()
    else:
        dice = dice.mean()

    return ce + dice_weight * dice


def compute_miou(
    preds: torch.Tensor,   # (N, H, W) int64 predicted class indices
    labels: torch.Tensor,  # (N, H, W) int64 ground truth
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


def segformer_forward(
    model: nn.Module,
    processor: object,
    images_chw: torch.Tensor,  # (B, 3, H, W) float32, ImageNet-normalised
    device: str,
    fp16: bool,
) -> torch.Tensor:
    """Return (B, NUM_CLASSES, H, W) upsampled logits from SegFormer."""
    b, _, h, w = images_chw.shape

    # Re-encode as the HF processor expects: list of HWC uint8 RGB ndarrays.
    _MEAN = torch.tensor([0.485, 0.456, 0.406], device=images_chw.device).view(1, 3, 1, 1)
    _STD  = torch.tensor([0.229, 0.224, 0.225], device=images_chw.device).view(1, 3, 1, 1)
    rgb_01 = images_chw * _STD + _MEAN
    rgb_u8 = (rgb_01.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    inputs = processor(images=list(rgb_u8), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits  # (B, C, H/4, W/4)
    logits = torch.nn.functional.interpolate(
        logits, size=(h, w), mode="bilinear", align_corners=False,
    )
    return logits  # (B, C, H, W)


def train_one_epoch(
    model: nn.Module,
    processor,
    loader: DataLoader,
    optimizer,
    criterion: nn.Module,
    device: str,
    fp16: bool,
    clip_norm: float,
    forward_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor] | None = None,
) -> float:
    """``forward_fn(model, images_chw) -> logits`` lets non-SegFormer architectures
    (AurigaNet, Mask2Former, UPerNet, DINOv2, ...) reuse this loop unchanged —
    default (None) preserves the exact original SegFormer/``processor`` behavior
    for every existing caller (train_orfd.py, train_qat.py, train_sparse.py, ...).
    """
    _forward = forward_fn or (lambda m, imgs: segformer_forward(m, processor, imgs, device, fp16=False))
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=fp16):
            logits = _forward(model, images)
            loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
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
    forward_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor] | None = None,
) -> tuple[float, float]:
    """Return (val_loss, mean_iou). See train_one_epoch for ``forward_fn``."""
    _forward = forward_fn or (lambda m, imgs: segformer_forward(m, processor, imgs, device, fp16=False))
    model.eval()
    total_loss = 0.0
    all_preds:  list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for images, labels in tqdm(loader, desc="val  ", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=fp16):
            logits = _forward(model, images)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    preds_cat  = torch.cat(all_preds,  dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    miou, _ = compute_miou(preds_cat, labels_cat)
    return total_loss / len(loader), miou

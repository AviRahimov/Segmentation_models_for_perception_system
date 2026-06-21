"""Fine-tune a semantic segmentation model on the ORFD binary freespace dataset.

Supported models
----------------
  segformer-b0   Start from nvidia/segformer-b0-finetuned-ade-512-512
  segformer-b1   Start from nvidia/segformer-b1-finetuned-ade-512-512
  segformer-b2   Start from nvidia/segformer-b2-finetuned-ade-512-512
  segformer-b4   Start from nvidia/segformer-b4-finetuned-ade-512-512
  auriganet      Start from random init (no public BDD100K weights); train from scratch

Usage
-----
    # Use config/train.yaml (default):
    python scripts/train_orfd.py

    # Override individual settings:
    python scripts/train_orfd.py --lr 1e-4 --epochs 50

    # Resume an interrupted run:
    python scripts/train_orfd.py --resume weights/orfd/segformer-b2/last.pth

Output
------
Best checkpoint (by validation mIoU) → weights/orfd/<model_name>/best.pth
Last checkpoint (full state)         → weights/orfd/<model_name>/last.pth
Training log                         → weights/orfd/<model_name>/train_log.json
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

# Make sure the src package is importable when running as a script.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from perception.datasets.orfd_torch import ORFDDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_orfd")

IGNORE_INDEX = 255    # augmentation-edge padding — excluded from loss and IoU
NUM_CLASSES  = 3      # 0 = non_traversable, 1 = traversable, 2 = sky


def _dice_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
    dice_weight: float = 0.5,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Dice + CrossEntropy combined loss.

    CE stabilises gradients; Dice directly optimises IoU and handles class
    imbalance.  Ignore-index pixels are masked out of the Dice computation.
    """
    import torch.nn.functional as F

    ce = F.cross_entropy(logits, labels,
                         ignore_index=ignore_index,
                         label_smoothing=label_smoothing)

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
    dice = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
    dice = dice.mean()

    return ce + dice_weight * dice


# --------------------------------------------------------------------------- #
# Config loader                                                                #
# --------------------------------------------------------------------------- #


def load_train_config(path: str) -> dict[str, Any]:
    """Read train.yaml and return a flat dict of default values for argparse."""
    import yaml
    p = Path(path)
    if not p.exists():
        return {}
    with p.open() as f:
        raw = yaml.safe_load(f) or {}
    ds = raw.get("dataset",  {}) or {}
    m  = raw.get("model",    {}) or {}
    tr = raw.get("training", {}) or {}
    return {
        "data":               ds.get("root",          "datasets/orfd"),
        "train_split":        ds.get("train_split",   "training"),
        "val_split":          ds.get("val_split",     "validation"),
        "model":              m.get("name",            "segformer-b2"),
        "out":                m.get("out_dir")         or None,
        "epochs":             tr.get("epochs",         100),
        "batch":              tr.get("batch_size",     8),
        "lr":                 tr.get("lr",             6e-5),
        "wd":                 tr.get("weight_decay",   0.01),
        "workers":            tr.get("workers",        4),
        "patience":           tr.get("patience",       10),
        "fp16":               tr.get("fp16",           True),
        "seed":               tr.get("seed",           None),
        "resume_from":        tr.get("resume_from",    "") or "",
        "n_warmup":           tr.get("n_warmup_epochs",       5),
        "clip_norm":          tr.get("grad_clip_norm",        1.0),
        "label_smoothing":    tr.get("label_smoothing",       0.1),
    }


# --------------------------------------------------------------------------- #
# RNG helpers                                                                  #
# --------------------------------------------------------------------------- #


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


def build_auriganet(device: str, fp16: bool, weights: str = "") -> tuple[nn.Module, None]:
    """Return (model, None) for AurigaNet fine-tuning on ORFD."""
    sys.path.insert(0, str(_ROOT / "src"))
    from perception.models.semantic._vendored.auriganet import AurigaNetArch

    logger.info("Building AurigaNet (num_seg_classes=3, with_detection=False) ...")
    model = AurigaNetArch(num_seg_classes=NUM_CLASSES, with_detection=False)

    if weights and Path(weights).is_file():
        ckpt = torch.load(weights, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("net", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("AurigaNet resume: %d missing keys", len(missing))
        logger.info("AurigaNet loaded from %s", weights)

    model = model.to(device)
    return model, None


def auriganet_forward(
    model: nn.Module,
    images_chw: torch.Tensor,  # (B, 3, H, W) float32, ImageNet-normalised
    device: str,
    fp16: bool,
) -> torch.Tensor:
    """Return (B, NUM_CLASSES, H, W) upsampled logits from AurigaNet."""
    b, _, h, w = images_chw.shape
    x = images_chw.to(device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=fp16):
        seg_logits, _embed, _det = model(x)  # (B, C, H/4, W/4)

    seg_logits = torch.nn.functional.interpolate(
        seg_logits.float(), size=(h, w), mode="bilinear", align_corners=False,
    )
    return seg_logits  # (B, C, H, W)


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
    # model_encoder = model.segformer.segformer
    model = model.to(device)
    return model, processor



# --------------------------------------------------------------------------- #
# Loss / metrics                                                               #
# --------------------------------------------------------------------------- #


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



# --------------------------------------------------------------------------- #
# Training loop                                                                #
# --------------------------------------------------------------------------- #


def train_one_epoch(
    model: nn.Module,
    processor,
    loader: DataLoader,
    optimizer,
    criterion: nn.Module,
    device: str,
    fp16: bool,
    clip_norm: float,
    is_auriganet: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=fp16):
            if is_auriganet:
                logits = auriganet_forward(model, images, device, fp16=False)
            else:
                logits = segformer_forward(model, processor, images, device, fp16=False)
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
    is_auriganet: bool = False,
) -> tuple[float, float]:
    """Return (val_loss, mean_iou)."""
    model.eval()
    total_loss = 0.0
    all_preds:  list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for images, labels in tqdm(loader, desc="val  ", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=fp16):
            if is_auriganet:
                logits = auriganet_forward(model, images, device, fp16=False)
            else:
                logits = segformer_forward(model, processor, images, device, fp16=False)
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
    # Two-pass: first extract --config, then load its values as argparse defaults.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(_ROOT / "config" / "train.yaml"))
    known, _ = pre.parse_known_args()
    cfg = load_train_config(known.config)

    p = argparse.ArgumentParser(description="Fine-tune segmentation model on ORFD/custom dataset")
    p.add_argument("--config", default=str(_ROOT / "config" / "train.yaml"),
                   help="Path to train.yaml config file")
    p.add_argument("--model",   default=cfg.get("model", "segformer-b2"),
                   choices=["segformer-b0", "segformer-b1", "segformer-b2", "segformer-b4", "auriganet"])
    p.add_argument("--data",    default=cfg.get("data",    "datasets/orfd"),
                   help="Path to dataset root (must contain training/ and validation/)")
    p.add_argument("--epochs",  type=int,   default=cfg.get("epochs",   100))
    p.add_argument("--batch",   type=int,   default=cfg.get("batch",    8))
    p.add_argument("--lr",      type=float, default=cfg.get("lr",       6e-5))
    p.add_argument("--wd",      type=float, default=cfg.get("wd",       0.01))
    p.add_argument("--workers", type=int,   default=cfg.get("workers",  4))
    p.add_argument("--patience",type=int,   default=cfg.get("patience", 10))
    p.add_argument("--seed",    type=int,   default=cfg.get("seed",     None))
    p.add_argument("--out",     default=cfg.get("out", None),
                   help="Output directory (default: weights/orfd/<model>/)")
    p.add_argument("--resume",  default=cfg.get("resume_from", ""),
                   help="Path to last.pth checkpoint to resume from")
    p.add_argument("--no-fp16", dest="fp16", action="store_false",
                   default=cfg.get("fp16", True),
                   help="Disable bfloat16 mixed-precision training")
    # Advanced knobs (expose previously hardcoded constants)
    p.add_argument("--n-warmup",       type=int,   default=cfg.get("n_warmup",       5),
                   help="Linear LR warmup epochs before cosine decay")
    p.add_argument("--clip-norm",      type=float, default=cfg.get("clip_norm",      1.0),
                   help="Gradient clipping max norm")
    p.add_argument("--label-smoothing",type=float, default=cfg.get("label_smoothing",0.1),
                   help="Cross-entropy label smoothing")
    p.add_argument("--freeze-backbone", action="store_true",
                   help="Freeze encoder; only train the segmentation head (and LoRA adapters if --lora)")
    p.add_argument("--lora", action="store_true",
                   help="Apply LoRA to SegFormer encoder Q/V projections (ignored for AurigaNet)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16   = args.fp16 and device == "cuda"

    if args.seed is not None:
        seed_everything(args.seed)
        logger.info("Global seed: %d", args.seed)

    if args.out:
        out_dir = Path(args.out)
        if not out_dir.is_absolute():
            out_dir = _ROOT / out_dir
    else:
        out_dir = _ROOT / "weights" / "orfd" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)
    logger.info("Device: %s  fp16: %s", device, fp16)

    # --- Datasets ---
    # Resolve relative paths against the project root so the script works
    # regardless of which directory the user runs it from.
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = _ROOT / data_path

    train_ds = ORFDDataset(str(data_path), split="training",   augment=True)
    val_ds   = ORFDDataset(str(data_path), split="validation", augment=False)

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
    is_auriganet = args.model == "auriganet"
    if is_auriganet:
        resume_weights = args.resume if args.resume and Path(args.resume).is_file() else ""
        model, processor = build_auriganet(device, fp16, weights=resume_weights)
    else:
        model, processor = build_segformer(args.model, device, fp16)

    # --- Optional: LoRA for SegFormer encoder ---
    if args.lora and not is_auriganet:
        from peft import get_peft_model, LoraConfig
        lora_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.1,
            target_modules=["query", "value"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        # PEFT freezes everything; re-enable decode_head for full fine-tuning.
        for p in model.decode_head.parameters():
            p.requires_grad_(True)
        logger.info("LoRA applied to SegFormer encoder Q/V projections.")
        model.print_trainable_parameters()

    # --- Optional: freeze backbone (encoder), train head (+ LoRA adapters) only ---
    if args.freeze_backbone:
        if is_auriganet:
            frozen = 0
            for name, p in model.named_parameters():
                if "Seg.area_fe" not in name:
                    p.requires_grad_(False)
                    frozen += p.numel()
            logger.info("AurigaNet: backbone frozen (%dM params).", frozen // 1_000_000)
        else:
            frozen = 0
            for name, p in model.named_parameters():
                # Keep decode_head trainable; keep any LoRA adapter trainable.
                if "decode_head" not in name and "lora_" not in name:
                    p.requires_grad_(False)
                    frozen += p.numel()
            logger.info("SegFormer: backbone frozen (%dM params).", frozen // 1_000_000)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info("Trainable params: %dM / %dM total.", trainable // 1_000_000, total // 1_000_000)

    # Rebuild param groups after freeze/LoRA (requires_grad may have changed).
    if is_auriganet:
        seg_params     = [p for p in model.Seg.area_fe.parameters() if p.requires_grad]
        backbone_params = [p for p in model.parameters()
                          if p.requires_grad and id(p) not in {id(q) for q in seg_params}]
        param_groups = [
            {"params": backbone_params, "lr": args.lr * 0.1},
            {"params": seg_params,      "lr": args.lr},
        ]
    else:
        head_params     = [p for p in model.decode_head.parameters() if p.requires_grad]
        head_ids        = {id(p) for p in head_params}
        backbone_params = [p for p in model.parameters()
                          if p.requires_grad and id(p) not in head_ids]
        param_groups = [
            {"params": backbone_params, "lr": args.lr * 0.1},
            {"params": head_params,      "lr": args.lr},
        ]

    optimizer = AdamW(param_groups, weight_decay=args.wd)
    warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                            total_iters=args.n_warmup)
    cosine_sched = CosineAnnealingLR(optimizer,
                                     T_max=max(1, args.epochs - args.n_warmup),
                                     eta_min=1e-7)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched],
                             milestones=[args.n_warmup])

    _label_smoothing = args.label_smoothing if not args.freeze_backbone else 0.0
    def criterion(logits, labels):
        return _dice_ce_loss(logits, labels, label_smoothing=_label_smoothing)

    # --- Resume ---
    start_epoch = 1
    best_miou   = 0.0

    if args.resume and not is_auriganet:
        # AurigaNet model weights are already loaded in build_auriganet; only
        # SegFormer needs the generic model.load_state_dict path.
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = torch.load(str(resume_path), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["net"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_miou   = ckpt.get("best_miou", ckpt.get("miou", 0.0))
        logger.info(
            "Resumed from %s  (epoch %d, best mIoU %.4f)",
            resume_path, start_epoch - 1, best_miou,
        )
    elif args.resume and is_auriganet:
        resume_path = Path(args.resume)
        if resume_path.exists():
            ckpt = torch.load(str(resume_path), map_location="cpu", weights_only=False)
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_miou   = ckpt.get("best_miou", ckpt.get("miou", 0.0))
            logger.info(
                "AurigaNet resumed from %s  (epoch %d, best mIoU %.4f)",
                resume_path, start_epoch - 1, best_miou,
            )

    # --- Training loop ---
    patience_left = args.patience
    log_entries: list[dict] = []

    # Reload existing log if resuming so we don't overwrite history.
    log_path = out_dir / "train_log.json"
    if args.resume and log_path.exists():
        try:
            log_entries = json.loads(log_path.read_text())
        except Exception:
            log_entries = []

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.perf_counter()

        train_loss = train_one_epoch(
            model, processor, train_loader, optimizer, criterion,
            device, fp16, args.clip_norm, is_auriganet=is_auriganet,
        )
        val_loss, val_miou = evaluate(
            model, processor, val_loader, criterion, device, fp16,
            is_auriganet=is_auriganet,
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
        log_path.write_text(json.dumps(log_entries, indent=2))

        # last.pth: full state for resume.
        _save_checkpoint(
            model, out_dir / "last.pth", epoch, val_miou,
            optimizer=optimizer, scheduler=scheduler, best_miou=best_miou,
        )

        if val_miou > best_miou:
            best_miou = val_miou
            patience_left = args.patience
            # best.pth: weights only (used by run_player.py).
            # If LoRA was used, merge adapters so the checkpoint is a plain state dict.
            if args.lora and not is_auriganet:
                merged = model.merge_and_unload()
                _save_checkpoint(merged, out_dir / "best.pth", epoch, val_miou)
            else:
                _save_checkpoint(model, out_dir / "best.pth", epoch, val_miou)
            logger.info("  → new best mIoU %.4f saved.", best_miou)
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("Early stopping: no improvement for %d epochs.", args.patience)
                break

    logger.info("Training done. Best val mIoU: %.4f", best_miou)
    logger.info("Best checkpoint: %s", out_dir / "best.pth")


def _save_checkpoint(
    model: nn.Module,
    path: Path,
    epoch: int,
    miou: float,
    optimizer=None,
    scheduler=None,
    best_miou: float | None = None,
) -> None:
    state: dict[str, Any] = {"net": model.state_dict(), "epoch": epoch, "miou": miou}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if best_miou is not None:
        state["best_miou"] = best_miou
    torch.save(state, str(path))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Distill Mask2Former-Large (ORFD-fine-tuned teacher) into SegFormer-B2
(production student architecture — the eventual Jetson deployment path is
unchanged either way).

Response-based distillation only (not feature-matching): Mask2Former predicts
(mask, class) query pairs, not dense per-pixel logits, so there's no natural
per-layer feature correspondence with SegFormer's dense encoder/decoder to
match directly — the teacher's *output* (converted to a dense per-pixel
probability map, same computation predict_logits() uses) is the only signal
transferred. Loss = hard-label Dice+CE against real ORFD GT (unchanged
_dice_ce_loss) + temperature-scaled KL against the teacher's dense
probabilities. The teacher is frozen throughout — only the student trains.

Student is warm-started from the production checkpoint by default (distilling
INTO the current best model, not re-deriving one from scratch).

Usage
-----
    python scripts/segmentation/training/train_distill.py
    python scripts/segmentation/training/train_distill.py --epochs 40 --kd-weight 0.5 --temperature 3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts" / "segmentation"))

from perception.datasets.orfd_torch import ORFDDataset
from _mask2former_common import build_mask2former, denormalize_to_uint8_rgb
from _orfd_common import (
    NUM_CLASSES,
    _dice_ce_loss,
    build_segformer,
    compute_miou,
    seed_everything,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_distill")

DEFAULT_TEACHER = str(_ROOT / "weights" / "segmentation" / "orfd" / "mask2former-large" / "best.pth")
DEFAULT_STUDENT_INIT = str(_ROOT / "weights" / "segmentation" / "orfd" / "frozen_backbone" / "segformer-b2" / "best.pth")


def _worker_init_fn(worker_id: int) -> None:
    np.random.seed(torch.initial_seed() % 2**32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distill Mask2Former-Large into SegFormer-B2")
    p.add_argument("--data",    default="datasets/segmentation/ORFD")
    p.add_argument("--teacher-weights", default=DEFAULT_TEACHER)
    p.add_argument("--teacher-backbone", default="facebook/mask2former-swin-large-ade-semantic")
    p.add_argument("--student-init", default=DEFAULT_STUDENT_INIT,
                   help="Student warm-start checkpoint (empty string = start from ADE20K-pretrained)")
    p.add_argument("--epochs",  type=int,   default=40)
    p.add_argument("--batch",   type=int,   default=8)
    p.add_argument("--lr",      type=float, default=6e-5)
    p.add_argument("--wd",      type=float, default=0.01)
    p.add_argument("--workers", type=int,   default=6)
    p.add_argument("--patience",type=int,   default=12)
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--out",     default=None)
    p.add_argument("--resume",  default="")
    p.add_argument("--no-fp16", dest="fp16", action="store_false", default=True)
    p.add_argument("--n-warmup",  type=int,   default=3)
    p.add_argument("--clip-norm", type=float, default=1.0)
    p.add_argument("--freeze-backbone", action="store_true", default=True,
                   help="Freeze SegFormer's MiT encoder; train only the decode head "
                        "(the recipe that beat full fine-tuning for this checkpoint originally)")
    p.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    p.add_argument("--temperature", type=float, default=3.0)
    p.add_argument("--kd-weight",   type=float, default=0.5,
                   help="Weight on the KD term; (1 - kd_weight) implicitly on the hard-label term "
                        "via --hard-weight, kept independent so both can be tuned")
    p.add_argument("--hard-weight", type=float, default=1.0)
    return p.parse_args()


@torch.no_grad()
def teacher_probs_batch(teacher_model, teacher_processor, images_chw: torch.Tensor,
                         device: str, target_hw: tuple[int, int]) -> torch.Tensor:
    """Frozen Mask2Former teacher -> (B, NUM_CLASSES, H, W) normalised probability
    distribution at target_hw, via the same query/mask->dense-map computation
    predict_logits() uses (minus argmax) — see _mask2former_common.py."""
    images_np = denormalize_to_uint8_rgb(images_chw)
    encoded = teacher_processor(images=images_np, return_tensors="pt")
    pixel_values = encoded["pixel_values"].to(device)

    outputs = teacher_model(pixel_values=pixel_values)
    class_queries_logits = outputs.class_queries_logits          # (B, Q, C+1)
    masks_queries_logits = outputs.masks_queries_logits            # (B, Q, h, w)
    masks_queries_logits = F.interpolate(
        masks_queries_logits, size=target_hw, mode="bilinear", align_corners=False,
    )
    masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]  # (B, Q, C)
    masks_probs = masks_queries_logits.sigmoid()                   # (B, Q, H, W)
    probs = torch.einsum("bqc,bqhw->bchw", masks_classes, masks_probs)  # (B, C, H, W)
    probs = probs.clamp(min=1e-8)
    probs = probs / probs.sum(dim=1, keepdim=True)
    return probs


def kd_loss(student_logits: torch.Tensor, teacher_probs: torch.Tensor, temperature: float) -> torch.Tensor:
    """Temperature-scaled KL divergence — student log-softmax(logits/T) vs. a
    tempered, renormalised teacher probability distribution. Standard
    Hinton-style T^2 scaling so the KD term's gradient magnitude doesn't
    shrink as temperature grows."""
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_tempered = teacher_probs.clamp(min=1e-8).pow(1.0 / temperature)
    teacher_tempered = teacher_tempered / teacher_tempered.sum(dim=1, keepdim=True)
    loss = F.kl_div(student_log_probs, teacher_tempered, reduction="batchmean")
    return loss * (temperature ** 2)


def train_one_epoch(student, processor, teacher, teacher_processor, loader,
                     optimizer, device, fp16, clip_norm, temperature, kd_weight, hard_weight) -> tuple[float, float]:
    student.train()
    total_hard, total_kd = 0.0, 0.0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        h, w = images.shape[-2:]

        teacher_probs = teacher_probs_batch(teacher, teacher_processor, images, device, (h, w))

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=fp16):
            from _orfd_common import segformer_forward
            student_logits = segformer_forward(student, processor, images, device, fp16=False)
            hard = _dice_ce_loss(student_logits, labels)
            kd = kd_loss(student_logits.float(), teacher_probs.float(), temperature)
            loss = hard_weight * hard + kd_weight * kd

        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=clip_norm)
        optimizer.step()
        total_hard += hard.item()
        total_kd += kd.item()

    n = len(loader)
    return total_hard / n, total_kd / n


@torch.no_grad()
def evaluate(student, processor, loader, device, fp16) -> tuple[float, float]:
    from _orfd_common import segformer_forward
    student.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc="val  ", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=fp16):
            logits = segformer_forward(student, processor, images, device, fp16=False)
            loss = _dice_ce_loss(logits, labels)
        total_loss += loss.item()
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())
    preds_cat, labels_cat = torch.cat(all_preds), torch.cat(all_labels)
    miou, _ = compute_miou(preds_cat, labels_cat)
    return total_loss / len(loader), miou


def _save_checkpoint(model, path, epoch, miou, optimizer=None, scheduler=None, best_miou=None) -> None:
    state: dict[str, Any] = {"net": model.state_dict(), "epoch": epoch, "miou": miou}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if best_miou is not None:
        state["best_miou"] = best_miou
    torch.save(state, str(path))


def main() -> int:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16   = args.fp16 and device == "cuda"

    if args.seed is not None:
        seed_everything(args.seed)
        logger.info("Global seed: %d", args.seed)

    out_dir = Path(args.out) if args.out else _ROOT / "weights" / "segmentation" / "orfd" / "distilled_segformer-b2"
    if not out_dir.is_absolute():
        out_dir = _ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)
    logger.info("Device: %s  fp16: %s  T=%.1f  kd_weight=%.2f  hard_weight=%.2f",
                device, fp16, args.temperature, args.kd_weight, args.hard_weight)

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
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    logger.info("Train: %d samples  Val: %d samples", len(train_ds), len(val_ds))

    logger.info("Loading frozen teacher: Mask2Former-Large from %s", args.teacher_weights)
    teacher, teacher_processor = build_mask2former(device, fp16=False, weights=args.teacher_weights,
                                                     backbone_id=args.teacher_backbone)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    logger.info("Loading student: SegFormer-B2, warm-start=%s", args.student_init or "(ADE20K-pretrained)")
    student_init = args.student_init if args.student_init and Path(args.student_init).is_file() else ""
    student, processor = build_segformer("segformer-b2", device, fp16)
    if student_init:
        from _segformer_checkpoint_common import load_remapped_state_dict
        state_dict = load_remapped_state_dict(Path(student_init))
        student.load_state_dict(state_dict, strict=True)
        logger.info("Student warm-started from %s", student_init)

    if args.freeze_backbone:
        frozen = 0
        for name, p in student.named_parameters():
            if "decode_head" not in name:
                p.requires_grad_(False)
                frozen += p.numel()
        logger.info("Student: MiT encoder frozen (%.2fM params).", frozen / 1e6)

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())
    logger.info("Student trainable params: %.2fM / %.2fM total.", trainable / 1e6, total / 1e6)

    head_params = [p for p in student.decode_head.parameters() if p.requires_grad]
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in student.parameters() if p.requires_grad and id(p) not in head_ids]
    param_groups = [
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params,     "lr": args.lr},
    ]
    optimizer = AdamW(param_groups, weight_decay=args.wd)
    warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.n_warmup)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.n_warmup), eta_min=1e-7)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[args.n_warmup])

    start_epoch = 1
    best_miou = 0.0
    log_entries: list[dict] = []
    log_path = out_dir / "train_log.json"

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            ckpt = torch.load(str(resume_path), map_location="cpu", weights_only=False)
            student.load_state_dict(ckpt["net"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_miou = ckpt.get("best_miou", ckpt.get("miou", 0.0))
            logger.info("Resumed from %s (epoch %d, best mIoU %.4f)", resume_path, start_epoch - 1, best_miou)
        if log_path.exists():
            try:
                log_entries = json.loads(log_path.read_text())
            except Exception:
                log_entries = []

    # Baseline: student's mIoU before any distillation, for a clean before/after comparison.
    _, baseline_miou = evaluate(student, processor, val_loader, device, fp16)
    logger.info("Student baseline mIoU (before distillation): %.4f", baseline_miou)

    patience_left = args.patience
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.perf_counter()

        hard_loss, kd = train_one_epoch(
            student, processor, teacher, teacher_processor, train_loader, optimizer,
            device, fp16, args.clip_norm, args.temperature, args.kd_weight, args.hard_weight,
        )
        val_loss, val_miou = evaluate(student, processor, val_loader, device, fp16)
        scheduler.step()

        elapsed = time.perf_counter() - t0
        logger.info(
            "Epoch %3d/%d  hard_loss=%.4f  kd_loss=%.4f  val_loss=%.4f  val_mIoU=%.4f  (%.1fs)",
            epoch, args.epochs, hard_loss, kd, val_loss, val_miou, elapsed,
        )

        log_entries.append(dict(epoch=epoch, hard_loss=hard_loss, kd_loss=kd,
                                 val_loss=val_loss, val_miou=val_miou))
        log_path.write_text(json.dumps(log_entries, indent=2))

        _save_checkpoint(student, out_dir / "last.pth", epoch, val_miou,
                          optimizer=optimizer, scheduler=scheduler, best_miou=best_miou)

        if val_miou > best_miou:
            best_miou = val_miou
            patience_left = args.patience
            _save_checkpoint(student, out_dir / "best.pth", epoch, val_miou)
            logger.info("  -> new best mIoU %.4f saved.", best_miou)
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("Early stopping: no improvement for %d epochs.", args.patience)
                break

    logger.info("Distillation done. Baseline mIoU: %.4f  Best distilled mIoU: %.4f  (delta %+.4f)",
                baseline_miou, best_miou, best_miou - baseline_miou)
    logger.info("Best checkpoint: %s", out_dir / "best.pth")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

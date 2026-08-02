"""Fine-tune a lightweight head on top of a FROZEN DINOv2 backbone on ORFD —
the "foundation model, cheap head-only training" candidate in the model
comparison phase (see _dinov2_common.py's docstring for the design rationale).

Usage
-----
    python scripts/segmentation/training/train_dinov2.py
    python scripts/segmentation/training/train_dinov2.py --lr 1e-3 --epochs 80

Output
------
Best checkpoint (by validation mIoU) → weights/segmentation/orfd/dinov2/best.pth
Last checkpoint (full state)         → weights/segmentation/orfd/dinov2/last.pth
Training log                         → weights/segmentation/orfd/dinov2/train_log.json
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
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts" / "segmentation"))

from perception.datasets.orfd_torch import ORFDDataset
from _dinov2_common import DINOV2_HF_BASE, build_dinov2, dinov2_forward
from _orfd_common import _dice_ce_loss, evaluate, seed_everything, train_one_epoch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_dinov2")


def _worker_init_fn(worker_id: int) -> None:
    np.random.seed(torch.initial_seed() % 2**32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune a DINOv2 + lightweight head on ORFD")
    p.add_argument("--data",    default="datasets/segmentation/ORFD")
    p.add_argument("--epochs",  type=int,   default=100)
    p.add_argument("--batch",   type=int,   default=16)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--wd",      type=float, default=0.01)
    p.add_argument("--workers", type=int,   default=6)
    p.add_argument("--patience",type=int,   default=15)
    p.add_argument("--seed",    type=int,   default=None)
    p.add_argument("--out",     default=None)
    p.add_argument("--resume",  default="")
    p.add_argument("--no-fp16", dest="fp16", action="store_false", default=True)
    p.add_argument("--n-warmup",       type=int,   default=5)
    p.add_argument("--clip-norm",      type=float, default=1.0)
    p.add_argument("--label-smoothing",type=float, default=0.1)
    p.add_argument("--backbone", default=DINOV2_HF_BASE,
                   help="HF DINOv2 backbone id (e.g. facebook/dinov2-large to escalate from base)")
    return p.parse_args()


def _save_checkpoint(model, path, epoch, miou, optimizer=None, scheduler=None, best_miou=None) -> None:
    state: dict[str, Any] = {"net": model.state_dict(), "epoch": epoch, "miou": miou}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if best_miou is not None:
        state["best_miou"] = best_miou
    torch.save(state, str(path))


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16   = args.fp16 and device == "cuda"

    if args.seed is not None:
        seed_everything(args.seed)
        logger.info("Global seed: %d", args.seed)

    out_dir = Path(args.out) if args.out else _ROOT / "weights" / "segmentation" / "orfd" / "dinov2"
    if not out_dir.is_absolute():
        out_dir = _ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)
    logger.info("Device: %s  fp16: %s", device, fp16)

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

    resume_weights = args.resume if args.resume and Path(args.resume).is_file() else ""
    model, _ = build_dinov2(device, fp16, weights=resume_weights, backbone_id=args.backbone)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info("Trainable params: %.2fM / %.2fM total (backbone frozen).", trainable / 1e6, total / 1e6)

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wd)
    warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.n_warmup)
    cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.n_warmup), eta_min=1e-7)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[args.n_warmup])

    def criterion(logits, labels):
        return _dice_ce_loss(logits, labels, label_smoothing=args.label_smoothing)

    def forward_fn(m: nn.Module, images: torch.Tensor) -> torch.Tensor:
        return dinov2_forward(m, images, device, fp16=fp16)

    start_epoch = 1
    best_miou   = 0.0
    log_entries: list[dict] = []
    log_path = out_dir / "train_log.json"

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            ckpt = torch.load(str(resume_path), map_location="cpu", weights_only=False)
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_miou   = ckpt.get("best_miou", ckpt.get("miou", 0.0))
            logger.info("Resumed from %s (epoch %d, best mIoU %.4f)", resume_path, start_epoch - 1, best_miou)
        if log_path.exists():
            try:
                log_entries = json.loads(log_path.read_text())
            except Exception:
                log_entries = []

    patience_left = args.patience
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.perf_counter()

        train_loss = train_one_epoch(
            model, None, train_loader, optimizer, criterion,
            device, fp16, args.clip_norm, forward_fn=forward_fn,
        )
        val_loss, val_miou = evaluate(
            model, None, val_loader, criterion, device, fp16, forward_fn=forward_fn,
        )
        scheduler.step()

        elapsed = time.perf_counter() - t0
        logger.info(
            "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  val_mIoU=%.4f  (%.1fs)",
            epoch, args.epochs, train_loss, val_loss, val_miou, elapsed,
        )

        log_entries.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss, val_miou=val_miou))
        log_path.write_text(json.dumps(log_entries, indent=2))

        _save_checkpoint(model, out_dir / "last.pth", epoch, val_miou,
                          optimizer=optimizer, scheduler=scheduler, best_miou=best_miou)

        if val_miou > best_miou:
            best_miou = val_miou
            patience_left = args.patience
            _save_checkpoint(model, out_dir / "best.pth", epoch, val_miou)
            logger.info("  -> new best mIoU %.4f saved.", best_miou)
        else:
            patience_left -= 1
            if patience_left <= 0:
                logger.info("Early stopping: no improvement for %d epochs.", args.patience)
                break

    logger.info("Training done. Best val mIoU: %.4f", best_miou)
    logger.info("Best checkpoint: %s", out_dir / "best.pth")


if __name__ == "__main__":
    main()

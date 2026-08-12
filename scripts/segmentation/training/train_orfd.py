"""Fine-tune a semantic segmentation model on the ORFD binary freespace dataset.

Supported models
----------------
  segformer-b0   Start from nvidia/segformer-b0-finetuned-ade-512-512
  segformer-b1   Start from nvidia/segformer-b1-finetuned-ade-512-512
  segformer-b2   Start from nvidia/segformer-b2-finetuned-ade-512-512
  segformer-b4   Start from nvidia/segformer-b4-finetuned-ade-512-512

Usage
-----
    # Use config/segmentation/train.yaml (default):
    python scripts/segmentation/training/train_orfd.py

    # Override individual settings:
    python scripts/segmentation/training/train_orfd.py --lr 1e-4 --epochs 50

    # Resume an interrupted run:
    python scripts/segmentation/training/train_orfd.py --resume weights/segmentation/orfd/segformer-b2/last.pth

Output
------
Best checkpoint (by validation mIoU) → weights/segmentation/orfd/<model_name>/best.pth
Last checkpoint (full state)         → weights/segmentation/orfd/<model_name>/last.pth
Training log                         → weights/segmentation/orfd/<model_name>/train_log.json
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
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

# Make sure the src package is importable when running as a script.
_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts" / "segmentation"))

from perception.datasets.orfd_torch import ORFDDataset
from _orfd_common import (
    IGNORE_INDEX,
    NUM_CLASSES,
    _dice_ce_loss,
    build_segformer,
    compute_miou,
    evaluate,
    segformer_forward,
    seed_everything,
    train_one_epoch,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_orfd")


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
        "data":               ds.get("root",          "datasets/segmentation/ORFD"),
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


def _worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker a unique but deterministic NumPy seed."""
    np.random.seed(torch.initial_seed() % 2**32)


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    # Two-pass: first extract --config, then load its values as argparse defaults.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(_ROOT / "config" / "segmentation" / "train.yaml"))
    known, _ = pre.parse_known_args()
    cfg = load_train_config(known.config)

    p = argparse.ArgumentParser(description="Fine-tune segmentation model on ORFD/custom dataset")
    p.add_argument("--config", default=str(_ROOT / "config" / "segmentation" / "train.yaml"),
                   help="Path to train.yaml config file")
    p.add_argument("--model",   default=cfg.get("model", "segformer-b2"),
                   choices=["segformer-b0", "segformer-b1", "segformer-b2", "segformer-b4"])
    p.add_argument("--data",    default=cfg.get("data",    "datasets/segmentation/ORFD"),
                   help="Path to dataset root (must contain training/ and validation/)")
    p.add_argument("--epochs",  type=int,   default=cfg.get("epochs",   100))
    p.add_argument("--batch",   type=int,   default=cfg.get("batch",    8))
    p.add_argument("--lr",      type=float, default=cfg.get("lr",       6e-5))
    p.add_argument("--wd",      type=float, default=cfg.get("wd",       0.01))
    p.add_argument("--workers", type=int,   default=cfg.get("workers",  4))
    p.add_argument("--patience",type=int,   default=cfg.get("patience", 10))
    p.add_argument("--seed",    type=int,   default=cfg.get("seed",     None))
    p.add_argument("--out",     default=cfg.get("out", None),
                   help="Output directory (default: weights/segmentation/orfd/<model>/)")
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
                   help="Apply LoRA to SegFormer encoder Q/V projections")
    p.add_argument("--gaza-data", default=None,
                   help="Optional path to the promoted Gaza-domain dataset "
                        "(datasets/segmentation/gaza_domain, see promote_gaza_labels.py). By "
                        "default trained jointly with --data via ConcatDataset + a "
                        "WeightedRandomSampler that balances the two domains and up-weights "
                        "rare-class Gaza-domain samples (see --gaza-only to change this). Omit "
                        "entirely to train on --data alone (unchanged behavior).")
    p.add_argument("--gaza-only", action="store_true",
                   help="Train on --gaza-data ALONE (not jointly with ORFD) -- for continuing "
                        "to fine-tune a checkpoint that's already trained on ORFD, on just the "
                        "225 new images, fast since the set is small. --data's validation split "
                        "is still used every epoch (unchanged) specifically to catch forgetting/"
                        "regression on the original distribution. Requires --gaza-data.")
    p.add_argument("--warm-start", default=None,
                   help="Path to a checkpoint to load ONLY the model weights from before training "
                        "(fresh optimizer/scheduler/epoch-0, unlike --resume which restores full "
                        "training state) -- e.g. the current production best.pth, to continue "
                        "training on --gaza-data without restarting from ADE20K. Mirrors "
                        "train_distill.py's --student-init.")
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
        out_dir = _ROOT / "weights" / "segmentation" / "orfd" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir)
    logger.info("Device: %s  fp16: %s", device, fp16)

    # --- Datasets ---
    # Resolve relative paths against the project root so the script works
    # regardless of which directory the user runs it from.
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = _ROOT / data_path

    if args.gaza_only and not args.gaza_data:
        raise SystemExit("--gaza-only requires --gaza-data")

    train_ds = ORFDDataset(str(data_path), split="training",   augment=True)
    val_ds   = ORFDDataset(str(data_path), split="validation", augment=False)

    sampler = None
    if args.gaza_data:
        from torch.utils.data import ConcatDataset, WeightedRandomSampler
        from perception.datasets.gaza_domain_torch import GazaDomainDataset

        gaza_path = Path(args.gaza_data)
        if not gaza_path.is_absolute():
            gaza_path = _ROOT / gaza_path
        gaza_ds = GazaDomainDataset(str(gaza_path), augment=True)

        if args.gaza_only:
            # Train on the 225 Gaza-domain images alone -- the checkpoint
            # being warm-started from is already trained on ORFD, so this
            # continues fine-tuning it on just the new domain rather than
            # re-mixing ORFD back into every batch. --data's validation
            # split is untouched (val_ds below) so it still catches
            # forgetting/regression each epoch even though ORFD isn't in
            # the training set anymore.
            sampler = WeightedRandomSampler(gaza_ds.sample_weights, num_samples=len(gaza_ds), replacement=True)
            train_ds = gaza_ds
            logger.info("Gaza-domain-only training: %d samples (ORFD validation still used to catch forgetting)",
                        len(gaza_ds))
        else:
            # Domain balance: give the (much smaller) Gaza-domain set equal
            # expected sampling mass to ORFD per epoch, on top of which its
            # own rare-class samples get an additional boost
            # (gaza_ds.sample_weights) -- see the GOOSE Class-Aware Repeat
            # Sampling technique this mirrors.
            domain_balance = len(train_ds) / len(gaza_ds)
            weights = [1.0] * len(train_ds) + [domain_balance * w for w in gaza_ds.sample_weights]
            combined_train_ds = ConcatDataset([train_ds, gaza_ds])
            sampler = WeightedRandomSampler(weights, num_samples=len(combined_train_ds), replacement=True)
            logger.info("Joint training: ORFD (%d) + Gaza-domain (%d), domain_balance=%.2f",
                        len(train_ds), len(gaza_ds), domain_balance)
            train_ds = combined_train_ds

    train_loader = DataLoader(
        train_ds, batch_size=args.batch,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        worker_init_fn=_worker_init_fn if args.seed is not None else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    logger.info("Train: %d samples  Val: %d samples", len(train_ds), len(val_ds))

    # --- Model ---
    model, processor = build_segformer(args.model, device, fp16)

    if args.warm_start:
        warm_start_path = Path(args.warm_start)
        if not warm_start_path.is_absolute():
            warm_start_path = _ROOT / warm_start_path
        if not warm_start_path.is_file():
            raise FileNotFoundError(f"--warm-start checkpoint not found: {warm_start_path}")
        from _segformer_checkpoint_common import load_remapped_state_dict
        state_dict = load_remapped_state_dict(warm_start_path)
        model.load_state_dict(state_dict, strict=True)
        logger.info("Model warm-started from %s (fresh optimizer/scheduler, epoch 0)", warm_start_path)

    # --- Optional: LoRA for SegFormer encoder ---
    if args.lora:
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

    if args.resume:
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
            device, fp16, args.clip_norm,
        )
        val_loss, val_miou = evaluate(
            model, processor, val_loader, criterion, device, fp16,
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
            if args.lora:
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

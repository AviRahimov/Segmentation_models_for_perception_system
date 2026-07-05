"""Hyperparameter sweep training — YOLO26 / YOLO11 (non-YOLOE).

Trains one or more named variants of a model with different freeze / augmentation
settings.  Each variant is saved under weights/detection/{model}/exp/{variant}/.

Round 2 is reserved for continual learning.  This script uses the exp/ subdirectory.

Usage
-----
    # All 5 variants for yolo26m:
    python scripts/detection/training/train_exp.py --model yolo26m

    # Specific variants only:
    python scripts/detection/training/train_exp.py --model yolo11m --variants freeze10 freeze10_aug_clean

    # Override dataset / device:
    python scripts/detection/training/train_exp.py --model yolo26m \\
        --data datasets/Detection_Dataset/data.yaml --device 0

Output
------
    weights/detection/{model}/exp/{variant}/best.pt
    weights/detection/{model}/exp/{variant}/last.pt
    weights/detection/{model}/exp/{variant}/results.csv
    weights/detection/{model}/exp/{variant}/weights/  (Ultralytics native dir)
"""
from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_exp")

# ---------------------------------------------------------------------------
# Model registry (standard YOLO only — no YOLOE; YOLOE uses YOLOEPETrainer)
# ---------------------------------------------------------------------------
_MODELS: dict[str, str] = {
    "yolo26n": "yolo26n.pt",
    "yolo26s": "yolo26s.pt",
    "yolo26m": "yolo26m.pt",
    "yolo26l": "yolo26l.pt",
    "yolo11n": "yolo11n.pt",
    "yolo11s": "yolo11s.pt",
    "yolo11m": "yolo11m.pt",
    "yolo11l": "yolo11l.pt",
}

# Per-model freeze default (same as round1, used by aug_clean variant)
_FREEZE_DEFAULTS: dict[str, int] = {
    "yolo26n": 10,
    "yolo26s":  8,
    "yolo26m":  6,
    "yolo26l":  4,
    "yolo11n":  7,
    "yolo11s":  6,
    "yolo11m":  5,
    "yolo11l":  4,
}

# ---------------------------------------------------------------------------
# Hyperparameter variants
# freeze=None → use _FREEZE_DEFAULTS for the model (same as round1)
# ---------------------------------------------------------------------------
_VARIANTS: dict[str, dict[str, Any]] = {
    "freeze0": {
        "freeze": 0,
        "mosaic": 1.0,
        "mixup": 0.2,
        "copy_paste": 0.15,
        "description": "Full fine-tune, round1 augmentation — lower bound reference",
    },
    "freeze10": {
        "freeze": 10,
        "mosaic": 1.0,
        "mixup": 0.2,
        "copy_paste": 0.15,
        "description": "Backbone frozen at layer 10 — research sweet-spot for ~150-image datasets",
    },
    "freeze21": {
        "freeze": 21,
        "mosaic": 1.0,
        "mixup": 0.2,
        "copy_paste": 0.15,
        "description": "Backbone + neck frozen — head-only training",
    },
    "aug_clean": {
        "freeze": None,
        "mosaic": 0.5,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "description": "Round1 freeze, clean augmentation (no blending — avoids boundary corruption)",
    },
    "freeze10_aug_clean": {
        "freeze": 10,
        "mosaic": 0.5,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "description": "freeze=10 + clean augmentation — expected best for small datasets",
    },
}

# Shared base hyperparams (same as round1, variants only override what changes)
_BASE_KWARGS: dict[str, Any] = {
    "epochs":        150,
    "imgsz":         640,
    "batch":         16,
    "lr0":           2e-4,
    "lrf":           0.01,
    "weight_decay":  5e-4,
    "warmup_epochs": 5,
    "close_mosaic":  30,
    "optimizer":     "AdamW",
    "patience":      20,
    "save_period":   20,
    "workers":       8,
    "degrees":       10.0,
    "translate":     0.2,
    "scale":         0.6,
    "flipud":        0.15,
    "fliplr":        0.5,
    "erasing":       0.4,
    "val":           True,
    "plots":         True,
    "verbose":       False,
    "exist_ok":      True,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hyperparameter sweep training — YOLO26 / YOLO11 variants"
    )
    p.add_argument("--model", required=True, choices=list(_MODELS.keys()),
                   help="Model to train (e.g. yolo26m, yolo11m)")
    p.add_argument("--variants", nargs="+", default=["all"],
                   metavar="VARIANT",
                   help=f"Variants to run (default: all). Choices: {list(_VARIANTS)}")
    p.add_argument("--data", default="datasets/Detection_Dataset/data.yaml",
                   help="Path to YOLO data.yaml")
    p.add_argument("--device", default="0",
                   help="CUDA device index or 'cpu' (default: 0)")
    p.add_argument("--seed",   type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)
        logger.info("Global seed: %d", args.seed)

    # Resolve which variants to run
    if args.variants == ["all"]:
        variants_to_run = list(_VARIANTS.keys())
    else:
        unknown = [v for v in args.variants if v not in _VARIANTS]
        if unknown:
            logger.error("Unknown variants: %s. Valid choices: %s", unknown, list(_VARIANTS))
            sys.exit(1)
        variants_to_run = args.variants

    # Resolve data path
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = _ROOT / data_path
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")

    base_weights = _MODELS[args.model]
    default_freeze = _FREEZE_DEFAULTS.get(args.model, 8)

    logger.info("Model:    %s  (base weights: %s)", args.model, base_weights)
    logger.info("Variants: %s", variants_to_run)
    logger.info("Data:     %s", data_path)

    results_summary: list[tuple[str, float, float]] = []  # (variant, mAP50, mAP50-95)

    for variant_name in variants_to_run:
        variant = _VARIANTS[variant_name]
        freeze = variant["freeze"] if variant["freeze"] is not None else default_freeze

        out_dir = _ROOT / "weights" / "detection" / args.model / "exp" / variant_name
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("")
        logger.info("=" * 70)
        logger.info("Variant: %s  |  freeze=%d  mosaic=%.1f  mixup=%.2f  copy_paste=%.2f",
                    variant_name, freeze, variant["mosaic"], variant["mixup"], variant["copy_paste"])
        logger.info("  %s", variant["description"])
        logger.info("  Output: %s", out_dir)
        logger.info("=" * 70)

        train_kwargs: dict[str, Any] = {
            **_BASE_KWARGS,
            "data":       str(data_path),
            "device":     args.device,
            "seed":       args.seed,
            "freeze":     freeze,
            "mosaic":     variant["mosaic"],
            "mixup":      variant["mixup"],
            "copy_paste": variant["copy_paste"],
            "project":    str(out_dir.parent),
            "name":       variant_name,
        }

        map50, map5095 = _run_variant(base_weights, train_kwargs, out_dir, args.model, variant_name)
        results_summary.append((variant_name, map50, map5095))

    # Print ranking table
    logger.info("")
    logger.info("=" * 70)
    logger.info("SWEEP COMPLETE — %s", args.model)
    logger.info("%-25s  %8s  %10s", "Variant", "mAP50", "mAP50-95")
    logger.info("-" * 50)
    for name, m50, m5095 in sorted(results_summary, key=lambda x: -x[1]):
        logger.info("%-25s  %8.4f  %10.4f", name, m50, m5095)
    logger.info("=" * 70)
    logger.info("Run summarize_exp.py to compare with round1 baseline.")


def _run_variant(
    base_weights: str,
    train_kwargs: dict[str, Any],
    out_dir: Path,
    model_name: str,
    variant_name: str,
) -> tuple[float, float]:
    """Train one variant and return (mAP50, mAP50-95). Returns (nan, nan) on error."""
    from ultralytics import YOLO

    logger.info("Loading base weights: %s", base_weights)
    model = YOLO(base_weights)

    logger.info("Training freeze=%d epochs=%d batch=%d ...",
                train_kwargs["freeze"], train_kwargs["epochs"], train_kwargs["batch"])
    results = model.train(**train_kwargs)

    # Locate best.pt — Ultralytics saves under {project}/{name}/weights/
    weights_dir = out_dir / "weights"
    best_src = weights_dir / "best.pt"
    last_src = weights_dir / "last.pt"

    if not best_src.exists() and hasattr(model, "trainer") and model.trainer is not None:
        best_src = model.trainer.best
        last_src = model.trainer.last

    best_dest = out_dir / "best.pt"
    last_dest = out_dir / "last.pt"

    if best_src.exists() and best_src.resolve() != best_dest.resolve():
        shutil.copy2(str(best_src), str(best_dest))
        logger.info("Best checkpoint → %s", best_dest)

    if last_src.exists() and last_src.resolve() != last_dest.resolve():
        shutil.copy2(str(last_src), str(last_dest))

    map50, map5095 = float("nan"), float("nan")
    if results is not None:
        try:
            metrics = results.results_dict
            map50   = metrics.get("metrics/mAP50(B)",    float("nan"))
            map5095 = metrics.get("metrics/mAP50-95(B)", float("nan"))
            logger.info("Variant %s — mAP50=%.4f  mAP50-95=%.4f", variant_name, map50, map5095)
        except Exception:
            pass

    return map50, map5095


if __name__ == "__main__":
    main()

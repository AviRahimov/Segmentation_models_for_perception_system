"""Round 1 detection training — YOLO26 and YOLOE-26.

Trains a single model variant on the detection dataset.  Model family (YOLO26
vs YOLOE-26) is inferred automatically from the --model name.

YOLO26 path:   standard Ultralytics trainer with freeze=8 (first 8 backbone
               layers frozen), AdamW, heavy augmentation for small datasets.

YOLOE-26 path: YOLOEPETrainer (linear probing) — at training start, class
               text embeddings are computed via MobileCLIP2 and fused into the
               classification head as static weights; only the head projection
               layers are trained.  freeze= is ignored by this trainer.

Usage
-----
    # YOLO26 variants (s/m/l):
    python scripts/detection/training/train_round1.py --model yolo26m
    python scripts/detection/training/train_round1.py --model yolo26l --batch 16

    # YOLOE-26 variants (s/m/l):
    python scripts/detection/training/train_round1.py --model yoloe-26m
    python scripts/detection/training/train_round1.py --model yoloe-26s --epochs 200

    # Override config:
    python scripts/detection/training/train_round1.py \\
        --config config/detection/train.yaml --model yolo26m --batch 32 --epochs 100

Output
------
    weights/detection/{model_name}/round1/best.pt   — best checkpoint by val mAP50
    weights/detection/{model_name}/round1/last.pt   — last epoch checkpoint
    weights/detection/{model_name}/round1/results.csv  — per-epoch metrics
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
import yaml

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_round1")

# Supported model names → (base_weights, is_yoloe)
_MODELS: dict[str, tuple[str, bool]] = {
    # YOLO26 — standard Ultralytics YOLO trainer
    "yolo26n":    ("yolo26n.pt",     False),
    "yolo26s":    ("yolo26s.pt",     False),
    "yolo26m":    ("yolo26m.pt",     False),
    "yolo26l":    ("yolo26l.pt",     False),
    # YOLOE-26 — detection-only weights don't exist; use seg checkpoint (same
    # backbone/neck) and transfer into detection YAML via YOLOEPETrainer.
    "yoloe-26n":  ("yoloe-26n-seg.pt",   True),
    "yoloe-26s":  ("yoloe-26s-seg.pt",   True),
    "yoloe-26m":  ("yoloe-26m-seg.pt",   True),
    "yoloe-26l":  ("yoloe-26l-seg.pt",   True),
    # YOLOv11 — standard Ultralytics YOLO trainer (DFL-based, same API as YOLO26)
    "yolo11n":    ("yolo11n.pt",     False),
    "yolo11s":    ("yolo11s.pt",     False),
    "yolo11m":    ("yolo11m.pt",     False),
}

# Scale-aware freeze defaults for standard YOLO models (YOLOEPETrainer ignores freeze).
# Pattern: smaller model = more frozen (less trainable capacity on 157-image dataset).
# YOLO26 and YOLO11 share the same 11-block backbone structure.
_FREEZE_DEFAULTS: dict[str, int] = {
    # YOLO26
    "yolo26n": 10,   # freeze 0–9 (through SPPF); only C2PSA + head trainable
    "yolo26s":  8,   # freeze 0–7
    "yolo26m":  6,   # freeze 0–5
    "yolo26l":  4,   # freeze 0–3
    # YOLOv11 (same 11-block backbone, same freeze logic)
    "yolo11n":  7,
    "yolo11s":  6,
    "yolo11m":  5,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict[str, Any]:
    """Load train.yaml and return a merged flat dict of defaults."""
    p = Path(config_path)
    if not p.is_absolute():
        p = _ROOT / p
    if not p.exists():
        return {}
    with p.open() as f:
        raw = yaml.safe_load(f) or {}
    ds = raw.get("dataset", {}) or {}
    m  = raw.get("model",   {}) or {}
    tr = raw.get("training", {}) or {}
    return {
        "data":          ds.get("path",          "datasets/Detection_Dataset/data.yaml"),
        "model":         m.get("name",            "yolo26m"),
        "out_dir":       m.get("out_dir",         "weights/detection/{model_name}/round1"),
        "epochs":        tr.get("epochs",          300),
        "imgsz":         tr.get("imgsz",           640),
        "batch":         tr.get("batch",           16),
        "freeze":        tr.get("freeze") or None,   # None = auto by scale
        "lr0":           tr.get("lr0",             2e-4),
        "lrf":           tr.get("lrf",             0.01),
        "weight_decay":  tr.get("weight_decay",    5e-4),
        "warmup_epochs": tr.get("warmup_epochs",   5),
        "close_mosaic":  tr.get("close_mosaic",    50),
        "optimizer":     tr.get("optimizer",       "AdamW"),
        "patience":      tr.get("patience",        80),
        "seed":          tr.get("seed",            42),
        "save_period":   tr.get("save_period",     20),
        "workers":       tr.get("workers",         8),
        "device":        tr.get("device",          "0"),
        "mosaic":        tr.get("mosaic",          1.0),
        "mixup":         tr.get("mixup",           0.2),
        "copy_paste":    tr.get("copy_paste",      0.15),
        "degrees":       tr.get("degrees",         10.0),
        "translate":     tr.get("translate",       0.2),
        "scale":         tr.get("scale",           0.6),
        "flipud":        tr.get("flipud",          0.15),
        "fliplr":        tr.get("fliplr",          0.5),
        "erasing":       tr.get("erasing",         0.4),
    }


def parse_args() -> argparse.Namespace:
    default_config = str(_ROOT / "config" / "detection" / "train.yaml")

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=default_config)
    known, _ = pre.parse_known_args()
    cfg = load_config(known.config)

    p = argparse.ArgumentParser(description="Round 1 detection training (YOLO26 / YOLOE-26)")
    p.add_argument("--config",  default=default_config)
    p.add_argument("--model",   default=cfg["model"],
                   choices=list(_MODELS.keys()),
                   help="Model variant to train")
    p.add_argument("--data",    default=cfg["data"],
                   help="Path to YOLO data.yaml")
    p.add_argument("--out",     default=None,
                   help="Override output directory (default: weights/detection/{model}/round1)")
    p.add_argument("--epochs",  type=int,   default=cfg["epochs"])
    p.add_argument("--imgsz",   type=int,   default=cfg["imgsz"])
    p.add_argument("--batch",   type=int,   default=cfg["batch"])
    p.add_argument("--freeze",  type=int,   default=cfg.get("freeze") or None,
                   help="Layers to freeze (YOLO26 only). Default: s=8, m=6, l=4 (auto by scale)")
    p.add_argument("--lr0",     type=float, default=cfg["lr0"])
    p.add_argument("--device",  default=cfg["device"])
    p.add_argument("--seed",    type=int,   default=cfg["seed"])
    p.add_argument("--patience",type=int,   default=cfg["patience"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    default_config = str(_ROOT / "config" / "detection" / "train.yaml")
    cfg = load_config(args.config if args.config != default_config else str(_ROOT / "config" / "detection" / "train.yaml"))

    if args.seed is not None:
        seed_everything(args.seed)
        logger.info("Global seed: %d", args.seed)

    base_weights, is_yoloe = _MODELS[args.model]

    # Resolve output directory
    out_dir_template = args.out or cfg["out_dir"]
    out_dir = Path(out_dir_template.format(model_name=args.model))
    if not out_dir.is_absolute():
        out_dir = _ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Model:      %s  (%s)", args.model, "YOLOE-26" if is_yoloe else "YOLO26")
    logger.info("Output dir: %s", out_dir)

    # Resolve data path
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = _ROOT / data_path
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")

    # Build shared train kwargs (passed to model.train())
    train_kwargs: dict[str, Any] = {
        "data":          str(data_path),
        "epochs":        args.epochs,
        "imgsz":         args.imgsz,
        "batch":         args.batch,
        "lr0":           args.lr0,
        "lrf":           cfg["lrf"],
        "weight_decay":  cfg["weight_decay"],
        "warmup_epochs": cfg["warmup_epochs"],
        "close_mosaic":  cfg["close_mosaic"],
        "optimizer":     cfg["optimizer"],
        "patience":      args.patience,
        "save_period":   cfg["save_period"],
        "workers":       cfg["workers"],
        "device":        args.device,
        "seed":          args.seed,
        # Augmentation
        "mosaic":        cfg["mosaic"],
        "mixup":         cfg["mixup"],
        "copy_paste":    cfg["copy_paste"],
        "degrees":       cfg["degrees"],
        "translate":     cfg["translate"],
        "scale":         cfg["scale"],
        "flipud":        cfg["flipud"],
        "fliplr":        cfg["fliplr"],
        "erasing":       cfg["erasing"],
        # Output routing: Ultralytics saves to {project}/{name}/weights/
        "project":       str(out_dir.parent),
        "name":          out_dir.name,
        "exist_ok":      True,
        "val":           True,
        "plots":         True,
        "verbose":       False,
    }

    if is_yoloe:
        _train_yoloe(base_weights, train_kwargs, out_dir, args.model)
    else:
        freeze = args.freeze if args.freeze is not None else _FREEZE_DEFAULTS.get(args.model, 8)
        logger.info("freeze=%d  (auto by scale: %s)", freeze, args.freeze is None)
        train_kwargs["freeze"] = freeze
        _train_yolo26(base_weights, train_kwargs, out_dir, args.model)


def _train_yolo26(base_weights: str, train_kwargs: dict[str, Any], out_dir: Path, model_name: str) -> None:
    """Train a standard YOLO26 closed-vocabulary detection model."""
    from ultralytics import YOLO

    logger.info("Loading YOLO26 base weights: %s", base_weights)
    model = YOLO(base_weights)

    logger.info("Starting YOLO26 training (freeze=%d, epochs=%d, batch=%d)...",
                train_kwargs["freeze"], train_kwargs["epochs"], train_kwargs["batch"])
    results = model.train(**train_kwargs)

    # Locate best checkpoint (Ultralytics saves to {project}/{name}/weights/)
    weights_dir = out_dir / "weights"
    best_src = weights_dir / "best.pt"
    last_src = weights_dir / "last.pt"

    if not best_src.exists():
        # Fallback: search in trainer save_dir
        if hasattr(model, "trainer") and model.trainer is not None:
            best_src = model.trainer.best
            last_src = model.trainer.last

    _report_and_copy(results, best_src, last_src, out_dir, model_name)


def _train_yoloe(base_weights: str, train_kwargs: dict[str, Any], out_dir: Path, model_name: str) -> None:
    """Fine-tune YOLOE-26 via YOLOEPETrainer (linear probing).

    Ultralytics only publishes YOLOE-26 *segmentation* checkpoints (yoloe-26s-seg.pt etc.),
    not detection-only weights.  YOLOEPETrainer also reconstructs the model internally from
    the checkpoint's embedded yaml, so passing the seg checkpoint would build the wrong
    architecture.  The workaround:

      1. Build the detection model from yaml (yoloe-26s.yaml) — correct architecture.
      2. Transfer all matching backbone+neck+detect-head weights from the seg checkpoint
         via strict=False (all 822 detection keys exist in the 958-key seg checkpoint).
      3. Save this initialised model as a temporary checkpoint whose embedded yaml points
         to the detection yaml, so YOLOEPETrainer sees the correct architecture.
      4. Load from the temp checkpoint → train with YOLOEPETrainer → delete temp file.
    """
    import torch
    from copy import deepcopy
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEPETrainer

    # 1. Build detection model from yaml
    det_yaml = f"{model_name}.yaml"
    logger.info("Building YOLOE-26 detection architecture from yaml: %s", det_yaml)
    det_model = YOLOE(det_yaml)

    # 2. Transfer backbone + neck weights from seg checkpoint (strict=False, 822/822 keys match)
    # Use Ultralytics downloader so the checkpoint is fetched if not cached locally.
    logger.info("Transferring pretrained weights from seg checkpoint: %s", base_weights)
    from ultralytics.utils.downloads import attempt_download_asset
    local_seg = attempt_download_asset(base_weights)
    seg_ckpt = torch.load(local_seg, map_location="cpu", weights_only=False)
    seg_state = seg_ckpt["model"].state_dict()
    result = det_model.model.load_state_dict(seg_state, strict=False)
    n_loaded = len(det_model.model.state_dict()) - len(result.missing_keys)
    logger.info("Weight transfer: %d/%d keys loaded, %d missing",
                n_loaded, len(det_model.model.state_dict()), len(result.missing_keys))

    # 3. Save temp checkpoint — YOLOEPETrainer rebuilds from ckpt['model'].yaml internally
    tmp_ckpt = out_dir / "_tmp_det_init.pt"
    torch.save({"epoch": -1, "best_fitness": None, "model": deepcopy(det_model.model),
                "ema": None, "updates": None, "optimizer": None,
                "train_args": {}, "train_metrics": {}, "train_results": None},
               tmp_ckpt)
    logger.info("Saved temp detection checkpoint: %s", tmp_ckpt)

    # 4. Load from temp checkpoint and train
    model = YOLOE(str(tmp_ckpt))
    logger.info("Loaded from temp checkpoint — task: %s, head: %s",
                model.task, type(model.model.model[-1]).__name__)

    # freeze= is not used by YOLOEPETrainer (it manages its own layer freeze)
    train_kwargs.pop("freeze", None)

    logger.info("Starting YOLOE-26 training via YOLOEPETrainer (epochs=%d, batch=%d)...",
                train_kwargs["epochs"], train_kwargs["batch"])
    results = model.train(trainer=YOLOEPETrainer, **train_kwargs)

    # Clean up temp checkpoint
    tmp_ckpt.unlink(missing_ok=True)

    weights_dir = out_dir / "weights"
    best_src = weights_dir / "best.pt"
    last_src = weights_dir / "last.pt"

    if not best_src.exists() and hasattr(model, "trainer") and model.trainer is not None:
        best_src = model.trainer.best
        last_src = model.trainer.last

    _report_and_copy(results, best_src, last_src, out_dir, model_name)


def _report_and_copy(results: Any, best_src: Path, last_src: Path, out_dir: Path, model_name: str) -> None:
    """Log final metrics and ensure best.pt / last.pt are in out_dir."""
    # Ultralytics already saves inside out_dir/weights/ — make top-level symlinks/copies
    # so callers can use `weights/detection/{model}/round1/best.pt` directly.
    best_dest = out_dir / "best.pt"
    last_dest = out_dir / "last.pt"

    if best_src.exists() and best_src.resolve() != best_dest.resolve():
        shutil.copy2(str(best_src), str(best_dest))
        logger.info("Best checkpoint → %s", best_dest)

    if last_src.exists() and last_src.resolve() != last_dest.resolve():
        shutil.copy2(str(last_src), str(last_dest))

    # Print summary metrics
    if results is not None:
        try:
            metrics = results.results_dict
            map50    = metrics.get("metrics/mAP50(B)",    float("nan"))
            map5095  = metrics.get("metrics/mAP50-95(B)", float("nan"))
            logger.info("=" * 60)
            logger.info("Training complete — %s", model_name)
            logger.info("  mAP50:     %.4f", map50)
            logger.info("  mAP50-95:  %.4f", map5095)
            logger.info("  Best checkpoint: %s", best_dest)
            logger.info("=" * 60)
        except Exception:
            logger.info("Training complete. Check %s for results.", out_dir)


if __name__ == "__main__":
    main()

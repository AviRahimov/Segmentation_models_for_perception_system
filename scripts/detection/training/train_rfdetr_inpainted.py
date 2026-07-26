#!/usr/bin/env python3
"""Phase 6 — retrain rfdetr-m on Detection_Dataset_hardneg_inpainted (base
Detection_Dataset_hardneg + promoted inpainted hard negatives), using the
SAME recipe as the production checkpoint (conservative_aug: RF-DETR's own
AUG_CONSERVATIVE preset, lr_vit_layer_decay=0.8, its documented
small-dataset recommendation) -- no Optuna hyperparameters.

Why not Optuna's tuned hyperparameters (as the original plan sketched):
Phase 5 found every tried Optuna trial (16, 10, 7) increased false positives
on the real test benchmark despite a higher training-time mAP50 proxy --
layering the inpainted negatives on top of a config already rejected for
that reason would confound the comparison. This isolates ONE variable
(the dataset) against the untouched production baseline, matching the
"one variable at a time" approach used for CDN/VFL/Optuna earlier.

Output directory safety: writes to
weights/detection/rfdetr-m/detection_dataset_hardneg_inpainted/conservative_aug/
-- a different dataset-name segment than the production checkpoint's own
path (.../detection_dataset_hardneg/conservative_aug/), so there's no
natural collision, but this still checks + refuses just in case, and
verifies the production checkpoint's file size is unchanged afterward --
same guard pattern as train_rfdetr_optuna_best.py (Phase 5).

Usage
-----
    python scripts/detection/training/train_rfdetr_inpainted.py
"""
from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]

try:
    import rfdetr_plus  # noqa: F401
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("train_rfdetr_inpainted")

_DATASET_DIR = _ROOT / "datasets/Detection_Dataset_hardneg_inpainted"
_OUT_DIR = _ROOT / "weights/detection/rfdetr-m/detection_dataset_hardneg_inpainted/conservative_aug"
_PRODUCTION_CKPT = _ROOT / "weights/detection/rfdetr-m/detection_dataset_hardneg/conservative_aug/best.pt"


def _read_final_metrics(out_dir: Path) -> tuple[float, float]:
    """Identical helper to train_rfdetr_optuna_best.py / train_detector_rfdetr.py
    -- duplicated rather than imported, matching their own zero-cross-import convention."""
    import csv

    metrics_csv = out_dir / "metrics.csv"
    if not metrics_csv.exists():
        return float("nan"), float("nan")
    best50, best5095 = float("nan"), float("nan")
    with metrics_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            raw = row.get("val/mAP_50", "")
            if not raw:
                continue
            try:
                m50 = float(raw)
                m5095 = float(row.get("val/mAP_50_95", "nan") or "nan")
            except ValueError:
                continue
            if best50 != best50 or m50 > best50:
                best50, best5095 = m50, m5095
    return best50, best5095


def main() -> int:
    from rfdetr.datasets.aug_config import AUG_CONSERVATIVE

    if not _DATASET_DIR.exists():
        logger.error("Dataset not found: %s -- run init_inpainted_dataset.py + "
                    "promote_inpainted_negatives.py first.", _DATASET_DIR)
        return 1
    if _OUT_DIR.resolve() == _PRODUCTION_CKPT.parent.resolve():
        logger.error("Refusing to run: output dir would collide with the production checkpoint dir (%s)",
                     _PRODUCTION_CKPT.parent)
        return 1
    if _OUT_DIR.exists() and any(_OUT_DIR.iterdir()):
        logger.error("Refusing to run: %s already exists and is non-empty. "
                    "Delete it yourself first if you want to redo this run.", _OUT_DIR)
        return 1
    if not _PRODUCTION_CKPT.exists():
        logger.error("Production checkpoint missing, aborting as a precaution: %s", _PRODUCTION_CKPT)
        return 1
    prod_size_before = _PRODUCTION_CKPT.stat().st_size

    n_train = len(list((_DATASET_DIR / "train" / "images").iterdir()))
    logger.info("Phase 6 retrain -- rfdetr-m, conservative_aug recipe (matches production), "
               "dataset=%s (%d train images)", _DATASET_DIR.relative_to(_ROOT), n_train)

    import rfdetr as rfdetr_pkg

    model = rfdetr_pkg.RFDETRMedium()
    model.train(
        dataset_dir=str(_DATASET_DIR),
        dataset_file="yolo",
        output_dir=str(_OUT_DIR),
        epochs=100,
        batch_size="auto",
        seed=42,
        progress_bar="tqdm",
        early_stopping=True,
        lr_vit_layer_decay=0.8,
        aug_config=dict(AUG_CONSERVATIVE),
    )

    best_src = _OUT_DIR / "checkpoint_best_total.pth"
    best_dest = _OUT_DIR / "best.pt"
    if best_src.exists():
        shutil.copy2(str(best_src), str(best_dest))
        logger.info("Best checkpoint -> %s", best_dest)
    else:
        logger.warning("Expected checkpoint not found: %s", best_src)

    map50, map5095 = _read_final_metrics(_OUT_DIR)
    logger.info("Phase 6 retrain done -- mAP50=%.4f  mAP50-95=%.4f", map50, map5095)

    if _PRODUCTION_CKPT.stat().st_size != prod_size_before:
        logger.error("Production checkpoint changed size during this run -- INVESTIGATE IMMEDIATELY: %s",
                     _PRODUCTION_CKPT)
        return 1
    logger.info("Verified production checkpoint untouched: %s", _PRODUCTION_CKPT)

    sys.path.insert(0, str(_ROOT / "scripts" / "detection" / "training"))
    from _survey_common import _log_experiment  # noqa: E402

    _log_experiment({
        "mode": "phase6_inpainted_retrain", "arch_family": "rfdetr", "model": "rfdetr-m",
        "dataset": "Detection_Dataset_hardneg_inpainted", "recipe": "conservative_aug",
        "run_dir": str(_OUT_DIR.relative_to(_ROOT)),
        "epochs": 100, "batch_size": "auto", "seed": 42, "init": "coco",
        "n_train_images": n_train,
        "train_mAP50": round(map50, 5) if map50 == map50 else None,
        "train_mAP50_95": round(map5095, 5) if map5095 == map5095 else None,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

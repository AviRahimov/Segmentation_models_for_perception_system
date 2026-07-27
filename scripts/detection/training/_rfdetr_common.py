"""Shared helpers for the RF-DETR one-off/training scripts in this directory
(train_detector_rfdetr.py, tune_rfdetr_optuna.py, train_rfdetr_optuna_best.py,
train_rfdetr_inpainted.py).

None of these scripts import from src/perception (that package tree pulls in
cv2 and other main-venv-only deps incompatible with the isolated
.venv-rfdetr-train), so they were each keeping a private, byte-identical copy
of read_final_metrics()/the production-checkpoint guard rather than sharing
one -- there's no actual venv-isolation reason for that, since this module
itself only uses stdlib (csv/shutil/pathlib/logging).
"""
from __future__ import annotations

import csv
import logging
import shutil
from pathlib import Path


def read_final_metrics(out_dir: Path) -> tuple[float, float]:
    """Best (not merely last) val/mAP_50 and val/mAP_50_95 from PyTorch
    Lightning's CSVLogger output. Most rows are train-only (val columns
    blank -- validation runs less often than train steps), so this scans
    every row rather than just the last one. NaN, NaN if the file is
    missing or the expected columns aren't present."""
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
            if best50 != best50 or m50 > best50:  # first hit, or a new best
                best50, best5095 = m50, m5095
    return best50, best5095


def copy_best_checkpoint(out_dir: Path, logger: logging.Logger) -> Path | None:
    """Copy RF-DETR's canonical checkpoint_best_total.pth to best.pt (the
    convention _scan_checkpoints() and this project's other tooling expect).
    Returns the destination path, or None if the source was never produced."""
    best_src = out_dir / "checkpoint_best_total.pth"
    best_dest = out_dir / "best.pt"
    if best_src.exists():
        shutil.copy2(str(best_src), str(best_dest))
        logger.info("Best checkpoint -> %s", best_dest)
        return best_dest
    logger.warning("Expected checkpoint not found: %s", best_src)
    return None


def check_output_dir_safe(out_dir: Path, production_ckpt: Path, logger: logging.Logger) -> bool:
    """Refuse to run if out_dir would collide with the production checkpoint's
    own directory, or if out_dir already has content from a previous attempt.
    Returns True if it's safe to proceed."""
    if out_dir.resolve() == production_ckpt.parent.resolve():
        logger.error("Refusing to run: output dir would collide with the production checkpoint dir (%s)",
                     production_ckpt.parent)
        return False
    if out_dir.exists() and any(out_dir.iterdir()):
        logger.error("Refusing to run: %s already exists and is non-empty. "
                    "Delete it yourself first if you want to redo this run.", out_dir)
        return False
    return True


def snapshot_production_checkpoint(production_ckpt: Path, logger: logging.Logger) -> int | None:
    """Returns the production checkpoint's current file size, or None (having
    already logged why) if it's missing -- call before training, pair with
    verify_production_checkpoint_unchanged() after."""
    if not production_ckpt.exists():
        logger.error("Production checkpoint missing, aborting as a precaution: %s", production_ckpt)
        return None
    return production_ckpt.stat().st_size


def verify_production_checkpoint_unchanged(production_ckpt: Path, size_before: int, logger: logging.Logger) -> bool:
    """Pair with snapshot_production_checkpoint(). Returns True if unchanged."""
    if production_ckpt.stat().st_size != size_before:
        logger.error("Production checkpoint changed size during this run -- INVESTIGATE IMMEDIATELY: %s",
                     production_ckpt)
        return False
    logger.info("Verified production checkpoint untouched: %s", production_ckpt)
    return True

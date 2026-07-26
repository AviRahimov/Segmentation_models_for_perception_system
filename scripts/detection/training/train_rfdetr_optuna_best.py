#!/usr/bin/env python3
"""Phase 5 — retrain rfdetr-m on the unmodified Detection_Dataset_hardneg
using tune_rfdetr_optuna.py's winning hyperparameters, nothing else changed.

Why a separate script (not reusing train_detector_rfdetr.py's survey/CLI):
this is a single deliberate retrain, not a sweep or interactive queue --
loads weights/detection/rfdetr-m/optuna/best_params.json directly and calls
the public RFDETR.train() API the same way _run_rfdetr_variant() in
train_detector_rfdetr.py does (dataset_dir/dataset_file/output_dir/epochs/
batch_size="auto"/early_stopping=True/progress_bar="tqdm"), just with the
Optuna-found hyperparameters layered on top instead of recipe defaults.

Output directory safety: writes to a NEW directory
(detection_dataset_hardneg/conservative_aug_optuna), deliberately never the
production path (detection_dataset_hardneg/conservative_aug/best.pt) --
refuses to run if asked to write there, and refuses if the target directory
already has content (no silent overwrite of a previous Phase 5 attempt --
delete it yourself first if you want to redo the run).

Usage
-----
    python scripts/detection/training/train_rfdetr_optuna_best.py
    # Try a different trial from the sweep instead of the #1 winner (e.g.
    # because the winner regressed on FP/FN despite a higher mAP50):
    python scripts/detection/training/train_rfdetr_optuna_best.py \\
        --params-json weights/detection/rfdetr-m/optuna/trial_010_params.json \\
        --recipe-suffix conservative_aug_optuna_trial10
"""
from __future__ import annotations

import argparse
import json
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
logger = logging.getLogger("train_rfdetr_optuna_best")

_DATASET_DIR = _ROOT / "datasets/Detection_Dataset_hardneg"
_PRODUCTION_CKPT = _ROOT / "weights/detection/rfdetr-m/detection_dataset_hardneg/conservative_aug/best.pt"

_AUG_PRESET_NAMES = {"none": None}  # populated below with the real objects


def _read_final_metrics(out_dir: Path) -> tuple[float, float]:
    """Identical to train_detector_rfdetr.py's helper -- duplicated rather
    than imported, matching that script's own zero-cross-import convention."""
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--params-json", type=Path,
                   default=_ROOT / "weights/detection/rfdetr-m/optuna/best_params.json",
                   help="Which trial's hyperparameters to retrain with (default: the #1 sweep winner)")
    p.add_argument("--recipe-suffix", default="conservative_aug_optuna",
                   help="Output dir name under detection_dataset_hardneg/ -- must differ per trial "
                        "tried, and must never be 'conservative_aug' (the production recipe name)")
    return p.parse_args()


def main() -> int:
    from rfdetr.datasets.aug_config import AUG_AGGRESSIVE, AUG_CONSERVATIVE

    _AUG_PRESET_NAMES.update({"conservative": AUG_CONSERVATIVE, "aggressive": AUG_AGGRESSIVE})

    args = parse_args()
    out_dir = _ROOT / "weights/detection/rfdetr-m/detection_dataset_hardneg" / args.recipe_suffix

    if args.recipe_suffix == "conservative_aug" or out_dir.resolve() == _PRODUCTION_CKPT.parent.resolve():
        logger.error("Refusing to run: --recipe-suffix would collide with the production checkpoint dir (%s)",
                     _PRODUCTION_CKPT.parent)
        return 1
    if out_dir.exists() and any(out_dir.iterdir()):
        logger.error("Refusing to run: %s already exists and is non-empty. "
                    "Delete it yourself first if you want to redo this run.", out_dir)
        return 1
    if not _PRODUCTION_CKPT.exists():
        logger.error("Production checkpoint missing, aborting as a precaution: %s", _PRODUCTION_CKPT)
        return 1
    prod_size_before = _PRODUCTION_CKPT.stat().st_size

    best = json.loads(args.params_json.read_text())
    params = dict(best["params"])
    aug_name = params.pop("aug_preset")
    params["aug_config"] = _AUG_PRESET_NAMES[aug_name]
    logger.info("Phase 5 retrain -- trial #%d's hyperparameters (mAP50=%.4f during the sweep): %s",
                best["trial"], best["map50"], {k: v for k, v in params.items() if k != "aug_config"})
    logger.info("aug_preset=%s -> %s", aug_name, "None (disabled)" if params["aug_config"] is None else "preset dict")

    import rfdetr as rfdetr_pkg

    model = rfdetr_pkg.RFDETRMedium()
    logger.info("Training rfdetr-m on %s -> %s", _DATASET_DIR.relative_to(_ROOT), out_dir.relative_to(_ROOT))
    model.train(
        dataset_dir=str(_DATASET_DIR),
        dataset_file="yolo",
        output_dir=str(out_dir),
        epochs=100,
        batch_size="auto",
        seed=42,
        progress_bar="tqdm",
        early_stopping=True,
        **params,
    )

    best_src = out_dir / "checkpoint_best_total.pth"
    best_dest = out_dir / "best.pt"
    if best_src.exists():
        shutil.copy2(str(best_src), str(best_dest))
        logger.info("Best checkpoint -> %s", best_dest)
    else:
        logger.warning("Expected checkpoint not found: %s", best_src)

    map50, map5095 = _read_final_metrics(out_dir)
    logger.info("Phase 5 retrain done -- mAP50=%.4f  mAP50-95=%.4f", map50, map5095)

    if _PRODUCTION_CKPT.stat().st_size != prod_size_before:
        logger.error("Production checkpoint changed size during this run -- INVESTIGATE IMMEDIATELY: %s",
                     _PRODUCTION_CKPT)
        return 1
    logger.info("Verified production checkpoint untouched: %s", _PRODUCTION_CKPT)

    sys.path.insert(0, str(_ROOT / "scripts" / "detection" / "training"))
    from _survey_common import _log_experiment  # noqa: E402

    _log_experiment({
        "mode": "optuna_phase5_retrain", "arch_family": "rfdetr", "model": "rfdetr-m",
        "dataset": "Detection_Dataset_hardneg", "recipe": args.recipe_suffix,
        "run_dir": str(out_dir.relative_to(_ROOT)),
        "epochs": 100, "batch_size": "auto", "seed": 42, "init": "coco",
        "optuna_trial": best["trial"], "optuna_sweep_mAP50": best["map50"],
        "optuna_params": {k: v for k, v in params.items() if k != "aug_config"} | {"aug_preset": aug_name},
        "train_mAP50": round(map50, 5) if map50 == map50 else None,
        "train_mAP50_95": round(map5095, 5) if map5095 == map5095 else None,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

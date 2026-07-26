#!/usr/bin/env python3
"""One-shot setup: copy Detection_Dataset_hardneg -> Detection_Dataset_hardneg_inpainted.

Why a full copy instead of promoting inpainted negatives straight into
Detection_Dataset_hardneg: that dataset is what the current production
rfdetr-m checkpoint was trained on. Promoting into it directly would mean
any future retrain (deliberate or accidental) uses a dataset that no
longer matches the checkpoint's own provenance -- confirmed to have
already happened once (see promote_inpainted_negatives.py's docstring).
A separate copy keeps Detection_Dataset_hardneg exactly as it was, while
giving Phase 6 (retrain rfdetr-m with the promoted inpainted negatives,
compare against the untouched baseline) a real, independent dataset to
add examples into.

Copies train/, valid/, test/, and data.yaml verbatim -- data.yaml's paths
(../train/images etc.) are relative to its own location, so they resolve
correctly in the copy without any editing.

Usage
-----
    python scripts/detection/tools/init_inpainted_dataset.py

Refuses to run if the destination already exists (no silent overwrite --
delete it yourself first if you want to redo this from scratch).
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("init_inpainted_dataset")

_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, default=_ROOT / "datasets/Detection_Dataset_hardneg")
    p.add_argument("--dest", type=Path, default=_ROOT / "datasets/Detection_Dataset_hardneg_inpainted")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.src.exists():
        logger.error("Source dataset not found: %s", args.src)
        return 1
    if args.dest.exists():
        logger.error("Refusing to run: %s already exists. Delete it yourself first if you want "
                    "to redo this from scratch.", args.dest)
        return 1

    logger.info("Copying %s -> %s ...", args.src, args.dest)
    shutil.copytree(args.src, args.dest)
    n_train = len(list((args.dest / "train" / "images").iterdir()))
    logger.info("Done. train/images now has %d file(s) (should match the source's count).", n_train)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

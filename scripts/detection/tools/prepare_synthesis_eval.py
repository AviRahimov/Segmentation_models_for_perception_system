"""Build an Ultralytics-compatible eval set from the synthetic dataset.

The synthetic data (datasets/Synthesis_Train/) uses its own class scheme
(0=Boulders, 1=Humans, 2=Cars, 3=Trees) and an ``img/`` directory name that
Ultralytics ``val()`` cannot resolve labels for (it requires ``images/``).

This script produces datasets/Synthesis_Eval/ with:

- ``images/``   — symlinks to the source pictures (or copies with --copy-images)
- ``labels/``   — labels remapped to the model scheme: Cars(2)->0 (Military
                  Vehicle), Humans(1)->1 (person); Boulders(0) and Trees(3)
                  dropped. Images left with no labels are kept as negatives.
- ``data.yaml`` — 2-class Ultralytics dataset config (val split only)

Usage
-----
    # All 355 images:
    python scripts/detection/tools/prepare_synthesis_eval.py

    # Random 50-image subset (deterministic via --seed):
    python scripts/detection/tools/prepare_synthesis_eval.py --n-samples 50

Then evaluate with:
    python scripts/detection/evaluation/compare_detection_models.py --mode table \\
        --data datasets/Synthesis_Eval/data.yaml --models pytorch:...
"""
from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("prepare_synthesis_eval")

# Synthetic class id -> model class id. Absent ids are dropped.
# Synthetic: 0=Boulders, 1=Humans, 2=Cars, 3=Trees
# Model:     0=Military Vehicle, 1=person
_CLASS_REMAP: dict[int, int] = {2: 0, 1: 1}
_MODEL_NAMES = ["Military Vehicle", "person"]

_IMG_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Remap synthetic labels to the model class scheme for evaluation"
    )
    p.add_argument("--source", default="datasets/Synthesis_Train",
                   help="Source dataset dir containing img/ and labels/")
    p.add_argument("--out", default="datasets/Synthesis_Eval",
                   help="Output dataset dir (recreated on every run)")
    p.add_argument("--n-samples", type=int, default=0, dest="n_samples",
                   help="Random subset size (0 = all images)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy-images", action="store_true", dest="copy_images",
                   help="Copy images instead of symlinking")
    return p.parse_args()


def remap_label_lines(text: str) -> tuple[list[str], Counter]:
    """Remap class ids in YOLO label lines; drop unmapped classes.

    Works for both bbox (5 fields) and polygon (class + 2N coords) lines —
    only the leading class id changes.
    """
    kept: list[str] = []
    seen: Counter = Counter()
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        seen[cid] += 1
        if cid not in _CLASS_REMAP:
            continue
        kept.append(" ".join([str(_CLASS_REMAP[cid])] + parts[1:]))
    return kept, seen


def main() -> int:
    args = parse_args()

    src = Path(args.source)
    if not src.is_absolute():
        src = _ROOT / src
    img_dir = src / "img"
    lbl_dir = src / "labels"
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        logger.error("Expected %s and %s to exist.", img_dir, lbl_dir)
        return 2

    out = Path(args.out)
    if not out.is_absolute():
        out = _ROOT / out
    if out.exists():
        shutil.rmtree(out)
    out_img = out / "images"
    out_lbl = out / "labels"
    out_img.mkdir(parents=True)
    out_lbl.mkdir(parents=True)

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in _IMG_SUFFIXES)
    if not images:
        logger.error("No images found in %s", img_dir)
        return 2

    if args.n_samples and args.n_samples < len(images):
        rng = random.Random(args.seed)
        images = sorted(rng.sample(images, args.n_samples), key=lambda p: p.name)

    totals: Counter = Counter()
    n_kept_lines = 0
    n_empty = 0
    n_missing_lbl = 0

    for img_path in images:
        # Image: symlink (default) or copy
        dest_img = out_img / img_path.name
        if args.copy_images:
            shutil.copy2(img_path, dest_img)
        else:
            dest_img.symlink_to(img_path.resolve())

        # Label: remap classes, drop unmapped
        src_lbl = lbl_dir / img_path.with_suffix(".txt").name
        kept: list[str] = []
        if src_lbl.exists():
            kept, seen = remap_label_lines(src_lbl.read_text())
            totals.update(seen)
        else:
            n_missing_lbl += 1
        if not kept:
            n_empty += 1
        n_kept_lines += len(kept)
        (out_lbl / src_lbl.name).write_text("\n".join(kept) + ("\n" if kept else ""))

    # data.yaml — val-only eval config in the model's class scheme
    names_yaml = "\n".join(f"  {i}: {n}" for i, n in enumerate(_MODEL_NAMES))
    (out / "data.yaml").write_text(
        f"path: {out}\n"
        f"train: images  # unused — eval-only dataset\n"
        f"val: images\n"
        f"nc: {len(_MODEL_NAMES)}\n"
        f"names:\n{names_yaml}\n"
    )

    logger.info("Eval set written to %s", out)
    logger.info("  images: %d (%s)", len(images),
                "copies" if args.copy_images else "symlinks")
    logger.info("  label lines kept: %d  (Cars->0: %d, Humans->1: %d)",
                n_kept_lines, totals[2], totals[1])
    logger.info("  dropped: Boulders=%d, Trees=%d", totals[0], totals[3])
    logger.info("  images with no remaining labels (negatives): %d", n_empty)
    if n_missing_lbl:
        logger.warning("  images with no label file at all: %d", n_missing_lbl)
    logger.info("data.yaml -> %s", out / "data.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Remap the 12-class Kaggle military dataset to the pipeline's 2-class scheme.

Source: datasets/military_object_dataset_kaggle (train/val/test, YOLO format,
12 classes, stale /kaggle/input path in its yaml).

Mapping (user-decided):
    vehicle (0) <- military_tank(2), military_truck(3), military_vehicle(4)
    person  (1) <- camouflage_soldier(0), soldier(6), civilian(5)
    dropped     -> weapon(1), civilian_vehicle(7), military_artillery(8),
                   trench(9), military_aircraft(10), military_warship(11)

Images whose labels become empty are kept as background negatives using a
hard-negative-first policy (not a blind random sample):

    1. images that contained civilian_vehicle — direct hard negatives for the
       "civilian car flagged as Military Vehicle" failure mode (all kept)
    2. images that contained military_artillery — tank-confusable (all kept)
    3. remaining empty images (aircraft/warship/... only) — seeded random fill
       up to --neg-fraction of the positive image count

Usage
-----
    python scripts/detection/tools/prepare_military_kaggle.py
    python scripts/detection/tools/prepare_military_kaggle.py --neg-fraction 0.05
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
logger = logging.getLogger("prepare_military_kaggle")

# Source class id -> new class id. Absent ids are dropped.
_CLASS_REMAP: dict[int, int] = {
    2: 0,  # military_tank      -> Military Vehicle
    3: 0,  # military_truck     -> Military Vehicle
    4: 0,  # military_vehicle   -> Military Vehicle
    0: 1,  # camouflage_soldier -> person
    6: 1,  # soldier            -> person
    5: 1,  # civilian           -> person
}
_SOURCE_NAMES = {
    0: "camouflage_soldier", 1: "weapon", 2: "military_tank",
    3: "military_truck", 4: "military_vehicle", 5: "civilian",
    6: "soldier", 7: "civilian_vehicle", 8: "military_artillery",
    9: "trench", 10: "military_aircraft", 11: "military_warship",
}
_MODEL_NAMES = ["Military Vehicle", "person"]

# Hard-negative priority: source class ids whose presence makes an
# otherwise-empty image a valuable background example.
_HARD_NEG_PRIORITY = (7, 8)  # civilian_vehicle, military_artillery

_IMG_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")
_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Remap the Kaggle military dataset to 2 classes with hard negatives"
    )
    p.add_argument("--source", default="datasets/military_object_dataset_kaggle")
    p.add_argument("--out", default="datasets/military_kaggle_2class",
                   help="Output dataset dir (recreated on every run)")
    p.add_argument("--neg-fraction", type=float, default=0.08, dest="neg_fraction",
                   help="Background negatives kept, as a fraction of positive images")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy-images", action="store_true", dest="copy_images",
                   help="Copy images instead of symlinking")
    return p.parse_args()


def remap_label_text(text: str) -> tuple[list[str], set[int], Counter]:
    """-> (remapped lines, original class ids present, per-class instance counts)."""
    kept: list[str] = []
    present: set[int] = set()
    counts: Counter = Counter()
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        present.add(cid)
        counts[cid] += 1
        if cid in _CLASS_REMAP:
            kept.append(" ".join([str(_CLASS_REMAP[cid])] + parts[1:]))
    return kept, present, counts


def _place_image(src_img: Path, dest_img: Path, copy: bool) -> None:
    if copy:
        shutil.copy2(src_img, dest_img)
    else:
        dest_img.symlink_to(src_img.resolve())


def process_split(split_src: Path, split_out: Path, neg_fraction: float,
                  seed: int, copy_images: bool) -> dict:
    img_dir = split_src / "images"
    lbl_dir = split_src / "labels"
    out_img = split_out / "images"
    out_lbl = split_out / "labels"
    out_img.mkdir(parents=True)
    out_lbl.mkdir(parents=True)

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in _IMG_SUFFIXES)

    positives: list[tuple[Path, list[str]]] = []
    # empty-after-remap images, bucketed by negative priority
    hard_negs: list[Path] = []
    soft_negs: list[Path] = []
    instance_counts: Counter = Counter()
    n_missing_lbl = 0

    for img_path in images:
        src_lbl = lbl_dir / img_path.with_suffix(".txt").name
        if not src_lbl.exists():
            n_missing_lbl += 1
            continue
        kept, present, counts = remap_label_text(src_lbl.read_text())
        instance_counts.update(counts)
        if kept:
            positives.append((img_path, kept))
        elif present & set(_HARD_NEG_PRIORITY):
            hard_negs.append(img_path)
        else:
            soft_negs.append(img_path)

    # Negative budget: hard negatives always in; soft negatives fill the rest.
    neg_budget = int(len(positives) * neg_fraction)
    chosen_negs = list(hard_negs)
    remaining = max(0, neg_budget - len(chosen_negs))
    if remaining and soft_negs:
        rng = random.Random(seed)
        chosen_negs += sorted(rng.sample(soft_negs, min(remaining, len(soft_negs))),
                              key=lambda p: p.name)

    for img_path, kept_lines in positives:
        _place_image(img_path, out_img / img_path.name, copy_images)
        (out_lbl / img_path.with_suffix(".txt").name).write_text(
            "\n".join(kept_lines) + "\n"
        )
    for img_path in chosen_negs:
        _place_image(img_path, out_img / img_path.name, copy_images)
        (out_lbl / img_path.with_suffix(".txt").name).write_text("")

    return {
        "positives": len(positives),
        "hard_negs": len(hard_negs),
        "soft_negs_kept": len(chosen_negs) - len(hard_negs),
        "empty_dropped": len(soft_negs) - (len(chosen_negs) - len(hard_negs)),
        "missing_labels": n_missing_lbl,
        "instances": instance_counts,
    }


def main() -> int:
    args = parse_args()

    src = Path(args.source)
    if not src.is_absolute():
        src = _ROOT / src
    out = Path(args.out)
    if not out.is_absolute():
        out = _ROOT / out

    missing = [s for s in _SPLITS if not (src / s / "images").is_dir()]
    if missing:
        logger.error("Source splits missing under %s: %s", src, missing)
        return 2

    if out.exists():
        shutil.rmtree(out)

    total_instances: Counter = Counter()
    for split in _SPLITS:
        stats = process_split(src / split, out / split, args.neg_fraction,
                              args.seed, args.copy_images)
        total_instances.update(stats["instances"])
        logger.info(
            "%-5s  positives=%d  negatives=%d (hard=%d, sampled=%d)  "
            "empty dropped=%d  missing labels=%d",
            split, stats["positives"],
            stats["hard_negs"] + stats["soft_negs_kept"],
            stats["hard_negs"], stats["soft_negs_kept"],
            stats["empty_dropped"], stats["missing_labels"],
        )

    names_yaml = "\n".join(f"  {i}: {n}" for i, n in enumerate(_MODEL_NAMES))
    (out / "data.yaml").write_text(
        f"# Generated by prepare_military_kaggle.py from {src.name}\n"
        f"path: {out.resolve()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"test: test/images\n"
        f"nc: {len(_MODEL_NAMES)}\n"
        f"names:\n{names_yaml}\n"
    )

    logger.info("")
    logger.info("Instance remap summary (all splits):")
    for cid in sorted(total_instances):
        action = (f"-> {_MODEL_NAMES[_CLASS_REMAP[cid]]}"
                  if cid in _CLASS_REMAP else "dropped")
        logger.info("  %2d %-20s %6d  %s",
                    cid, _SOURCE_NAMES.get(cid, "?"), total_instances[cid], action)
    logger.info("")
    logger.info("Dataset written to %s (images %s)", out,
                "copied" if args.copy_images else "symlinked")
    logger.info("data.yaml -> %s", out / "data.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

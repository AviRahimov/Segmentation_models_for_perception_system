#!/usr/bin/env python3
"""Clip-grouped train/val split for the promoted Gaza-domain image set.

A per-frame random split would leak: the 225 promoted images are extracted
from only 10 source video clips, so adjacent frames from the same clip are
near-duplicates. Holding out individual frames would let near-identical
images sit in both train and val, inflating val mIoU artificially. This
script holds out whole clips instead.

Clip grouping is recovered from the promoted filename convention (Roboflow
export): ``<clip-name>_<NNN>_png.rf.<hash>.png``.

Selection is deterministic (no RNG): clips are sorted ascending by frame
count and greedily added to the val set until the cumulative fraction
reaches --val-frac -- no need for a --seed to reproduce this split.

Usage
-----
    python scripts/segmentation/tools/split_gaza_domain.py \\
        --root datasets/segmentation/gaza_domain
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

_STEM_RE = re.compile(r"^(?P<clip>.+)_(?P<frame>\d+)_png\.rf\.[A-Za-z0-9]+$")


def group_by_clip(image_stems: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for stem in image_stems:
        m = _STEM_RE.match(stem)
        if not m:
            raise ValueError(f"Stem does not match the expected '<clip>_<NNN>_png.rf.<hash>' convention: {stem}")
        groups[m.group("clip")].append(stem)
    return groups


def choose_val_clips(groups: dict[str, list[str]], val_frac: float) -> set[str]:
    total = sum(len(v) for v in groups.values())
    target = val_frac * total
    val_clips: set[str] = set()
    cum = 0
    for clip, stems in sorted(groups.items(), key=lambda kv: len(kv[1])):
        val_clips.add(clip)
        cum += len(stems)
        if cum >= target:
            break
    return val_clips


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="datasets/segmentation/gaza_domain")
    p.add_argument("--val-frac", type=float, default=0.2)
    args = p.parse_args()

    root = Path(args.root)
    images_dir = root / "images"
    stems = sorted(f.stem for f in images_dir.glob("*.png"))
    if not stems:
        raise SystemExit(f"No promoted images found under {images_dir}")

    groups = group_by_clip(stems)
    val_clips = choose_val_clips(groups, args.val_frac)

    train_stems, val_stems = [], []
    for clip, clip_stems in groups.items():
        (val_stems if clip in val_clips else train_stems).extend(clip_stems)
    train_stems.sort()
    val_stems.sort()

    splits_dir = root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "train.txt").write_text("\n".join(train_stems) + "\n")
    (splits_dir / "val.txt").write_text("\n".join(val_stems) + "\n")

    total = len(train_stems) + len(val_stems)
    print(f"{len(groups)} clips, {total} images total")
    print(f"val:   {len(val_stems)} images ({len(val_stems) / total:.1%}) from {len(val_clips)} clips:")
    for clip in sorted(val_clips):
        print(f"    {len(groups[clip]):3d}  {clip}")
    print(f"train: {len(train_stems)} images ({len(train_stems) / total:.1%}) from {len(groups) - len(val_clips)} clips")
    print(f"Wrote {splits_dir / 'train.txt'} and {splits_dir / 'val.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

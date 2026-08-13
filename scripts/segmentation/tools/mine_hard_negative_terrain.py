#!/usr/bin/env python3
"""Mine ORFD training images for the "large rock/hillside touching the road"
pattern and emit per-image oversampling weights.

Why: the Gaza-only fine-tuned checkpoints visibly over-segment large rocks/
dunes/hillsides adjacent to a road as traversable on non-Gaza footage
(confirmed via rendered video frame inspection) -- the 225 Gaza training
images are almost all flat rubble/urban terrain with few or no "boulder next
to a passable road" examples, so a Gaza-only or lightly-mixed fine-tune has
little signal to keep that pattern sharp. This script scans ORFD's own
existing labels (no new labeling needed) for images that already contain
that exact geometric pattern and up-weights them for the corrective joint
ORFD+Gaza training run (see train_orfd.py's --hard-negative-weights).

Detection: for each label mask, find connected components per class; an
image is flagged if a large non_traversable (class 0) blob is geometrically
adjacent (via dilation) to a large traversable (class 1) blob -- "large" and
"adjacent" both configurable, defaulting to values chosen for 512x512 masks.

Output: a flat JSON {image_stem: weight} (1.0 normally, --hard-weight for a
flagged image), keyed by ORFDDataset.pairs' image path stem so it's directly
usable from train_orfd.py without re-deriving pairing logic, and robust to
the exact indices/order ORFDDataset happens to build internally.

Usage
-----
    python scripts/segmentation/tools/mine_hard_negative_terrain.py \\
        --data datasets/segmentation/ORFD --split training \\
        --out weights/segmentation/orfd/hard_negative_weights_orfd_train.json \\
        --preview-dir /tmp/hard_negative_preview --preview-n 30
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))

from perception.datasets.orfd_torch import ORFDDataset, _remap_label  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("mine_hard_negative_terrain")

NON_TRAVERSABLE, TRAVERSABLE = 0, 1


def _large_blob_masks(label: np.ndarray, class_id: int, min_area_frac: float) -> list[np.ndarray]:
    """Connected components of (label == class_id) with area >= min_area_frac * label.size."""
    binary = (label == class_id).astype(np.uint8)
    n_labels, comp = cv2.connectedComponents(binary, connectivity=8)
    min_area = min_area_frac * label.size
    blobs = []
    for comp_id in range(1, n_labels):  # 0 is background
        mask = comp == comp_id
        if mask.sum() >= min_area:
            blobs.append(mask)
    return blobs


def is_rock_near_road(label: np.ndarray, min_area_frac: float = 0.05, dilation_px: int = 15) -> bool:
    """True iff a large non_traversable blob is geometrically adjacent (via
    dilation) to a large traversable blob -- the rock/hillside-next-to-road
    pattern the Gaza-tuned checkpoints over-predict traversable on."""
    nontrav_blobs = _large_blob_masks(label, NON_TRAVERSABLE, min_area_frac)
    trav_blobs = _large_blob_masks(label, TRAVERSABLE, min_area_frac)
    if not nontrav_blobs or not trav_blobs:
        return False

    trav_union = np.zeros(label.shape, dtype=np.uint8)
    for m in trav_blobs:
        trav_union |= m.astype(np.uint8)
    kernel = np.ones((dilation_px, dilation_px), np.uint8)
    trav_dilated = cv2.dilate(trav_union, kernel).astype(bool)

    return any((m & trav_dilated).any() for m in nontrav_blobs)


def mine_weights(
    data_root: str,
    split: str,
    min_area_frac: float,
    dilation_px: int,
    hard_weight: float,
    preview_dir: Path | None = None,
    preview_n: int = 30,
) -> dict[str, float]:
    ds = ORFDDataset(data_root, split=split, augment=False)
    weights: dict[str, float] = {}
    n_hard = 0
    previewed = 0

    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    for img_path, gt_path in ds.pairs:
        gt_raw = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt_raw is None:
            logger.warning("Could not load %s -- skipping (weight 1.0)", gt_path)
            weights[img_path.stem] = 1.0
            continue
        label = _remap_label(gt_raw)
        hard = is_rock_near_road(label, min_area_frac, dilation_px)
        weights[img_path.stem] = hard_weight if hard else 1.0
        if hard:
            n_hard += 1
            if preview_dir is not None and previewed < preview_n:
                _save_preview(img_path, label, preview_dir, min_area_frac, dilation_px)
                previewed += 1

    logger.info("%d/%d images flagged as rock-near-road hard negatives (weight %.1f)",
                n_hard, len(weights), hard_weight)
    return weights


def _save_preview(img_path: Path, label: np.ndarray, preview_dir: Path,
                   min_area_frac: float, dilation_px: int) -> None:
    """Overlay: red = non_traversable blobs, green = traversable blobs,
    yellow = the dilated traversable region actually tested for overlap --
    lets a human sanity-check the thresholds before trusting them."""
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return
    h, w = label.shape
    bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)

    trav_blobs = _large_blob_masks(label, TRAVERSABLE, min_area_frac)
    trav_union = np.zeros(label.shape, dtype=np.uint8)
    for m in trav_blobs:
        trav_union |= m.astype(np.uint8)
    trav_dilated = cv2.dilate(trav_union, np.ones((dilation_px, dilation_px), np.uint8))

    overlay = bgr.copy()
    overlay[label == NON_TRAVERSABLE] = (0, 0, 255)   # red, BGR
    overlay[trav_union.astype(bool)] = (0, 200, 0)     # green
    overlay[(trav_dilated.astype(bool)) & ~trav_union.astype(bool)] = (0, 255, 255)  # yellow ring
    blended = cv2.addWeighted(bgr, 0.4, overlay, 0.6, 0.0)
    cv2.imwrite(str(preview_dir / f"{img_path.stem}.png"), blended)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="datasets/segmentation/ORFD")
    p.add_argument("--split", default="training")
    p.add_argument("--min-area-frac", type=float, default=0.05,
                   help="Minimum fraction of frame area for a blob to count as 'large' (default 0.05).")
    p.add_argument("--dilation-px", type=int, default=15,
                   help="Dilation radius (px) used to test rock/road adjacency (default 15).")
    p.add_argument("--hard-weight", type=float, default=3.0,
                   help="Sample weight for flagged images (default 3.0, matches GazaDomainDataset's "
                        "existing rare-pattern up-weighting convention).")
    p.add_argument("--out", default="weights/segmentation/orfd/hard_negative_weights_orfd_train.json")
    p.add_argument("--preview-dir", default=None,
                   help="Optional directory to dump overlay images for the first --preview-n flagged "
                        "samples, for a manual sanity check of the thresholds before trusting them.")
    p.add_argument("--preview-n", type=int, default=30)
    args = p.parse_args()

    data_root = Path(args.data)
    if not data_root.is_absolute():
        data_root = _ROOT / data_root

    preview_dir = Path(args.preview_dir) if args.preview_dir else None

    weights = mine_weights(
        str(data_root), args.split, args.min_area_frac, args.dilation_px,
        args.hard_weight, preview_dir, args.preview_n,
    )

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(weights, indent=2))
    logger.info("Wrote %s", out_path)
    if preview_dir is not None:
        logger.info("Preview overlays written to %s -- inspect before trusting these thresholds.", preview_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

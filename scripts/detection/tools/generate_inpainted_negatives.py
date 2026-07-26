#!/usr/bin/env python3
"""Phase 3 — inpainting-based hard-negative generation (pilot batch), LaMa backend.

Why: TIDE (tide_analysis.py) showed background error (confident detections on
pure background) is a real chunk of rfdetr-m's false positives, and the
Optuna sweep (Phase 4/5) showed that chasing mAP50 alone makes this WORSE, not
better. A complementary, orthogonal fix: manufacture real-background hard
negatives "for free" from already-labeled training images -- remove the
labeled object(s) via inpainting (context-aware fill, NOT a naive blackout
rectangle, which would just teach the model to recognize a flat-color-block
artifact instead of genuine background rejection) and keep the surrounding
real scene.

Uses LaMa (`simple-lama-inpainting`, WACV 2022) rather than a generative
diffusion model deliberately: LaMa is a feed-forward, non-text-conditioned
GAN that propagates real neighboring texture into the masked hole -- it
cannot hallucinate a NEW person/vehicle into the hole the way a diffusion
model could, which would be actively counterproductive here (an unlabeled
positive mislabeled as background). Runs in its own venv (.venv-inpaint) to
keep torch/opencv version churn for this one-off tool isolated from both the
main venv and .venv-rfdetr-train.

First-round manual review verdict: LaMa handles person removal well but
struggles on Military Vehicle removal (large masks, textured/hazy
backgrounds) -- see generate_inpainted_negatives_iopaint.py for the
follow-up comparison against ZITS (after MAT/FcF/SDXL/PowerPaint were also
tried and rejected) on the identical sampled images and masks (shared via
_inpaint_common.py).

Per sampled image, builds up to 3 variants depending on which classes are
actually present (skips a variant that would be a no-op):
  - both_removed: mask covers every GT box -> new label file is empty.
  - vehicle_only_removed: mask covers only Military Vehicle boxes -> keeps
    any person lines from the original label.
  - person_only_removed: mask covers only person boxes -> keeps any vehicle
    lines from the original label.

This step ONLY writes to datasets/Detection_Dataset_inpaint_review/ -- it
never touches the real training set (Detection_Dataset_hardneg). Promotion
of specific approved candidates into the training set is a separate,
deliberate step (promote_inpainted_negatives.py) after you've reviewed the
_preview/ panels by hand.

Usage
-----
    .venv-inpaint/bin/python3 scripts/detection/tools/generate_inpainted_negatives.py
    .venv-inpaint/bin/python3 scripts/detection/tools/generate_inpainted_negatives.py --n-images 18 --seed 42

Output: datasets/Detection_Dataset_inpaint_review/<variant>/{images,labels}/
        datasets/Detection_Dataset_inpaint_review/_preview/<variant>__<stem>.jpg
        (3-panel: [original | mask overlay | inpainted result])
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from simple_lama_inpainting import SimpleLama

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _inpaint_common import (  # noqa: E402
    SRC_DIR, build_mask, build_variants, filter_surviving_kept_lines, parse_label_lines,
    preview_panel, sample_images,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("generate_inpainted_negatives")

_ROOT = Path(__file__).resolve().parents[3]
_OUT_ROOT = _ROOT / "datasets/Detection_Dataset_inpaint_review"
_PAD_MODULO = 8  # must match simple_lama_inpainting.utils.util.prepare_img_and_mask's default


def _run_lama(lama: SimpleLama, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pad to a multiple of 8 and crop back to the original size ourselves.

    simple-lama-inpainting's own prepare_img_and_mask() (utils/util.py) pads
    both the image and mask up to a multiple of 8 before running the model,
    but SimpleLama.__call__ returns that model output AS-IS -- it never crops
    back to the input size. For any image whose height or width isn't already
    a multiple of 8 (most real photos), the returned array comes back a few
    pixels larger with no error, no warning, and no shape check. Confirmed by
    reading the library source directly and reproducing it: for a 408x612
    image, the returned array was 408x616. Padding/cropping it ourselves here
    keeps the output size correct and lets us verify (empirically checked
    during development: diff==0 pixel-for-pixel outside the mask, real diff
    inside it) that this is the only issue -- the underlying inpainting
    itself is correct once the size bookkeeping is right."""
    h, w = rgb.shape[:2]
    ph = h if h % _PAD_MODULO == 0 else (h // _PAD_MODULO + 1) * _PAD_MODULO
    pw = w if w % _PAD_MODULO == 0 else (w // _PAD_MODULO + 1) * _PAD_MODULO
    rgb_pad = np.pad(rgb, ((0, ph - h), (0, pw - w), (0, 0)), mode="symmetric")
    mask_pad = np.pad(mask, ((0, ph - h), (0, pw - w)), mode="symmetric")
    result = np.array(lama(Image.fromarray(rgb_pad), Image.fromarray(mask_pad)))
    return result[:h, :w]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-images", type=int, default=18)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--src", type=Path, default=SRC_DIR)
    p.add_argument("--out", type=Path, default=_OUT_ROOT)
    p.add_argument("--relabel-only", action="store_true",
                   help="Skip model loading and image/preview writes entirely -- for candidates "
                        "already rendered on disk, just recompute each mask and overwrite its "
                        "label .txt with filter_surviving_kept_lines()'s corrected text. Use this "
                        "to pick up a fix to the kept-label logic without re-running any inpainting.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    lbl_dir = args.src / "labels"
    sampled = sample_images(args.src, args.n_images, args.seed)
    logger.info("Sampled %d source images (seed=%d)", len(sampled), args.seed)

    lama = None if args.relabel_only else SimpleLama()
    preview_dir = args.out / "_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    n_written = {"both_removed": 0, "vehicle_only_removed": 0, "person_only_removed": 0}
    n_relabeled = 0
    for img_path in sampled:
        label_path = lbl_dir / (img_path.stem + ".txt")
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            logger.warning("Could not read %s, skipping", img_path)
            continue
        h, w = bgr.shape[:2]
        lines = parse_label_lines(label_path, w, h)
        variants = build_variants(lines)

        bgr_rgb = None if args.relabel_only else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for variant, polys, kept_lines, protect_polys in variants:
            mask = build_mask(polys, h, w, protect_polys=protect_polys)
            kept_label_text = filter_surviving_kept_lines(mask, kept_lines)

            out_lbl_dir = args.out / variant / "labels"
            if args.relabel_only:
                out_label_path = out_lbl_dir / (img_path.stem + ".txt")
                if not out_label_path.exists():
                    continue  # this candidate was never generated -- nothing to relabel
                out_label_path.write_text(kept_label_text + ("\n" if kept_label_text else ""))
                n_relabeled += 1
                continue

            result_rgb = _run_lama(lama, bgr_rgb, mask)
            result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

            out_img_dir = args.out / variant / "images"
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_lbl_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_img_dir / img_path.name), result_bgr)
            (out_lbl_dir / (img_path.stem + ".txt")).write_text(
                kept_label_text + ("\n" if kept_label_text else ""))

            panel = preview_panel(bgr, mask, result_bgr, variant)
            cv2.imwrite(str(preview_dir / f"{variant}__{img_path.stem}.jpg"), panel)
            n_written[variant] += 1
            logger.info("  %s / %s -> %d region(s) removed", img_path.name, variant, len(polys))

    if args.relabel_only:
        logger.info("Relabeled %d candidate(s) in place.", n_relabeled)
        return 0

    logger.info("Done. Written per variant: %s", n_written)
    logger.info("Review panels: %s", preview_dir)
    logger.info("This is a PAUSE point -- inspect _preview/ by hand before running "
               "promote_inpainted_negatives.py on anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Phase 3 follow-up — same sampled images, same fixed masks (via
_inpaint_common.py) as the retired LaMa run (generate_inpainted_negatives.py,
since removed after ZITS won on quality), ZITS as the inpainting backend, so
the comparison isolates ONE variable (the model) at a time.

Why ZITS (after MAT/FcF/SDXL/PowerPaint were all tried and rejected): FLUX.1
Fill (web-researched) has the same "hallucinates unrelated objects" problem
as SDXL. SmartEraser (CVPR 2025, explicitly designed to prevent hallucination
during removal) requires a heavy custom conda/shell-script setup with
manually-downloaded weights -- not a fit for this project's "easy,
pip-installable" bar. PowerPaint (tried with both a generic fixed prompt and
an auto-captioned one) hallucinated new vehicle-like objects in both modes
despite its dedicated object-removal task mode -- dropped. ZITS is
non-generative and structure/edge-aware -- same "cannot invent a new object"
safety property as LaMa/MAT/FcF, a different architecture family from all of
them (predicts wireframe/edge structure before texture synthesis).

Runs with hd_strategy=HDStrategy.ORIGINAL -- IOPaint's default HDStrategy.CROP
has a confirmed bug for masks with multiple separate regions (e.g. a vehicle
+ a person kept-carved-out elsewhere in frame): boxes_from_mask() and the
per-box crop/paste-back logic in InpaintModel.__call__ silently left one
region completely untouched in a direct repro (both regions were in the
mask; only one got inpainted, no error raised). ORIGINAL runs the whole
image through in one pass, avoiding that code path entirely -- these images
are all well under 1000px, so this has no real performance cost.

IOPaint's own reference contract (iopaint/batch_processing.py, read directly
rather than guessed): ModelManager expects an RGB uint8 image, an 'L'-mode
(grayscale) uint8 mask, and returns a BGR result.

Usage
-----
    .venv-inpaint/bin/python3 scripts/detection/tools/generate_inpainted_negatives_iopaint.py

Output: datasets/detection/Detection_Dataset_inpaint_review_zits/<variant>/{images,labels}/
        datasets/detection/Detection_Dataset_inpaint_review_zits/_preview/<variant>__<stem>.jpg
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import torch
from iopaint.model.utils import torch_gc
from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy, InpaintRequest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _inpaint_common import (  # noqa: E402
    SRC_DIR, build_mask, build_variants, filter_surviving_kept_lines, parse_label_lines,
    preview_panel, sample_images,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("generate_inpainted_negatives_iopaint")

_ROOT = Path(__file__).resolve().parents[3]
_OUT_ROOT = _ROOT / "datasets/detection/Detection_Dataset_inpaint_review_zits"


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
    logger.info("Sampled %d source images (seed=%d) -- identical set to the LaMa run", len(sampled), args.seed)

    model_manager = None
    request = None
    if not args.relabel_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading zits on %s ...", device)
        model_manager = ModelManager(name="zits", device=device)
        request = InpaintRequest(hd_strategy=HDStrategy.ORIGINAL)

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

        rgb = None if args.relabel_only else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
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

            result_bgr = model_manager(rgb, mask, request)  # BGR per iopaint's own convention
            torch_gc()

            out_img_dir = args.out / variant / "images"
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_lbl_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_img_dir / img_path.name), result_bgr)
            (out_lbl_dir / (img_path.stem + ".txt")).write_text(
                kept_label_text + ("\n" if kept_label_text else ""))

            panel = preview_panel(bgr, mask, result_bgr, "zits")
            cv2.imwrite(str(preview_dir / f"{variant}__{img_path.stem}.jpg"), panel)
            n_written[variant] += 1
            logger.info("  %s / %s -> %d region(s) removed", img_path.name, variant, len(polys))

    if args.relabel_only:
        logger.info("Relabeled %d candidate(s) in place.", n_relabeled)
        return 0

    logger.info("Done. Written per variant: %s", n_written)
    logger.info("Review panels: %s", preview_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Promote reviewed SAM3 Gaza-domain labels into a real training dataset.

Mirrors scripts/detection/tools/promote_inpainted_negatives.py's structure
exactly -- same review-before-promote philosophy, same guard rationale.
Deliberately the ONLY step in this pipeline that writes to a dataset a
training script would actually consume. run_sam3_labeling.py and
rasterize_sam3_labels.py only ever write into the review area
(datasets/segmentation/gaza_domain_review/) -- this script does nothing
until you've looked through _preview/ by hand and deleted what you don't
trust.

Refuses to target datasets/segmentation/ORFD directly (pass --allow-orfd to
override) -- promoting straight into the dataset the production checkpoint
was trained/validated on is exactly the mistake this guard exists to
prevent (the detection pipeline hit this for real once: a partial promotion
run silently succeeded for only 25/266 approved candidates, then a routine
retrain used that half-contaminated dataset without anyone noticing until
much later). The intended destination is a SEPARATE dataset directory
(datasets/segmentation/gaza_domain/) so ORFD stays untouched and directly
comparable against.

Usage
-----
    # Recommended: promote everything still present in _preview/ (i.e.
    # whatever you didn't delete during review):
    python scripts/segmentation/tools/promote_gaza_labels.py --from-preview

    # Or name candidates explicitly:
    python scripts/segmentation/tools/promote_gaza_labels.py \\
        --candidates stock-footage-driving-around-devastated-city_001

    --dry-run prints the copy plan without copying.

Copies (never moves) the original image + rasterized label PNG + confidence
map from datasets/segmentation/gaza_domain_review/{images,labels}/ into
<dest>/{images,labels}/. Refuses to overwrite an already-promoted file.
Renders a post-promotion audit panel ([image | label-as-color-overlay])
from the final on-disk state, and appends a timestamped promotion log.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "scripts" / "segmentation"))
sys.path.insert(0, str(_HERE))

from _class_catalogues_loader import SAM3_FINEGRAINED_NAMES  # noqa: E402
from run_sam3_labeling import _PALETTE  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("promote_gaza_labels")

_REVIEW_ROOT = _ROOT / "datasets/segmentation/gaza_domain_review"
_GUARDED_DATASET = _ROOT / "datasets/segmentation/ORFD"
_IGNORE_INDEX = 255


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--review-root", type=Path, default=_REVIEW_ROOT,
                   help="Review area written by run_sam3_labeling.py + rasterize_sam3_labels.py")
    p.add_argument("--dest", type=Path, default=_ROOT / "datasets/segmentation/gaza_domain",
                   help="Dataset root to promote into. Refuses to target ORFD itself unless "
                        "--allow-orfd is also passed.")
    p.add_argument("--allow-orfd", action="store_true",
                   help="Required to let --dest point at datasets/segmentation/ORFD itself. "
                        "Think twice: that's the dataset the production checkpoint was trained on.")
    p.add_argument("--from-preview", action="store_true",
                   help="Derive the approved candidate list from whatever filenames currently "
                        "remain in <review-root>/_preview/ (your review = keep/delete pass), "
                        "instead of --list/--candidates.")
    p.add_argument("--list", type=Path, help="Text file, one stem per line (# comments allowed)")
    p.add_argument("--candidates", nargs="*", default=[], help="Stems given directly on the CLI")
    p.add_argument("--dry-run", action="store_true", help="Print what would be copied without copying")
    return p.parse_args()


def _draw_label_audit(image_path: Path, label_path: Path) -> "cv2.Mat | None":
    """[original image | label rendered as a color overlay], built from the
    just-written destination files so it always reflects exactly what ended
    up in the dataset."""
    image_bgr = cv2.imread(str(image_path))
    label_map = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
    if image_bgr is None or label_map is None:
        return None
    overlay = np.zeros_like(image_bgr)
    for idx in range(len(SAM3_FINEGRAINED_NAMES)):
        overlay[label_map == idx] = _PALETTE[idx % len(_PALETTE)]
    blended = cv2.addWeighted(overlay, 0.5, image_bgr, 0.5, 0)
    ignored_frac = float((label_map == _IGNORE_INDEX).mean())

    def _labeled(img, text):
        img = img.copy()
        cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    return cv2.hconcat([
        _labeled(image_bgr, "original"),
        _labeled(blended, f"promoted label (ignored={ignored_frac:.0%})"),
    ])


def main() -> int:
    args = parse_args()
    args.dest = args.dest if args.dest.is_absolute() else _ROOT / args.dest
    args.review_root = args.review_root if args.review_root.is_absolute() else _ROOT / args.review_root

    if args.dest.resolve() == _GUARDED_DATASET.resolve() and not args.allow_orfd:
        logger.error("Refusing to promote into %s without --allow-orfd -- that's the dataset the "
                    "production checkpoint was trained/validated on. Use a separate --dest "
                    "(default: datasets/segmentation/gaza_domain) instead, or pass --allow-orfd "
                    "if you really mean this.", _GUARDED_DATASET)
        return 1

    raw_candidates = list(args.candidates)
    if args.list:
        raw_candidates += args.list.read_text().splitlines()
    if args.from_preview:
        preview_dir = args.review_root / "_preview"
        if not preview_dir.exists():
            logger.error("No _preview/ folder found under %s", args.review_root)
            return 1
        raw_candidates += [p.stem for p in sorted(preview_dir.iterdir())]
    stems = [s.strip() for s in raw_candidates if s.strip() and not s.strip().startswith("#")]
    if not stems:
        logger.error("No candidates given -- pass --from-preview, --list FILE, or --candidates ...")
        return 1

    out_img_dir, out_lbl_dir = args.dest / "images", args.dest / "labels"
    audit_dir = args.dest / "_label_audit"
    if not args.dry_run:
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)
        audit_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines = []
    n_promoted = n_audited = 0
    for stem in stems:
        src_images = list((args.review_root / "images").glob(f"{stem}.*"))
        src_label = args.review_root / "labels" / f"{stem}.png"
        src_conf = args.review_root / "labels" / f"{stem}_conf.npy"
        if not src_images or not src_label.exists():
            logger.warning("Skipping %s -- source image or label not found under %s", stem, args.review_root)
            continue
        src_image = src_images[0]
        dest_image = out_img_dir / f"{stem}{src_image.suffix}"
        dest_label = out_lbl_dir / f"{stem}.png"
        if dest_image.exists() or dest_label.exists():
            logger.warning("Skipping %s -- already promoted (found %s)", stem, dest_image.name)
            continue
        if args.dry_run:
            logger.info("[dry-run] would copy %s -> %s", src_image.relative_to(_ROOT), dest_image.relative_to(_ROOT))
            n_promoted += 1
            continue

        shutil.copy2(src_image, dest_image)
        shutil.copy2(src_label, dest_label)
        if src_conf.exists():
            shutil.copy2(src_conf, out_lbl_dir / f"{stem}_conf.npy")
        manifest_lines.append(f"{datetime.now().isoformat(timespec='seconds')}  sam3  {stem}")
        n_promoted += 1
        logger.info("Promoted %s -> %s", stem, dest_image.relative_to(_ROOT))

        panel = _draw_label_audit(dest_image, dest_label)
        if panel is not None:
            cv2.imwrite(str(audit_dir / f"{stem}.jpg"), panel)
            n_audited += 1

    if manifest_lines:
        log_path = args.dest / "_gaza_promotions.log"
        with log_path.open("a") as f:
            f.write("\n".join(manifest_lines) + "\n")
        logger.info("Logged %d promotion(s) -> %s", len(manifest_lines), log_path)

    logger.info("%s %d/%d candidate(s)", "Would promote" if args.dry_run else "Promoted", n_promoted, len(stems))
    if not args.dry_run:
        logger.info("Wrote %d label-audit panel(s) -> %s", n_audited, audit_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

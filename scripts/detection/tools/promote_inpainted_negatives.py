#!/usr/bin/env python3
"""Phase 3 — promote approved inpainted candidates into a training set.

Deliberately the ONLY step that writes to a real training dataset --
everything upstream (generate_inpainted_negatives.py / _iopaint.py) only
ever writes to the separate review areas
(datasets/Detection_Dataset_inpaint_review[_zits]/). This script does
nothing until you've looked at _preview/ by hand and picked which
candidates are actually convincing.

Refuses to target datasets/Detection_Dataset_hardneg directly (pass
--dest explicitly if you really mean to, e.g. --dest
datasets/Detection_Dataset_hardneg) -- promoting straight into the
dataset a production checkpoint was trained on is exactly the mistake
this guard exists to prevent (confirmed to have happened once already:
a partial promotion run silently succeeded for only 25/266 approved
candidates because of the OLD hardcoded review-root bug below, then a
routine training survey retrained over the production checkpoint using
that half-contaminated dataset). The intended destination is a
SEPARATE dataset directory (see init_inpainted_dataset.py) so the
original stays untouched and comparable against.

Usage
-----
    # Recommended: promote everything still present in a source's
    # _preview/ folder (i.e. whatever you didn't delete during review)
    # into a fresh dataset copy:
    python scripts/detection/tools/promote_inpainted_negatives.py \\
        --source zits --from-preview --dest datasets/Detection_Dataset_hardneg_inpainted

    # Or name candidates explicitly, one per line in a file:
    python scripts/detection/tools/promote_inpainted_negatives.py \\
        --source zits --list approved.txt --dest datasets/Detection_Dataset_hardneg_inpainted

    # Or directly on the command line:
    python scripts/detection/tools/promote_inpainted_negatives.py \\
        --source zits --dest datasets/Detection_Dataset_hardneg_inpainted \\
        --candidates both_removed/getty_24a975e59efd_jpg.rf.qYwODMdIV7y2DXtW6B1F

Copies the image+label pair from
datasets/Detection_Dataset_inpaint_review[_zits]/<variant>/{images,labels}/
into <dest>/train/{images,labels}/, prefixing the filename with the
variant name so promoted files can never collide with an existing train
image and stay traceable back to which removal they came from. Refuses
to overwrite a file that's already been promoted.

Also renders a label-audit panel per promoted candidate --
[original source image | promoted image with its final label's polygons
drawn] -- into <dest>/_label_audit/<variant>__<stem>.jpg, so you can
visually confirm each promoted label actually matches what's still
visible in the image (this is what surfaces cases where a kept object
got silently erased by the removal mask -- see
_inpaint_common.filter_surviving_kept_lines).
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _inpaint_common import CLASS_NAMES, SRC_DIR, parse_label_lines  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("promote_inpainted_negatives")

_ROOT = Path(__file__).resolve().parents[3]
_SOURCES = {
    "lama": _ROOT / "datasets/Detection_Dataset_inpaint_review",
    "zits": _ROOT / "datasets/Detection_Dataset_inpaint_review_zits",
}
_GUARDED_DATASET = _ROOT / "datasets/Detection_Dataset_hardneg"
_VARIANTS = ("both_removed", "vehicle_only_removed", "person_only_removed")
_BOX_COLOR = (0, 200, 0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=list(_SOURCES), default="zits",
                   help="Which review folder to promote from (default: zits)")
    p.add_argument("--dest", type=Path, default=_ROOT / "datasets/Detection_Dataset_hardneg_inpainted",
                   help="Dataset root to promote into (its train/{images,labels}/ must already exist -- "
                        "see init_inpainted_dataset.py). Refuses to target Detection_Dataset_hardneg "
                        "itself unless --allow-hardneg is also passed.")
    p.add_argument("--allow-hardneg", action="store_true",
                   help="Required to let --dest point at Detection_Dataset_hardneg itself. "
                        "Think twice: that's the dataset the production checkpoint was trained on.")
    p.add_argument("--from-preview", action="store_true",
                   help="Derive the approved candidate list from whatever filenames currently "
                        "remain in <source>/_preview/ (i.e. your review = keep/delete pass), "
                        "instead of --list/--candidates.")
    p.add_argument("--list", type=Path, help="Text file, one candidate per line (# comments allowed)")
    p.add_argument("--candidates", nargs="*", default=[], help="Candidates given directly on the CLI")
    p.add_argument("--dry-run", action="store_true", help="Print what would be copied without copying")
    return p.parse_args()


def _parse_candidate(raw: str) -> tuple[str, str]:
    """Accepts either 'variant__stem[.ext]' (a preview filename) or
    'variant/stem'. Returns (variant, stem)."""
    raw = raw.strip()
    if not raw or raw.startswith("#"):
        return "", ""
    raw = Path(raw).stem if raw.endswith((".jpg", ".jpeg", ".png")) else raw
    if "__" in raw:
        variant, stem = raw.split("__", 1)
    elif "/" in raw:
        variant, stem = raw.split("/", 1)
    else:
        raise ValueError(f"Can't parse candidate {raw!r} -- expected 'variant__stem' or 'variant/stem'")
    if variant not in _VARIANTS:
        raise ValueError(f"Unknown variant {variant!r} in candidate {raw!r}; expected one of {_VARIANTS}")
    return variant, stem


def _draw_label_audit(review_root: Path, variant: str, stem: str, dest_label: Path) -> "cv2.Mat | None":
    """[original source image | promoted image with the JUST-WRITTEN label's
    polygons drawn]. Uses the promoted (post-copy) label file, not the
    in-review one, so this always reflects exactly what ended up in the
    dataset."""
    src_img = SRC_DIR / "images" / f"{stem}.jpg"
    promoted_img = review_root / variant / "images"
    promoted_candidates = list(promoted_img.glob(f"{stem}.*"))
    if not promoted_candidates:
        return None
    orig_bgr = cv2.imread(str(src_img))
    result_bgr = cv2.imread(str(promoted_candidates[0]))
    if orig_bgr is None or result_bgr is None:
        return None
    h, w = result_bgr.shape[:2]
    lines = parse_label_lines(dest_label, w, h)
    annotated = result_bgr.copy()
    for cid, _, poly in lines:
        name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else str(cid)
        cv2.polylines(annotated, [poly], isClosed=True, color=_BOX_COLOR, thickness=2)
        x, y = poly[:, 0].min(), poly[:, 1].min()
        cv2.putText(annotated, name, (int(x), max(14, int(y) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _BOX_COLOR, 2, cv2.LINE_AA)

    def _label(img, text):
        img = img.copy()
        cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
        cv2.putText(img, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    return cv2.hconcat([_label(orig_bgr, "original"), _label(annotated, f"{variant} (promoted label)")])


def main() -> int:
    args = parse_args()
    review_root = _SOURCES[args.source]
    args.dest = args.dest if args.dest.is_absolute() else _ROOT / args.dest

    if args.dest.resolve() == _GUARDED_DATASET.resolve() and not args.allow_hardneg:
        logger.error("Refusing to promote into %s without --allow-hardneg -- that's the dataset the "
                    "production checkpoint was trained on. Use init_inpainted_dataset.py to create a "
                    "separate copy instead (recommended), or pass --allow-hardneg if you really mean this.",
                    _GUARDED_DATASET)
        return 1

    raw_candidates = list(args.candidates)
    if args.list:
        raw_candidates += args.list.read_text().splitlines()
    if args.from_preview:
        preview_dir = review_root / "_preview"
        if not preview_dir.exists():
            logger.error("No _preview/ folder found under %s", review_root)
            return 1
        raw_candidates += [p.name for p in sorted(preview_dir.iterdir())]
    if not raw_candidates:
        logger.error("No candidates given -- pass --from-preview, --list FILE, or --candidates ...")
        return 1

    pairs = [c for c in (_parse_candidate(r) for r in raw_candidates) if c != ("", "")]
    if not pairs:
        logger.error("Nothing to promote after parsing.")
        return 1

    train_dir = args.dest / "train"
    out_img_dir, out_lbl_dir = train_dir / "images", train_dir / "labels"
    if not args.dry_run:
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = args.dest / "_label_audit"
    if not args.dry_run:
        audit_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines = []
    n_promoted = 0
    n_audited = 0
    for variant, stem in pairs:
        src_images = list((review_root / variant / "images").glob(f"{stem}.*"))
        src_label = review_root / variant / "labels" / f"{stem}.txt"
        if not src_images or not src_label.exists():
            logger.warning("Skipping %s/%s -- source image or label not found under %s", variant, stem, review_root)
            continue
        src_image = src_images[0]
        dest_stem = f"{variant}__{stem}"
        dest_image = out_img_dir / f"{dest_stem}{src_image.suffix}"
        dest_label = out_lbl_dir / f"{dest_stem}.txt"
        if dest_image.exists() or dest_label.exists():
            logger.warning("Skipping %s -- already promoted (found %s)", dest_stem, dest_image.name)
            continue
        if args.dry_run:
            logger.info("[dry-run] would copy %s -> %s", src_image.relative_to(_ROOT), dest_image.relative_to(_ROOT))
            n_promoted += 1
            continue
        shutil.copy2(src_image, dest_image)
        shutil.copy2(src_label, dest_label)
        manifest_lines.append(f"{datetime.now().isoformat(timespec='seconds')}  {args.source}  {dest_stem}")
        n_promoted += 1
        logger.info("Promoted %s -> %s", f"{variant}/{stem}", dest_image.relative_to(_ROOT))

        panel = _draw_label_audit(review_root, variant, stem, dest_label)
        if panel is not None:
            cv2.imwrite(str(audit_dir / f"{dest_stem}.jpg"), panel)
            n_audited += 1

    if manifest_lines:
        manifest = train_dir / "_inpainted_promotions.log"
        with manifest.open("a") as f:
            f.write("\n".join(manifest_lines) + "\n")
        logger.info("Logged %d promotion(s) -> %s", len(manifest_lines), manifest)

    logger.info("%s %d/%d candidate(s)", "Would promote" if args.dry_run else "Promoted", n_promoted, len(pairs))
    if not args.dry_run:
        logger.info("Wrote %d label-audit panel(s) -> %s", n_audited, audit_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Full-image FP review -- render every image containing a false positive
(at the deployment conf threshold) with GT, TP, and FP boxes all drawn on
the ORIGINAL, uncropped frame.

Why: leaderboard.py's --fp-gallery crops a tight context window around each
flagged box (see _save_fp_gallery), which is enough to judge "is this box on
a real object" but not "does this image actually have an unlabeled real
object nearby that GT missed" -- that requires seeing the whole frame and
every GT box in it side by side with what the model actually predicted.

Draws (BGR):
  - GT boxes: green, "GT <class>"
  - Correctly-matched predictions (TP) at conf>=0.40: blue, thin
  - False positives at conf>=0.40: red, thick + label -- the ones to inspect

Usage
-----
    python scripts/detection/evaluation/fp_full_image_review.py \\
        --weights weights/detection/rfdetr-m/detection_dataset_hardneg/conservative_aug/best.pt \\
                  weights/detection/rfdetr-m/detection_dataset_hardneg/conservative_aug_optuna/best.pt

Output: reports/detection/fp_full_images/<model_label>/<image_stem>.png --
one render per (model, image-with-at-least-one-FP-in-either-model) pair, so
the same source frame is directly comparable across models.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import cv2

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

from _ap_utils import (  # noqa: E402
    _match_class, collect_predictions, collect_predictions_rfdetr,
    is_rfdetr_checkpoint, load_rfdetr_for_eval, load_yolo_gts,
)
from leaderboard import _COLLAPSE
from leaderboard import _label as _leaderboard_label  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("fp_full_image_review")

_BENCHMARK_CLASSES = ["Military Vehicle", "person"]
_DEPLOY_CONF = 0.40


def _label(ckpt: Path) -> str:
    """Filesystem-safe variant of leaderboard.py's own _label() (this one's
    result becomes a directory name, so '/' must not survive)."""
    return _leaderboard_label(ckpt).replace("/", "_")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, nargs="+", type=Path)
    p.add_argument("--benchmark", default="datasets/detection/Detection_Dataset/test")
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default="0")
    p.add_argument("--out", type=Path, default=Path("reports/detection/fp_full_images"))
    return p.parse_args()


def _classify_boxes(preds, gts):
    """Per-image lists of (box, class_name, score, kind) for kind in {tp, fp}
    at the deployment conf threshold, via the same greedy IoU matching
    leaderboard.py/tide_analysis.py use (_match_class)."""
    by_image: dict[str, list[tuple]] = defaultdict(list)
    for cls in _BENCHMARK_CLASSES:
        cls_preds, tp_flags, _ = _match_class(preds, gts, cls, iou_thr=0.5, min_score=_DEPLOY_CONF)
        for p, is_tp in zip(cls_preds, tp_flags):
            by_image[p.image_id].append((p.box, p.class_name, p.score, "tp" if is_tp else "fp"))
    return by_image


def _gts_by_image(gts):
    out: dict[str, list] = defaultdict(list)
    for g in gts:
        out[g.image_id].append(g)
    return out


def _draw(img_path: str, boxes: list[tuple], gts: list) -> "cv2.Mat":
    img = cv2.imread(img_path)
    for g in gts:
        x1, y1, x2, y2 = (int(v) for v in g.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(img, f"GT {g.class_name}", (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
    for box, cls_name, score, kind in boxes:
        x1, y1, x2, y2 = (int(v) for v in box)
        if kind == "tp":
            color, thick = (255, 120, 0), 1
        else:
            color, thick = (0, 0, 255), 3
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        label = f"{'FP' if kind == 'fp' else ''} {cls_name} {score:.2f}".strip()
        y_txt = y2 + 18 if kind == "fp" else max(14, y1 - 6)
        cv2.putText(img, label, (x1, y_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return img


def main() -> int:
    args = parse_args()
    bench_dir = args.benchmark if Path(args.benchmark).is_absolute() else _ROOT / args.benchmark
    img_dir, lbl_dir = bench_dir / "images", bench_dir / "labels"
    pairs = load_yolo_gts(img_dir, lbl_dir, _BENCHMARK_CLASSES)
    logger.info("Benchmark: %s (%d images)", bench_dir, len(pairs))

    out_root = args.out if args.out.is_absolute() else _ROOT / args.out

    per_model_boxes: dict[str, dict] = {}
    per_model_gts: dict[str, dict] = {}
    fp_image_union: set[str] = set()

    for w in args.weights:
        weights = w if w.is_absolute() else _ROOT / w
        label = _label(weights)
        logger.info("Evaluating %s ...", label)
        if is_rfdetr_checkpoint(weights):
            model = load_rfdetr_for_eval(weights, confidence_floor=args.conf)
            preds, gts = collect_predictions_rfdetr(model, pairs, _COLLAPSE)
        else:
            from ultralytics import YOLO
            model = YOLO(str(weights))
            preds, gts = collect_predictions(model, pairs, _COLLAPSE, imgsz=args.imgsz,
                                             conf=args.conf, device=args.device)
        del model

        by_image = _classify_boxes(preds, gts)
        per_model_boxes[label] = by_image
        per_model_gts[label] = _gts_by_image(gts)
        n_fp_images = {img for img, boxes in by_image.items() if any(k == "fp" for *_, k in boxes)}
        fp_image_union |= n_fp_images
        logger.info("  %d image(s) with >=1 FP", len(n_fp_images))

    logger.info("Rendering %d image(s) (union across models) x %d model(s) ...",
                len(fp_image_union), len(args.weights))
    for label, by_image in per_model_boxes.items():
        out_dir = out_root / label
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("*.png"):
            old.unlink()
        gts_by_image = per_model_gts[label]
        for img_path in fp_image_union:
            img = _draw(img_path, by_image.get(img_path, []), gts_by_image.get(img_path, []))
            stem = Path(img_path).stem
            has_fp = any(k == "fp" for *_, k in by_image.get(img_path, []))
            tag = "FP" if has_fp else "clean"
            cv2.imwrite(str(out_dir / f"{tag}_{stem}.png"), img)
        logger.info("  %s -> %s", label, out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

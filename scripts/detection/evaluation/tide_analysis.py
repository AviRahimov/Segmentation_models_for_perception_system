#!/usr/bin/env python3
"""TIDE error-type breakdown for the current best checkpoint per model family.

Why: manually eyeballing FP-gallery crops can't distinguish "the model
hallucinated a box on pure background" (a real detection-quality problem)
from "the model drew two overlapping boxes on one real object, and the
second one — while it visually sits right on something real — is counted
as a false positive because one-to-one GT matching only keeps one" (a
duplicate-detection artifact, not a hallucination). TIDE (Bolya et al.,
ECCV 2020) quantifies this split directly: Classification / Localization /
Both / Duplicate / Background / Missed, instead of one undifferentiated
"FP" bucket.

Reuses this project's own Pred/GT collection machinery (_ap_utils.py, same
as leaderboard.py) rather than round-tripping through a COCO JSON file --
tidecv.Data has a direct Python API (add_ground_truth/add_detection) that
takes exactly that shape once boxes are converted from this project's
xyxy convention to TIDE's COCO-style xywh.

Usage
-----
    python scripts/detection/evaluation/tide_analysis.py
    python scripts/detection/evaluation/tide_analysis.py \\
        --only weights/detection/rfdetr-m/detection_dataset_hardneg/conservative_aug/best.pt

Output: reports/detection/tide_analysis.md (+ console summary via tidecv's own printer)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / "scripts" / "detection" / "training"))

from _ap_utils import (  # noqa: E402
    collect_predictions,
    collect_predictions_rfdetr,
    is_rfdetr_checkpoint,
    load_rfdetr_for_eval,
    load_yolo_gts,
)
from leaderboard import _COLLAPSE, _label  # noqa: E402

from tidecv import TIDE, Data  # noqa: E402
from tidecv.quantify import (  # noqa: E402
    BackgroundError, BoxError, ClassError, DuplicateError, MissedError, OtherError,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("tide_analysis")

_BENCHMARK_CLASSES = ["Military Vehicle", "person"]
_CLASS_TO_ID = {name: i for i, name in enumerate(_BENCHMARK_CLASSES)}

_MAIN_ERROR_TYPES = [ClassError, BoxError, OtherError, DuplicateError, BackgroundError, MissedError]

_DEFAULT_CHECKPOINTS = [
    _ROOT / "weights/detection/rfdetr-s/detection_dataset/conservative_aug/best.pt",
    _ROOT / "weights/detection/rfdetr-m/detection_dataset_hardneg/conservative_aug/best.pt",
    _ROOT / "weights/detection/yolo11m/yolo_dataset_auto_labeled/freeze21/best.pt",
    _ROOT / "weights/detection/yolo26m/yolo_dataset_auto_labeled/freeze10_aug_clean_ft/best.pt",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", default="datasets/Detection_Dataset/test",
                   help="Benchmark split dir containing images/ and labels/")
    p.add_argument("--only", nargs="*", default=None,
                   help="Evaluate only these checkpoints (paths); default: one per model family")
    p.add_argument("--conf", type=float, default=0.05,
                   help="Confidence floor for collecting raw predictions from the model")
    p.add_argument("--conf-thr", type=float, default=0.40, dest="conf_thr",
                   help="Confidence threshold predictions are filtered to before TIDE sees them. "
                        "Matches leaderboard.py's _DEPLOY_CONF so error-type counts here are "
                        "directly comparable to the FP/img and FN/img numbers already reported "
                        "there. Without this, TIDE's own all-confidence-levels convention (same "
                        "as AP itself) counts every low-score speculative detection the deployed "
                        "player would never actually show, wildly inflating every error count.")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default="0")
    p.add_argument("--out", default="reports/detection/tide_analysis.md")
    return p.parse_args()


def _xyxy_to_xywh(box: tuple[float, float, float, float]) -> list[float]:
    # tidecv stores whatever box object it's given as-is (Data._prepare_box
    # is a no-op) and later hands lists of these straight to pycocotools'
    # Cython mask_utils.iou(). Confirmed empirically (reproduced with a
    # minimal 2-box example) that pycocotools==2.0.11's iou() requires each
    # box to be a plain `list`, not a `tuple` -- a tuple fails its internal
    # bbox-vs-RLE type detection and raises "list input can be bounding box
    # (Nx4) or RLEs" despite being structurally identical data. Also cast to
    # float explicitly: RF-DETR's Detection.bbox_xyxy is int-typed
    # (core/types.py), and int-typed coordinates hit the same failure.
    x1, y1, x2, y2 = (float(v) for v in box)
    return [x1, y1, x2 - x1, y2 - y1]


def _build_tide_data(preds, gts, image_ids: dict[str, int]) -> tuple[Data, Data]:
    gt_data = Data("gt")
    pred_data = Data("preds")
    for g in gts:
        if g.class_name not in _CLASS_TO_ID:
            continue
        img_id = image_ids.setdefault(g.image_id, len(image_ids))
        gt_data.add_ground_truth(img_id, _CLASS_TO_ID[g.class_name], _xyxy_to_xywh(g.box))
    for p in preds:
        if p.class_name not in _CLASS_TO_ID:
            continue
        img_id = image_ids.setdefault(p.image_id, len(image_ids))
        pred_data.add_detection(img_id, _CLASS_TO_ID[p.class_name], p.score, _xyxy_to_xywh(p.box))
    return gt_data, pred_data


def main() -> int:
    args = parse_args()

    bench_dir = Path(args.benchmark)
    if not bench_dir.is_absolute():
        bench_dir = _ROOT / bench_dir
    img_dir, lbl_dir = bench_dir / "images", bench_dir / "labels"
    pairs = load_yolo_gts(img_dir, lbl_dir, _BENCHMARK_CLASSES)
    logger.info("Benchmark: %s  (%d images)", bench_dir, len(pairs))

    if args.only:
        ckpts = [Path(c) if Path(c).is_absolute() else _ROOT / c for c in args.only]
    else:
        ckpts = _DEFAULT_CHECKPOINTS

    tide = TIDE()
    rows = []
    for ckpt in ckpts:
        if not ckpt.exists():
            logger.warning("missing checkpoint: %s", ckpt)
            continue
        label = _label(ckpt)
        logger.info("Evaluating %s ...", label)

        if is_rfdetr_checkpoint(ckpt):
            model = load_rfdetr_for_eval(ckpt, confidence_floor=args.conf)
            preds, gts = collect_predictions_rfdetr(model, pairs, _COLLAPSE)
        else:
            from ultralytics import YOLO
            model = YOLO(str(ckpt))
            preds, gts = collect_predictions(model, pairs, _COLLAPSE, imgsz=args.imgsz,
                                             conf=args.conf, device=args.device)
        del model

        preds = [p for p in preds if p.score >= args.conf_thr]

        image_ids: dict[str, int] = {}
        gt_data, pred_data = _build_tide_data(preds, gts, image_ids)

        run = tide.evaluate(gt_data, pred_data, name=label)
        # FalsePositiveError/FalseNegativeError are never instantiated into
        # run.errors -- they're only dict keys tidecv uses internally to
        # compute dAP-if-fixed (see fix_special_errors()). Every main error
        # type except MissedError is some form of false positive, and
        # MissedError is exactly a false negative, so derive both directly
        # from the (verified-correct) main-error counts instead.
        counts = {et.short_name: len([e for e in run.errors if isinstance(e, et)]) for et in _MAIN_ERROR_TYPES}
        n_fp = sum(v for k, v in counts.items() if k != MissedError.short_name)
        n_fn = counts.get(MissedError.short_name, 0)
        dap = {et.short_name: v for et, v in run.fix_main_errors().items()}

        rows.append({"label": label, "ap": run.ap, "counts": counts, "fp": n_fp, "fn": n_fn, "dap": dap})
        logger.info("  AP=%.2f  counts=%s  FP=%d FN=%d", run.ap, counts, n_fp, n_fn)

    tide.summarize()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    et_names = [et.short_name for et in _MAIN_ERROR_TYPES]
    lines = [
        "# TIDE Error-Type Breakdown\n\n",
        f"> Benchmark: `{bench_dir.relative_to(_ROOT)}` ({len(pairs)} images) | "
        f"pos_threshold=0.5, background_threshold=0.1 (tidecv defaults) | "
        f"predictions filtered to conf>={args.conf_thr:.2f} before TIDE sees them (matches "
        f"leaderboard.py's deployment operating point) — so error counts below are directly "
        f"comparable to that report's FP/img and FN/img, unlike TIDE's own default "
        f"all-confidence-levels convention. 'AP' here is therefore a single-point metric at "
        f"this threshold, not the usual threshold-free Average Precision.\n\n",
        "Error types (raw counts, one row of `run.errors` each):\n"
        "- **Cls** classification error (right location, wrong class)\n"
        "- **Loc** localization error (right class, box doesn't overlap enough)\n"
        "- **Both** both wrong\n"
        "- **Dupe** duplicate detection (a second box on an already-matched real object)\n"
        "- **Bkg** background error (confident detection on nothing — true hallucination)\n"
        "- **Miss** missed a real object entirely (contributes to FN)\n\n",
        "| Model | AP | " + " | ".join(et_names) + " | FP (total) | FN (total) |\n",
        "|---|---|" + "---|" * len(et_names) + "---|---|\n",
    ]
    for r in rows:
        lines.append(
            f"| `{r['label']}` | {r['ap']:.2f} | "
            + " | ".join(str(r["counts"].get(n, 0)) for n in et_names)
            + f" | {r['fp']} | {r['fn']} |\n"
        )
    lines.append("\n## dAP impact per error type (TIDE's own headline metric — how much AP would "
                 "improve if this error type were fixed)\n\n")
    lines.append("| Model | " + " | ".join(et_names) + " |\n")
    lines.append("|---|" + "---|" * len(et_names) + "\n")
    for r in rows:
        lines.append(f"| `{r['label']}` | " + " | ".join(f"{r['dap'].get(n, 0):.2f}" for n in et_names) + " |\n")

    out_path.write_text("".join(lines))
    logger.info("TIDE report -> %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

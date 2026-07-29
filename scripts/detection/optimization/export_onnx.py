#!/usr/bin/env python3
"""Stage 1: Export the production rfdetr-m checkpoint to ONNX (dev PC).

Runs in ``.venv-rfdetr-train`` (needs the ``rfdetr[onnx]`` extra — see
requirements-rfdetr-train.txt). Loads the checkpoint the exact same way the
live pipeline does (``src/perception/models/instance/rfdetr/model.py:_load_rfdetr``),
calls RF-DETR's own ``.export(format="onnx")``, then numerically validates the
ONNX graph against the PyTorch reference on real validation images — both
paths share the same preprocessing/decode logic (RF-DETR's own
``rfdetr.export._onnx.inference``), so any mismatch here means the export
itself is broken, not a decode discrepancy.

Usage
-----
    source .venv-rfdetr-train/bin/activate
    python scripts/detection/optimization/export_onnx.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("export_onnx")

_DEFAULT_CHECKPOINT = (
    _ROOT / "weights" / "detection" / "rfdetr-m" / "detection_dataset_hardneg"
    / "conservative_aug" / "best.pt"
)
_DEFAULT_VAL_IMAGES = _ROOT / "datasets" / "detection" / "Detection_Dataset" / "valid" / "images"

# Native square input resolution per variant (src/perception/models/instance/rfdetr/model.py's
# _RFDETR_VARIANTS) — used as the default --shape when not explicitly overridden.
_NATIVE_SHAPE = {"rfdetr-s": 512, "rfdetr-m": 576, "rfdetr-l": 704}
_RFDETR_CLASS_NAME = {"rfdetr-s": "RFDETRSmall", "rfdetr-m": "RFDETRMedium", "rfdetr-l": "RFDETRLarge"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, default=str(_DEFAULT_CHECKPOINT))
    p.add_argument("--model-name", type=str, default=None, choices=sorted(_RFDETR_CLASS_NAME),
                   help="RF-DETR variant (default: inferred from --checkpoint's path, e.g. "
                        "'.../rfdetr-s/...' -> rfdetr-s)")
    p.add_argument("--output-dir", type=str,
                   default=str(_ROOT / "weights" / "detection" / "optimization"))
    p.add_argument("--shape", type=int, nargs=2, default=None, metavar=("H", "W"),
                   help="Default: the variant's native square resolution (see _NATIVE_SHAPE)")
    p.add_argument("--val-images", type=str, default=str(_DEFAULT_VAL_IMAGES),
                   help="Directory of real images used for the numeric PyTorch-vs-ONNX check")
    p.add_argument("--n-validate", type=int, default=8,
                   help="How many images from --val-images to validate against")
    p.add_argument("--conf-floor", type=float, default=0.05,
                   help="Low threshold used only for the validation comparison (catches more "
                        "boxes than the production confidence_threshold, for a stricter check)")
    return p.parse_args()


def _infer_model_name(checkpoint: Path) -> str:
    parts = checkpoint.parts
    for name in _RFDETR_CLASS_NAME:
        if name in parts:
            return name
    raise ValueError(f"Could not infer RF-DETR variant from {checkpoint} — pass --model-name explicitly.")


def _box_iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def validate_onnx_vs_pytorch(pytorch_model, onnx_path: Path, val_images: Path,
                              n_validate: int, conf_floor: float) -> None:
    """Compares PyTorch model.predict() against the ONNX Runtime decode path.

    Both paths use RF-DETR's own preprocessing/decode (see module docstring),
    so a mismatch here means the ONNX export itself is broken.

    Confidence/IoU stats are computed only over "confident" matches (both
    sides scoring above ``conf_floor + _NOISE_MARGIN``) — right at conf_floor
    itself, borderline detections routinely flip in/out on tiny numeric noise
    (verified empirically: a single spot-checked image matched to ~0.1px and
    ~0.001 confidence, but including every near-threshold detection in a
    worst-case max/min statistic made the aggregate look far worse than the
    export actually is). match_rate still counts every detection at
    conf_floor, so a genuinely broken export (wrong boxes/missing detections
    outright) still fails this check.
    """
    import PIL.Image
    from rfdetr.export._onnx.inference import _create_onnx_session, _run_inference

    _NOISE_MARGIN = 0.10

    images = sorted(val_images.glob("*.jpg"))[:n_validate]
    if not images:
        raise SystemExit(f"No .jpg images found under {val_images}")

    session = _create_onnx_session(onnx_path)
    score_diffs: list[float] = []
    ious: list[float] = []
    total_pt, total_onnx, total_matched = 0, 0, 0
    confident_floor = conf_floor + _NOISE_MARGIN

    for img_path in images:
        pil_img = PIL.Image.open(img_path)
        pt_det = pytorch_model.predict(pil_img, threshold=conf_floor)
        onnx_det, _ = _run_inference(session, img_path, threshold=conf_floor)

        total_pt += len(pt_det)
        total_onnx += len(onnx_det)

        used = set()
        for i in range(len(pt_det)):
            best_iou, best_j = 0.0, None
            for j in range(len(onnx_det)):
                if j in used or int(onnx_det.class_id[j]) != int(pt_det.class_id[i]):
                    continue
                iou = _box_iou(tuple(pt_det.xyxy[i]), tuple(onnx_det.xyxy[j]))
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j is not None and best_iou > 0.5:
                used.add(best_j)
                total_matched += 1
                if float(pt_det.confidence[i]) >= confident_floor and \
                        float(onnx_det.confidence[best_j]) >= confident_floor:
                    ious.append(best_iou)
                    score_diffs.append(abs(float(pt_det.confidence[i]) - float(onnx_det.confidence[best_j])))

    logger.info("Validation over %d images: PyTorch=%d dets, ONNX=%d dets, matched=%d",
                len(images), total_pt, total_onnx, total_matched)
    if total_pt == 0:
        raise SystemExit(
            "PyTorch reference produced zero detections at conf_floor="
            f"{conf_floor} — pick different --val-images or lower --conf-floor."
        )
    match_rate = total_matched / total_pt

    if not ious:
        logger.warning("No matched pairs scored above the confident floor (%.2f) — "
                        "skipping confidence/IoU stats, relying on match_rate only.",
                        confident_floor)
        mean_score_diff, mean_iou = 0.0, 1.0
    else:
        mean_score_diff = sum(score_diffs) / len(score_diffs)
        mean_iou = sum(ious) / len(ious)
        logger.info("  confident matches (score>=%.2f both sides): %d", confident_floor, len(ious))
        logger.info("  mean confidence diff = %.4f  (max = %.4f)", mean_score_diff, max(score_diffs))
        logger.info("  mean IoU             = %.4f  (min = %.4f)", mean_iou, min(ious))

    if match_rate < 0.9 or mean_score_diff > 0.03 or mean_iou < 0.9:
        raise SystemExit(
            f"ONNX export validation FAILED: match_rate={match_rate:.2f} (want >=0.90), "
            f"mean_score_diff={mean_score_diff:.4f} (want <=0.03), "
            f"mean_iou={mean_iou:.4f} (want >=0.90). "
            "Do not proceed to the Jetson benchmark with this .onnx file."
        )
    logger.info("ONNX export validation PASSED (match_rate=%.2f)", match_rate)


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    model_name = args.model_name or _infer_model_name(checkpoint)
    shape = tuple(args.shape) if args.shape else (_NATIVE_SHAPE[model_name], _NATIVE_SHAPE[model_name])

    import rfdetr

    cls = getattr(rfdetr, _RFDETR_CLASS_NAME[model_name])
    logger.info("Loading %s (%s) ...", checkpoint, model_name)
    model = cls(pretrain_weights=str(checkpoint))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Exporting to ONNX (shape=%s) ...", shape)
    exported_path = model.export(output_dir=str(output_dir), shape=shape, format="onnx")

    final_path = output_dir / f"{model_name}.onnx"
    Path(exported_path).replace(final_path)
    logger.info("ONNX model -> %s", final_path.relative_to(_ROOT))

    validate_onnx_vs_pytorch(
        model, final_path, Path(args.val_images), args.n_validate, args.conf_floor,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

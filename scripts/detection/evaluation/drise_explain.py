#!/usr/bin/env python3
"""D-RISE black-box saliency for one image's detections.

Why: TIDE (tide_analysis.py) tells you WHICH error category dominates (e.g.
rfdetr-m's FPs are mostly background error -- confident detections on pure
background). It can't tell you WHY the model fired there. D-RISE (Petsiuk et
al., CVPR 2021) answers that: it randomly masks the input hundreds of times,
watches how much each detection's score degrades, and turns that into a
heatmap over the ORIGINAL image showing which pixels the detector actually
relied on for that one specific box. Point it at a TIDE-flagged worst
offender or an existing FP-gallery crop and get a direct visual answer
(e.g. confirms/refutes "it's firing on the ego-vehicle hull" or "it's firing
on dust/rubble texture" instead of guessing from the crop alone).

Architecture-agnostic by design (xaitk-saliency's DRISEStack only needs a
predict-function callable) -- works unchanged on both RF-DETR and YOLO
checkpoints via the same is_rfdetr_checkpoint() dispatch tide_analysis.py
uses. Reuses this project's own model-loading helpers (_ap_utils.py) rather
than duplicating inference glue.

Given a full image (not a pre-cropped gallery image -- D-RISE needs the real
surrounding context and coordinate frame, and running the detector on a crop
alone would re-frame the coordinate system pointlessly), this:
  1. Runs the detector at --conf to find its own detections in the image, OR
     accepts an explicit --box/--label/--score to explain one hand-picked
     detection (e.g. a box read off a TIDE error / FP-gallery crop whose
     coordinates you already know).
  2. For each detection to explain, runs DRISEStack (n random masks -- each
     one a full detector forward pass, so this is NOT cheap: budget roughly
     n * (single-image inference time) per box).
  3. Saves a jet-colormap heatmap overlay next to the original image.

Usage
-----
    # Explain every detection the model itself finds in an image:
    python scripts/detection/evaluation/drise_explain.py \\
        --weights weights/detection/rfdetr-m/detection_dataset_hardneg/conservative_aug/best.pt \\
        --image datasets/detection/Detection_Dataset/test/images/some_frame.jpg

    # Explain one specific hand-picked box (e.g. from a TIDE-flagged crop):
    python scripts/detection/evaluation/drise_explain.py \\
        --weights weights/detection/rfdetr-m/detection_dataset_hardneg/conservative_aug/best.pt \\
        --image datasets/detection/Detection_Dataset/test/images/some_frame.jpg \\
        --box 120,340,410,560 --label "Military Vehicle" --score 0.62

    # Fast/rough pass while iterating (fewer masks -> noisier heatmap):
    python scripts/detection/evaluation/drise_explain.py ... --n 200

Output: reports/detection/drise/<image_stem>_box{i}_<label>_saliency.png
(side-by-side [original | heatmap overlay], one file per explained box).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

from _ap_utils import is_rfdetr_checkpoint, load_rfdetr_for_eval  # noqa: E402
from leaderboard import _COLLAPSE  # noqa: E402

from smqtk_detection.interfaces.detect_image_objects import DetectImageObjects  # noqa: E402
from smqtk_image_io.bbox import AxisAlignedBoundingBox  # noqa: E402
from xaitk_saliency.impls.gen_object_detector_blackbox_sal.drise import DRISEStack  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("drise_explain")

_BENCHMARK_CLASSES = ["Military Vehicle", "person"]
_CLASS_TO_ID = {name: i for i, name in enumerate(_BENCHMARK_CLASSES)}


class _ProjectModelBlackbox(DetectImageObjects):
    """Adapts this project's RF-DETR/YOLO wrappers to xaitk-saliency's
    DetectImageObjects protocol -- detect_objects(images) -> per-image list
    of (AxisAlignedBoundingBox, {class_name: score}).

    DRISEStack feeds this hundreds of randomly-masked variants of the same
    image; there's no channel-order conversion here because whatever array
    format the caller's ref_image used (this script always loads BGR via
    cv2.imread) is exactly what comes back out of the mask perturbation and
    is fed straight to the wrapped model unchanged -- both this project's
    RFDETRInstanceModel.predict() and ultralytics' YOLO.predict() already
    expect BGR/whatever-cv2-gives, same as the rest of this codebase.
    """

    def __init__(self, model, is_rfdetr: bool, imgsz: int, conf: float, device: str) -> None:
        self._model = model
        self._is_rfdetr = is_rfdetr
        self._imgsz = imgsz
        self._conf = conf
        self._device = device

    def detect_objects(self, img_iter):
        for img in img_iter:
            bgr = np.ascontiguousarray(img)
            dets: list[tuple[AxisAlignedBoundingBox, dict]] = []
            if self._is_rfdetr:
                raw = self._model.predict(bgr)
                raw = [(d.class_name, float(d.score), tuple(float(v) for v in d.bbox_xyxy)) for d in raw]
            else:
                results = self._model.predict(bgr, imgsz=self._imgsz, conf=self._conf,
                                              device=self._device, verbose=False)[0]
                names = results.names
                raw = [
                    (names[int(c)], float(s), tuple(float(v) for v in b))
                    for b, s, c in zip(results.boxes.xyxy.cpu().numpy(),
                                       results.boxes.conf.cpu().numpy(),
                                       results.boxes.cls.cpu().numpy())
                ]
            for class_name, score, (x1, y1, x2, y2) in raw:
                mapped = _COLLAPSE.get(class_name)
                if mapped is None:
                    continue
                bbox = AxisAlignedBoundingBox([x1, y1], [x2, y2])
                scores = {name: (score if name == mapped else 0.0) for name in _BENCHMARK_CLASSES}
                dets.append((bbox, scores))
            yield dets

    def get_config(self) -> dict:  # required by smqtk's Configurable base
        return {}


def _load_model(weights: Path, conf_floor: float):
    if is_rfdetr_checkpoint(weights):
        return load_rfdetr_for_eval(weights, confidence_floor=conf_floor), True
    from ultralytics import YOLO
    return YOLO(str(weights)), False


def _find_own_detections(blackbox: _ProjectModelBlackbox, bgr: np.ndarray) -> list[tuple]:
    """Run the wrapped model once (not through DRISE) to list what it detects
    in the unperturbed image -- these become the reference detections to
    explain when --box isn't given."""
    return next(iter(blackbox.detect_objects([bgr])))


def _overlay_heatmap(bgr: np.ndarray, sal: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """sal is nominally in [-1, 1], but DRISEStack's actual outputs observed on
    real detections cluster tightly near the top of that range (e.g.
    [0.79, 1.0] for a confident box) rather than spanning it -- a fixed
    (sal+1)*127.5 rescale washes the whole map out to a single near-uniform
    hot color with no visible contrast. Rescale per-box using this map's own
    observed min/max instead, so genuine relative importance is visible."""
    lo, hi = float(sal.min()), float(sal.max())
    sal_u8 = np.zeros_like(sal, dtype=np.uint8) if hi <= lo else \
        np.clip((sal - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(sal_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(bgr, 0.5, heat, 0.5, 0)
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return np.hstack([bgr, overlay])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--box", default=None, help="x1,y1,x2,y2 -- explain this exact box instead of "
                                               "re-running detection on --image")
    p.add_argument("--label", default=None, help="Class name for --box (required if --box is given)")
    p.add_argument("--score", type=float, default=1.0, help="Confidence/objectness for --box")
    p.add_argument("--conf", type=float, default=0.05, help="Detection threshold when auto-finding boxes")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default="0")
    p.add_argument("--n", type=int, default=1000, help="Number of random masks (DRISEStack). "
                                                        "Lower = faster, noisier heatmap.")
    p.add_argument("--s", type=int, default=8, help="Mask grid granularity")
    p.add_argument("--p1", type=float, default=0.5, help="Per-cell mask-on probability")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("reports/detection/drise"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    weights = args.weights if args.weights.is_absolute() else _ROOT / args.weights
    image_path = args.image if args.image.is_absolute() else _ROOT / args.image
    out_dir = args.out if args.out.is_absolute() else _ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        logger.error("Could not read image: %s", image_path)
        return 1

    model, is_rfdetr = _load_model(weights, conf_floor=args.conf)
    blackbox = _ProjectModelBlackbox(model, is_rfdetr, args.imgsz, args.conf, args.device)

    if args.box:
        if not args.label:
            logger.error("--label is required when --box is given")
            return 1
        x1, y1, x2, y2 = (float(v) for v in args.box.split(","))
        to_explain = [(AxisAlignedBoundingBox([x1, y1], [x2, y2]),
                      {name: (args.score if name == args.label else 0.0) for name in _BENCHMARK_CLASSES})]
    else:
        to_explain = _find_own_detections(blackbox, bgr)
        logger.info("Found %d detection(s) in %s at conf>=%.2f", len(to_explain), image_path.name, args.conf)
        if not to_explain:
            logger.warning("Nothing to explain -- lower --conf or pass --box explicitly")
            return 0

    bboxes = np.array([[b.min_vertex[0], b.min_vertex[1], b.max_vertex[0], b.max_vertex[1]]
                       for b, _ in to_explain])
    scores = np.array([[s[name] for name in _BENCHMARK_CLASSES] for _, s in to_explain])
    objectness = np.array([max(s.values()) for _, s in to_explain])

    drise = DRISEStack(n=args.n, s=args.s, p1=args.p1, seed=args.seed)
    logger.info("Running D-RISE (n=%d masks) for %d box(es) -- this is n forward passes per box...",
                args.n, len(to_explain))
    sal_maps = drise.generate(bgr, bboxes, scores, blackbox, objectness)

    stem = image_path.stem
    for i, ((box, cls_scores), sal) in enumerate(zip(to_explain, sal_maps)):
        label = max(cls_scores, key=cls_scores.get)
        box_xyxy = (box.min_vertex[0], box.min_vertex[1], box.max_vertex[0], box.max_vertex[1])
        panel = _overlay_heatmap(bgr, sal, box_xyxy)
        safe_label = label.replace(" ", "_")
        out_path = out_dir / f"{stem}_box{i}_{safe_label}_saliency.png"
        cv2.imwrite(str(out_path), panel)
        logger.info("  box %d (%s, score=%.2f) -> %s", i, label, cls_scores[label], out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

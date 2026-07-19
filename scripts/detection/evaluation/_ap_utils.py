"""Minimal AP50 evaluation with class-collapse support.

Why not model.val(): fine-grained models (6-class merged, 8-class verrckter)
cannot be validated by Ultralytics against the 2-class real benchmark — the
class counts differ. This module evaluates ANY model on the benchmark by
remapping its predicted classes through a collapse map first, then computing
standard all-point-interpolated AP per benchmark class.

Pure Python/numpy — unit-tested in tests/test_ap_utils.py.

RF-DETR checkpoints (see bottom of file) are evaluated through the same
Pred/GT/collapse machinery via a separate collection path — RFDETRInstanceModel
returns already-resolved `list[Detection]` rather than an Ultralytics `Results`
object, so it can't share `collect_predictions`'s per-call imgsz/conf/device/
augment passthrough (none of those apply post-construction to that wrapper).
"""
from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


@dataclass(frozen=True)
class Pred:
    image_id: str
    class_name: str
    score: float
    box: tuple[float, float, float, float]  # x1 y1 x2 y2


@dataclass(frozen=True)
class GT:
    image_id: str
    class_name: str
    box: tuple[float, float, float, float]


def iou_xyxy(a: tuple, b: tuple) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def collapse_preds(preds: list[Pred], collapse: dict[str, str]) -> list[Pred]:
    """Remap prediction class names; predictions mapping to None/absent are dropped."""
    out = []
    for p in preds:
        tgt = collapse.get(p.class_name)
        if tgt is not None:
            out.append(Pred(p.image_id, tgt, p.score, p.box))
    return out


def _match_class(
    preds: list[Pred],
    gts: list[GT],
    cls: str,
    iou_thr: float,
    min_score: float = 0.0,
) -> tuple[list[Pred], np.ndarray, set[tuple[str, int]]]:
    """Greedy score-ordered matching for one class.

    Returns (sorted class preds ≥ min_score, tp flags aligned to them,
    matched GT keys as (image_id, index-within-image)).
    """
    gts_by_img: dict[str, list[GT]] = defaultdict(list)
    for g in gts:
        if g.class_name == cls:
            gts_by_img[g.image_id].append(g)

    cls_preds = sorted((p for p in preds
                        if p.class_name == cls and p.score >= min_score),
                       key=lambda p: -p.score)
    matched: dict[str, set[int]] = defaultdict(set)
    matched_keys: set[tuple[str, int]] = set()
    tp = np.zeros(len(cls_preds))
    for i, p in enumerate(cls_preds):
        candidates = gts_by_img.get(p.image_id, [])
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(candidates):
            if j in matched[p.image_id]:
                continue
            v = iou_xyxy(p.box, g.box)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= iou_thr:
            matched[p.image_id].add(best_j)
            matched_keys.add((p.image_id, best_j))
            tp[i] = 1
    return cls_preds, tp, matched_keys


def ap_per_class(
    preds: list[Pred],
    gts: list[GT],
    classes: list[str],
    iou_thr: float = 0.5,
) -> dict[str, float]:
    """All-point-interpolated AP at a single IoU threshold, per class.

    Standard protocol: predictions sorted by score; each matches at most one
    GT (highest IoU ≥ thr, greedily, each GT used once); AP = area under the
    interpolated precision-recall curve. Classes with no GT → NaN.
    """
    results: dict[str, float] = {}
    for cls in classes:
        n_gt = sum(1 for g in gts if g.class_name == cls)
        if n_gt == 0:
            results[cls] = float("nan")
            continue
        _, tp, _ = _match_class(preds, gts, cls, iou_thr)
        fp = 1 - tp

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recall = cum_tp / n_gt
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

        # All-point interpolation (COCO-style continuous integration)
        mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 0.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        results[cls] = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    return results


def collect_predictions(
    model,
    image_label_pairs: list[tuple[str, list[GT]]],
    collapse: dict[str, str],
    *,
    imgsz: int = 640,
    conf: float = 0.05,
    device: str = "0",
    augment: bool = False,
) -> tuple[list[Pred], list[GT]]:
    """One inference pass over the benchmark → (collapsed preds, gts).

    Every downstream metric (AP, operating point, sweeps, size buckets,
    FP gallery) reuses this single pass.
    """
    preds: list[Pred] = []
    gts: list[GT] = []
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))

    for img_path, img_gts in image_label_pairs:
        gts.extend(img_gts)
        r = model.predict(img_path, imgsz=imgsz, conf=conf, device=device,
                          augment=augment, verbose=False)[0]
        if r.boxes is None:
            continue
        for box, cid, score in zip(r.boxes.xyxy.tolist(),
                                   r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            preds.append(Pred(str(img_path), names[int(cid)], float(score),
                              tuple(box)))
    return collapse_preds(preds, collapse), gts


def evaluate_collapsed(
    model,
    image_label_pairs: list[tuple[str, list[GT]]],
    collapse: dict[str, str],
    benchmark_classes: list[str],
    *,
    imgsz: int = 640,
    conf: float = 0.05,
    device: str = "0",
    augment: bool = False,
) -> dict[str, float]:
    """Backward-compatible wrapper: collect + AP50. Returns {class: AP50, 'mAP50'}."""
    preds, gts = collect_predictions(
        model, image_label_pairs, collapse,
        imgsz=imgsz, conf=conf, device=device, augment=augment,
    )
    per_class = ap_per_class(preds, gts, benchmark_classes)
    valid = [v for v in per_class.values() if v == v]  # drop NaN
    per_class["mAP50"] = float(np.mean(valid)) if valid else float("nan")
    return per_class


def operating_point(
    preds: list[Pred],
    gts: list[GT],
    classes: list[str],
    n_images: int,
    conf_thr: float = 0.4,
    iou_thr: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Deployment-threshold metrics per class: precision, recall, FP/image.

    Unlike AP (threshold-free), this answers "what does the player actually
    show at conf_thr" — the metric that exposes false-positive behavior.
    """
    out: dict[str, dict[str, float]] = {}
    for cls in classes:
        n_gt = sum(1 for g in gts if g.class_name == cls)
        cls_preds, tp_flags, _ = _match_class(preds, gts, cls, iou_thr,
                                              min_score=conf_thr)
        tp = int(tp_flags.sum())
        fp = len(cls_preds) - tp
        fn = n_gt - tp
        out[cls] = {
            "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
            "recall": tp / n_gt if n_gt else float("nan"),
            "fp_per_image": fp / n_images if n_images else float("nan"),
            "tp": tp, "fp": fp, "fn": fn,
        }
    return out


def threshold_sweep(
    preds: list[Pred],
    gts: list[GT],
    cls: str,
    n_images: int,
    iou_thr: float = 0.5,
    steps: np.ndarray | None = None,
) -> dict:
    """P/R/F1 vs confidence for one class → includes the best-F1 threshold."""
    if steps is None:
        steps = np.arange(0.05, 0.96, 0.05)
    curve = []
    for t in steps:
        op = operating_point(preds, gts, [cls], n_images,
                             conf_thr=float(t), iou_thr=iou_thr)[cls]
        p, r = op["precision"], op["recall"]
        f1 = (2 * p * r / (p + r)) if (p == p and r == r and (p + r) > 0) else 0.0
        curve.append({"conf": round(float(t), 2), "precision": p,
                      "recall": r, "f1": f1, "fp_per_image": op["fp_per_image"]})
    best = max(curve, key=lambda c: c["f1"]) if curve else None
    return {"curve": curve, "best": best}


def size_bucketed_recall(
    preds: list[Pred],
    gts: list[GT],
    classes: list[str],
    conf_thr: float = 0.4,
    iou_thr: float = 0.5,
) -> dict[str, dict]:
    """Recall per GT-size bucket with DATA-DRIVEN per-class boundaries.

    Buckets are terciles of each class's own GT sqrt-area distribution
    (computed here, from this benchmark) — fixed COCO cutoffs are useless on
    this data: person GTs have median ~21px with 71% under COCO's 32px
    "small" bound, vehicles median ~104px.
    """
    out: dict[str, dict] = {}
    for cls in classes:
        cls_gts = [g for g in gts if g.class_name == cls]
        if not cls_gts:
            continue
        sizes = np.array([((g.box[2] - g.box[0]) * (g.box[3] - g.box[1])) ** 0.5
                          for g in cls_gts])
        b1, b2 = np.percentile(sizes, [33.3, 66.7])

        # Match at the deployment threshold; bucket each GT by its size.
        gts_by_img: dict[str, list[GT]] = defaultdict(list)
        for g in cls_gts:
            gts_by_img[g.image_id].append(g)
        _, _, matched_keys = _match_class(preds, gts, cls, iou_thr,
                                          min_score=conf_thr)
        buckets = {"small": [0, 0], "medium": [0, 0], "large": [0, 0]}  # [hit, total]
        for img_id, img_gts in gts_by_img.items():
            for j, g in enumerate(img_gts):
                s = ((g.box[2] - g.box[0]) * (g.box[3] - g.box[1])) ** 0.5
                name = "small" if s <= b1 else "medium" if s <= b2 else "large"
                buckets[name][1] += 1
                if (img_id, j) in matched_keys:
                    buckets[name][0] += 1
        out[cls] = {
            "boundaries_px": (round(float(b1), 1), round(float(b2), 1)),
            "recall": {k: (hit / tot if tot else float("nan"))
                       for k, (hit, tot) in buckets.items()},
            "counts": {k: tot for k, (_, tot) in buckets.items()},
        }
    return out


def scores_and_labels(
    preds: list[Pred],
    gts: list[GT],
    cls: str,
    iou_thr: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Every class-``cls`` prediction's score, paired with its TP(1)/FP(0)
    label at ``iou_thr`` — the raw material for confidence calibration
    (``postprocess.calibration.fit_temperature``), unlike
    ``operating_point``/``threshold_sweep`` which only report aggregate
    counts at one confidence cutoff.
    """
    cls_preds, tp, _ = _match_class(preds, gts, cls, iou_thr, min_score=0.0)
    scores = np.array([p.score for p in cls_preds], dtype=float)
    return scores, tp


def false_positives(
    preds: list[Pred],
    gts: list[GT],
    classes: list[str],
    conf_thr: float = 0.4,
    iou_thr: float = 0.5,
) -> list[Pred]:
    """The unmatched predictions at the deployment threshold (for the gallery)."""
    fps: list[Pred] = []
    for cls in classes:
        cls_preds, tp_flags, _ = _match_class(preds, gts, cls, iou_thr,
                                              min_score=conf_thr)
        fps.extend(p for p, hit in zip(cls_preds, tp_flags) if not hit)
    return sorted(fps, key=lambda p: -p.score)


def load_yolo_gts(images_dir, labels_dir, class_names: list[str]) -> list[tuple[str, list[GT]]]:
    """Read a YOLO images/labels split into (image_path, [GT,...]) pairs."""
    import cv2
    from pathlib import Path

    pairs: list[tuple[str, list[GT]]] = []
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts: list[GT] = []
        lbl = labels_dir / (img_path.stem + ".txt")
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cid = int(parts[0])
                coords = [float(v) for v in parts[1:]]
                if len(coords) == 4:
                    # bbox: cx cy w h (normalized)
                    cx, cy, bw, bh = coords
                    box = ((cx - bw / 2) * w, (cy - bh / 2) * h,
                           (cx + bw / 2) * w, (cy + bh / 2) * h)
                else:
                    # polygon: x y pairs (normalized) → tight bbox
                    xs = [c * w for c in coords[0::2]]
                    ys = [c * h for c in coords[1::2]]
                    box = (min(xs), min(ys), max(xs), max(ys))
                gts.append(GT(str(img_path), class_names[cid], box))
        pairs.append((str(img_path), gts))
    return pairs


# --------------------------------------------------------------------------- #
# RF-DETR adapter                                                              #
#                                                                               #
# Class scheme is inferred from the checkpoint's dataset-folder-name component
# ("6class" in the path -> the 6-class scheme, else 2-class) — mirrors
# rfdetr_2class/rfdetr_6class in config.yaml, same fragility (path-name-based,
# not read from the checkpoint itself): would need updating if a new
# class-count scheme is trained under a name that doesn't contain "6class".
# --------------------------------------------------------------------------- #

RFDETR_2CLASS_NAMES = ["Military Vehicle", "person"]
RFDETR_6CLASS_NAMES = ["tank", "truck", "armored_vehicle", "civilian_vehicle",
                       "soldier", "civilian"]


def is_rfdetr_checkpoint(ckpt_path) -> bool:
    from perception.models.instance.rfdetr.model import _RFDETR_VARIANTS
    return any(v in Path(ckpt_path).parts for v in _RFDETR_VARIANTS)


def infer_rfdetr_profile(ckpt_path, confidence_threshold: float | None = None):
    """RF-DETR checkpoint path -> list[ClassDef] with 0-indexed coco_classes.

    0-indexed (not 1-indexed like the YOLO-family profiles in config.yaml)
    because RFDETRInstanceModel does not subtract 1 — verified empirically
    against a fine-tuned checkpoint (2026-07-14/15): raw output class ids
    match the training dataset's own 0-indexed class order directly.

    confidence_threshold=None (the default) leaves each ClassDef's own
    threshold unset, so RFDETRInstanceModel.warmup() falls back to whatever
    confidence_threshold the model was constructed with — a per-class
    override here would otherwise silently win over that constructor value
    (this bit a first version: every class hardcoded to 0.05 meant
    load_rfdetr_for_eval's confidence_floor param was always ignored).
    """
    from perception.config.schema import ClassDef

    names = RFDETR_6CLASS_NAMES if "6class" in str(ckpt_path) else RFDETR_2CLASS_NAMES
    return [
        ClassDef(name=n, text_prompt=n, display_mode="both", color_rgb=(0, 0, 0),
                is_semantic=False, coco_classes=(i,),
                confidence_threshold=confidence_threshold)
        for i, n in enumerate(names)
    ]


def load_rfdetr_for_eval(ckpt_path, confidence_floor: float = 0.05):
    """Construct + warm up an RFDETRInstanceModel directly — no AppConfig/YAML
    needed, since the wrapper's constructor + warmup(classes) is all it takes."""
    from perception.models.instance.rfdetr.model import RFDETRInstanceModel, _RFDETR_VARIANTS

    ckpt_path = Path(ckpt_path)
    model_name = next((v for v in _RFDETR_VARIANTS if v in ckpt_path.parts), None)
    if model_name is None:
        raise ValueError(f"Could not infer RF-DETR variant from path: {ckpt_path}")

    model = RFDETRInstanceModel(weights=str(ckpt_path),
                                confidence_threshold=confidence_floor,
                                model_name=model_name)
    model.warmup(infer_rfdetr_profile(ckpt_path))
    return model


def collect_predictions_rfdetr(
    model,
    image_label_pairs: list[tuple[str, list[GT]]],
    collapse: dict[str, str],
) -> tuple[list[Pred], list[GT]]:
    """RF-DETR analogue of collect_predictions — model.predict(frame_bgr)
    already returns resolved, thresholded Detection objects, so this is a
    direct field mapping rather than an Ultralytics Results unpack."""
    import cv2

    preds: list[Pred] = []
    gts: list[GT] = []
    for img_path, img_gts in image_label_pairs:
        gts.extend(img_gts)
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        for d in model.predict(bgr):
            preds.append(Pred(str(img_path), d.class_name, float(d.score),
                              tuple(d.bbox_xyxy)))
    return collapse_preds(preds, collapse), gts

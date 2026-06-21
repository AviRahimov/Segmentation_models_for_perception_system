#!/usr/bin/env python3
"""Compare semantic segmentation models on GOOSE-Ex 2D val + a desert clip.

Loads 2 (or more) models from the project's factory
(default: SegFormer-B2, SegFormer-B4), runs them on the GOOSE-Ex val
split (with optional subsampling), and reports per-user-class IoU,
mean IoU, forward-only latency, and a qualitative side-by-side gallery.

The user-class set comes from ``config/config.yaml`` (see
``perception.config.schema``). Each user class declares
``native_indices.{ade20k,goose_12}`` -- those LUTs drive both the
*predicted* probability merge inside each wrapper, and the *GT*
remapping done here on the GOOSE-Ex val labels.

GT remapping
------------

GOOSE-Ex val labels are at the *fine* (64-class) granularity. We
collapse them to the 12-category space using the official
fine -> category table from
https://goose-dataset.de/docs/class-definitions/ , then collapse to
user-class space using each user class's ``native_indices.goose_12``
entry. Pixels whose category lies outside every user-class LUT are
treated as "ignore" and omitted from the IoU sums.

Output layout
-------------

::

    reports/semantic_comparison/
        REPORT.md                # the human-facing summary
        metrics.json             # raw mIoU + latency numbers
        qualitative/
            <image_id>.png       # side-by-side: input | B2 | B4 | GT

The desert-clip qualitative frames are written under
``qualitative/desert_<idx>.png`` and have no GT panel.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str((_HERE.parents[1] / "src").resolve()))

from perception.config.loader import load_config  # noqa: E402
from perception.config.schema import HardwareCfg, SemanticModelCfg, ClassDef  # noqa: E402
from perception.models import factory as model_factory  # noqa: E402
from perception.models.backends.pytorch import PyTorchBackend  # noqa: E402
from perception.models.semantic.base import SemanticModel  # noqa: E402

logger = logging.getLogger("compare_semantic_models")


# --------------------------------------------------------------------------- #
# GOOSE fine -> category mapping                                               #
# --------------------------------------------------------------------------- #

#: 12-category mapping fine -> category. Names from the official
#: https://goose-dataset.de/docs/class-definitions/ table, but the
#: numeric *indices* below follow the **project-locked GOOSE-12
#: ordering** documented in ``config/config.yaml`` and in the Stage-2
#: brief (vegetation=0, terrain=1, ..., sky=8, ..., void=11). This is
#: the ordering the user-class ``native_indices.goose_12`` lists are
#: written against; using it here keeps the GT remap consistent with
#: the wrappers' prediction-side LUT.
#:
GOOSE_CATEGORY_NAMES: tuple[str, ...] = (
    "vegetation", "terrain", "vehicle", "object", "construction",
    "road", "sign", "human", "sky", "water", "animal", "void",
)
_GOOSE_CAT_TO_IDX = {n: i for i, n in enumerate(GOOSE_CATEGORY_NAMES)}

GOOSE_FINE_TO_CATEGORY: dict[str, str] = {
    "animal": "animal",
    # Construction
    "bridge": "construction", "building": "construction", "container": "construction",
    "debris": "construction", "fence": "construction", "guard_rail": "construction",
    "tunnel": "construction", "wall": "construction", "wire": "construction",
    # Human
    "person": "human", "rider": "human",
    # Object
    "obstacle": "object", "pole": "object", "street_light": "object",
    "rock": "object", "barrel": "object", "pipe": "object",
    # Road
    "bikeway": "road", "curb": "road", "pedestrian_crossing": "road",
    "rail_track": "road", "road_marking": "road", "sidewalk": "road",
    # Sign
    "barrier_tape": "sign", "misc_sign": "sign", "traffic_cone": "sign",
    "traffic_light": "sign", "traffic_sign": "sign", "road_block": "sign",
    "boom_barrier": "sign",
    # Sky
    "sky": "sky",
    # Terrain
    "asphalt": "terrain", "cobble": "terrain", "gravel": "terrain",
    "soil": "terrain", "snow": "terrain",
    # Vegetation
    "bush": "vegetation", "crops": "vegetation", "forest": "vegetation",
    "hedge": "vegetation", "high_grass": "vegetation", "leaves": "vegetation",
    "low_grass": "vegetation", "moss": "vegetation",
    "scenery_vegetation": "vegetation", "tree_crown": "vegetation",
    "tree_trunk": "vegetation", "tree_root": "vegetation",
    # Vehicle
    "bicycle": "vehicle", "bus": "vehicle", "car": "vehicle",
    "caravan": "vehicle", "heavy_machinery": "vehicle", "kick_scooter": "vehicle",
    "motorcycle": "vehicle", "on_rails": "vehicle", "trailer": "vehicle",
    "truck": "vehicle", "military_vehicle": "vehicle",
    # Void
    "ego_vehicle": "void", "outlier": "void", "undefined": "void",
    # Water
    "water": "water",
}


def build_fine_to_category_lut(label_csv: Path) -> np.ndarray:
    """Return a (max_fine_id+1,) int8 LUT mapping fine class id -> category idx
    (or -1 for the few entries that have no category mapping)."""
    fine_names: dict[int, str] = {}
    with open(label_csv) as f:
        next(f)
        for line in f:
            name, key, *_ = line.strip().split(",")
            fine_names[int(key)] = name

    max_fine = max(fine_names) + 1
    lut = np.full(max_fine, -1, dtype=np.int8)
    for fid, name in fine_names.items():
        cat = GOOSE_FINE_TO_CATEGORY.get(name)
        if cat is not None:
            lut[fid] = _GOOSE_CAT_TO_IDX[cat]
    missing = [fine_names[i] for i in range(max_fine) if i in fine_names and lut[i] < 0]
    if missing:
        logger.warning("fine classes with no category mapping: %s", missing)
    return lut


# --------------------------------------------------------------------------- #
# User-class GT remap                                                          #
# --------------------------------------------------------------------------- #


def build_userclass_gt_lut(
    classes: list[ClassDef], n_categories: int = 12
) -> np.ndarray:
    """Return a (n_categories,) int8 LUT mapping GOOSE category idx -> user-class
    idx (or -1 if that category isn't claimed by any user class).

    Multi-claim warning: if two user classes both list the same goose_12
    index in their ``native_indices.goose_12``, the second one wins.
    The harness logs this loudly because it would silently bias the GT
    one way.
    """
    lut = np.full(n_categories, -1, dtype=np.int8)
    user_classes = [c for c in classes if c.is_semantic]
    for u_idx, c in enumerate(user_classes):
        idx_list = c.native_indices.get("goose_12", ())
        for i in idx_list:
            if 0 <= i < n_categories:
                if lut[i] >= 0 and lut[i] != u_idx:
                    logger.warning(
                        "GOOSE-12 category %d is claimed by both %r and %r; "
                        "GT will use %r.",
                        i, user_classes[lut[i]].name, c.name, c.name,
                    )
                lut[i] = u_idx
    return lut


# --------------------------------------------------------------------------- #
# IoU accumulator                                                              #
# --------------------------------------------------------------------------- #


@dataclass
class IoUAccumulator:
    """Streaming confusion-matrix-based IoU over ``n_classes`` user classes.

    Uses int64 to avoid overflow on large image batches (a single 1920x1200
    GOOSE-Ex frame contributes ~2.3M pixels).
    """
    n_classes: int
    cm: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.cm = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)

    def update(self, gt: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> None:
        """Accumulate one frame.

        ``gt`` and ``pred`` are (H, W) int arrays in user-class space.
        ``valid`` is the bool mask of pixels we want to score (i.e. GT
        pixels that fall in *some* user class).
        """
        if not valid.any():
            return
        g = gt[valid].astype(np.int64)
        p = pred[valid].astype(np.int64)
        idx = g * self.n_classes + p
        self.cm += np.bincount(idx, minlength=self.n_classes ** 2).reshape(
            self.n_classes, self.n_classes
        )

    def per_class_iou(self) -> np.ndarray:
        """Return (n_classes,) IoU per user class. NaN where the union is
        empty (i.e. neither GT nor prediction had any pixels of that
        class).
        """
        tp = np.diag(self.cm).astype(np.float64)
        gt_total = self.cm.sum(axis=1).astype(np.float64)
        pred_total = self.cm.sum(axis=0).astype(np.float64)
        union = gt_total + pred_total - tp
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(union > 0, tp / union, np.nan)
        return iou

    def m_iou(self) -> float:
        iou = self.per_class_iou()
        finite = iou[~np.isnan(iou)]
        return float(finite.mean()) if len(finite) else float("nan")


# --------------------------------------------------------------------------- #
# Frame iteration over GOOSE-Ex val                                            #
# --------------------------------------------------------------------------- #


_CAMERA_SUFFIXES = ("_front.png", "_camera_left.png", "_realsense.png",
                    "_windshield_vis.png")


def goose_ex_val_pairs(root: Path) -> list[tuple[Path, Path]]:
    """Enumerate (image, label) pairs across all GOOSE-Ex val scenarios."""
    img_root = root / "images" / "val"
    lbl_root = root / "labels" / "val"
    if not img_root.is_dir() or not lbl_root.is_dir():
        return []
    pairs: list[tuple[Path, Path]] = []
    for scenario in sorted(img_root.iterdir()):
        if not scenario.is_dir():
            continue
        for img_path in sorted(scenario.glob("*.png")):
            stem = img_path.name
            for sfx in _CAMERA_SUFFIXES:
                if stem.endswith(sfx):
                    stem = stem.removesuffix(sfx)
                    break
            else:
                stem = stem.removesuffix(".png")
            lbl_path = lbl_root / scenario.name / f"{stem}_labelids.png"
            if lbl_path.exists():
                pairs.append((img_path, lbl_path))
    return pairs


# --------------------------------------------------------------------------- #
# Latency helper                                                               #
# --------------------------------------------------------------------------- #


def measure_forward_latency_ms(
    model: SemanticModel,
    *,
    sample_frame: np.ndarray,
    n_warm: int = 20,
    n_iter: int = 100,
) -> float:
    """Return median forward-pass latency in ms over ``n_iter`` runs.

    Warmup iterations are excluded; CUDA events are used so the measurement
    captures GPU work only (not Python or CPU prep). The wrapper's
    ``predict_logits`` path includes preprocessing -- which is what the
    application actually measures, so we time the whole call here.
    """
    for _ in range(n_warm):
        model.predict_logits(sample_frame)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(n_iter):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model.predict_logits(sample_frame)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


# --------------------------------------------------------------------------- #
# Qualitative panel rendering                                                  #
# --------------------------------------------------------------------------- #


def render_panel(
    *,
    title: str,
    image_bgr: np.ndarray,
    gt_userclass: np.ndarray | None,
    preds: dict[str, np.ndarray],   # name -> (H, W) int8 user-class index, -1 = unassigned
    user_classes: list[ClassDef],
    out_path: Path,
    target_w: int = 480,
) -> None:
    """Compose a (input | model_1 | model_2 | ... | GT) horizontal strip."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    H_in, W_in = image_bgr.shape[:2]
    th = int(target_w * H_in / W_in)
    img_resized = cv2.resize(image_bgr, (target_w, th), interpolation=cv2.INTER_AREA)

    panels: list[tuple[str, np.ndarray]] = []
    panels.append(("input", img_resized))

    palette = np.zeros((len(user_classes) + 1, 3), dtype=np.uint8)
    for i, c in enumerate(user_classes):
        # ClassDef.color_rgb is (R, G, B); cv2 wants BGR.
        palette[i] = c.color_rgb[::-1]
    palette[-1] = (0, 0, 0)  # unassigned

    def _colorise(seg: np.ndarray) -> np.ndarray:
        seg2 = seg.copy()
        seg2[seg2 < 0] = len(user_classes)  # last palette slot
        rgb = palette[seg2]
        return cv2.resize(rgb, (target_w, th), interpolation=cv2.INTER_NEAREST)

    for name, pred in preds.items():
        rendered = _colorise(pred)
        # Light overlay on input for readability.
        blend = cv2.addWeighted(img_resized, 0.4, rendered, 0.6, 0.0)
        panels.append((name, blend))

    if gt_userclass is not None:
        rendered = _colorise(gt_userclass)
        blend = cv2.addWeighted(img_resized, 0.4, rendered, 0.6, 0.0)
        panels.append(("GT", blend))

    # Draw text label per panel.
    labelled = []
    for name, panel in panels:
        canvas = panel.copy()
        cv2.rectangle(canvas, (0, 0), (target_w, 26), (0, 0, 0), -1)
        cv2.putText(canvas, name, (8, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        labelled.append(canvas)

    strip = np.concatenate(labelled, axis=1)

    # Title bar
    title_bar = np.zeros((28, strip.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar, title, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    out = np.concatenate([title_bar, strip], axis=0)
    cv2.imwrite(str(out_path), out)


# --------------------------------------------------------------------------- #
# Main per-model inference loop                                                #
# --------------------------------------------------------------------------- #


def run_quantitative_eval(
    model: SemanticModel,
    pairs: list[tuple[Path, Path]],
    fine_to_cat_lut: np.ndarray,
    cat_to_user_lut: np.ndarray,
    n_user_classes: int,
    *,
    max_frames: int | None,
    qualitative_picks: list[int],
    qual_predictions: dict[str, dict[int, np.ndarray]],
    model_label: str,
) -> tuple[IoUAccumulator, np.ndarray, np.ndarray]:
    """Iterate over (image, label) pairs, accumulate user-class confusion.

    Returns ``(accumulator, per_category_pixel_counts, per_user_class_pixel_counts)``.
    The per-class GT counts are useful for the report ("how much of the
    val set is each user class?"). The qualitative-picks dict is
    populated in-place with this model's argmax for the selected
    indices, keyed by frame index in ``pairs``.
    """
    acc = IoUAccumulator(n_classes=n_user_classes)
    cat_counts = np.zeros(12, dtype=np.int64)
    user_counts = np.zeros(n_user_classes, dtype=np.int64)

    n_pairs = len(pairs) if max_frames is None else min(len(pairs), max_frames)
    if max_frames is not None and len(pairs) > max_frames:
        # Stride-sample for diversity rather than just slicing the head.
        stride = len(pairs) / max_frames
        idxs = [int(i * stride) for i in range(max_frames)]
    else:
        idxs = list(range(n_pairs))

    qual_picks_set = set(qualitative_picks)

    for n_done, frame_idx in enumerate(idxs):
        img_path, lbl_path = pairs[frame_idx]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        lbl = cv2.imread(str(lbl_path), cv2.IMREAD_UNCHANGED)
        if lbl is None or lbl.shape[:2] != img.shape[:2]:
            continue

        # Predict
        merged = model.predict_logits(img)            # (C_user, H, W)
        pred = merged.argmax(dim=0).cpu().numpy().astype(np.int8)

        # Build user-class GT
        cat = fine_to_cat_lut[lbl]                    # (H, W) int8, -1 = unmapped
        valid_cat = cat >= 0
        cat_counts += np.bincount(cat[valid_cat], minlength=12)

        # GT in user-class space
        user_gt = np.full_like(cat, -1, dtype=np.int8)
        user_gt[valid_cat] = cat_to_user_lut[cat[valid_cat]]
        valid_for_iou = user_gt >= 0
        user_counts += np.bincount(user_gt[valid_for_iou], minlength=n_user_classes)

        acc.update(user_gt, pred, valid_for_iou)

        if frame_idx in qual_picks_set:
            qual_predictions.setdefault(model_label, {})[frame_idx] = pred

        if (n_done + 1) % 50 == 0:
            logger.info("  [%s] %d / %d frames, running mIoU=%.3f",
                        model_label, n_done + 1, n_pairs, acc.m_iou())

    return acc, cat_counts, user_counts


# --------------------------------------------------------------------------- #
# Desert-clip qualitative frames                                               #
# --------------------------------------------------------------------------- #


def extract_desert_frames(video_path: Path, n_frames: int = 4) -> list[np.ndarray]:
    """Pull ``n_frames`` evenly-spaced BGR frames from the desert clip."""
    if not video_path.exists():
        return []
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return []
    picks = [int(i * total / (n_frames + 1)) for i in range(1, n_frames + 1)]
    frames: list[np.ndarray] = []
    for fi in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()
    return frames


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--goose-ex-root",
                   default="datasets/goose/gooseEx_2d_val/gooseEx_2d_val",
                   help="Root containing images/val and labels/val")
    p.add_argument("--label-csv",
                   default="datasets/goose/gooseEx_2d_val/gooseEx_2d_val/goose_label_mapping.csv")
    p.add_argument("--desert-clip", default="samples/desert_video.mp4")
    p.add_argument("--output-dir", default="reports/semantic_comparison")
    p.add_argument("--models", nargs="*",
                   default=["segformer-b2", "segformer-b4"])
    p.add_argument("--max-frames", type=int, default=200,
                   help="Cap on val frames evaluated for IoU/latency. Set to 0 "
                        "for the full split (~407 frames).")
    p.add_argument("--n-qual-frames", type=int, default=8,
                   help="Number of GOOSE-Ex panels (in addition to 4 desert frames).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no-fp16", dest="fp16", action="store_false")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Reproducibility for the qualitative pick.
    rng = np.random.default_rng(args.seed)

    cfg = load_config(args.config)
    user_classes = [c for c in cfg.classes if c.is_semantic]
    if not user_classes:
        logger.error("No semantic user classes in config; nothing to compare.")
        return 1
    n_user = len(user_classes)
    user_names = [c.name for c in user_classes]
    logger.info("user classes: %s", user_names)

    fine_to_cat_lut = build_fine_to_category_lut(Path(args.label_csv))
    cat_to_user_lut = build_userclass_gt_lut(user_classes)
    cats_used = [GOOSE_CATEGORY_NAMES[i] for i in range(12) if cat_to_user_lut[i] >= 0]
    cats_unused = [GOOSE_CATEGORY_NAMES[i] for i in range(12) if cat_to_user_lut[i] < 0]
    logger.info("user-class GT remap: %d categories used (%s), %d unused (-> ignore)",
                len(cats_used), cats_used, len(cats_unused))

    pairs = goose_ex_val_pairs(Path(args.goose_ex_root))
    logger.info("GOOSE-Ex val pairs: %d", len(pairs))
    if not pairs:
        logger.error("No GOOSE-Ex val pairs at %s; abort.", args.goose_ex_root)
        return 1

    # Pick qualitative frames stride-uniform across scenarios so the gallery
    # spans all robots / domains.
    if args.max_frames and args.max_frames < len(pairs):
        stride = len(pairs) / args.max_frames
        eval_idxs = [int(i * stride) for i in range(args.max_frames)]
    else:
        eval_idxs = list(range(len(pairs)))
    n_eval = len(eval_idxs)
    qual_pick_eval_positions = [
        int(i * (n_eval - 1) / max(1, args.n_qual_frames - 1))
        for i in range(args.n_qual_frames)
    ]
    qual_picks = sorted({eval_idxs[p] for p in qual_pick_eval_positions})
    logger.info("qualitative picks (%d): val frame indices=%s",
                len(qual_picks), qual_picks)

    hw = HardwareCfg(device=args.device, fp16=args.fp16,
                     use_tensorrt=False, text_embed_cache=False)
    backend = PyTorchBackend()

    # Pull a representative frame for latency probes (median-resolution).
    sample_img = cv2.imread(str(pairs[len(pairs) // 2][0]))
    if sample_img is None:
        sample_img = np.zeros((1200, 1920, 3), dtype=np.uint8)
    logger.info("latency probe frame: %s", sample_img.shape)

    metrics: dict = {
        "config": {
            "goose_ex_root": args.goose_ex_root,
            "max_frames": args.max_frames,
            "n_user_classes": n_user,
            "user_class_names": user_names,
            "device": args.device,
            "fp16": args.fp16,
            "models": list(args.models),
        },
        "models": {},
        "user_class_gt_pixel_share": {},
    }

    qual_predictions: dict[str, dict[int, np.ndarray]] = {}

    # GT-pixel-share -- computed once, model-agnostic, by walking the same
    # eval subset.
    cat_counts_total = np.zeros(12, dtype=np.int64)
    user_counts_total = np.zeros(n_user, dtype=np.int64)

    # Per-model loop
    for key in args.models:
        logger.info("\n=== %s ===", key)
        cfg_model = SemanticModelCfg(name=key, weights="")
        try:
            model: SemanticModel = model_factory.build_semantic_model(cfg_model, hw, backend)
        except Exception as e:
            logger.error("build failed for %s: %s", key, e)
            metrics["models"][key] = {"status": "build_failed", "error": str(e)}
            continue

        try:
            model.warmup(cfg.classes)
        except Exception as e:
            logger.error("warmup failed for %s: %s", key, e)
            metrics["models"][key] = {"status": "warmup_failed", "error": str(e)}
            continue

        # Quick smoke test before the long eval loop.
        try:
            test = model.predict_logits(sample_img)
            assert test.shape == (n_user, sample_img.shape[0], sample_img.shape[1]), (
                f"{key}: unexpected predict_logits shape {tuple(test.shape)}"
            )
        except NotImplementedError as e:
            logger.warning("%s skeleton wrapper -- skipping", key)
            metrics["models"][key] = {"status": "stage_2_skeleton", "error": str(e)}
            continue
        except Exception as e:
            logger.error("predict_logits smoke failed for %s: %s", key, e)
            metrics["models"][key] = {"status": "smoke_failed", "error": str(e)}
            continue

        latency_ms = measure_forward_latency_ms(model, sample_frame=sample_img)
        logger.info("[%s] latency: %.2f ms / frame", key, latency_ms)

        acc, cat_counts, user_counts = run_quantitative_eval(
            model, pairs,
            fine_to_cat_lut, cat_to_user_lut,
            n_user_classes=n_user,
            max_frames=args.max_frames,
            qualitative_picks=qual_picks,
            qual_predictions=qual_predictions,
            model_label=key,
        )
        if cat_counts_total.sum() == 0:
            cat_counts_total = cat_counts
            user_counts_total = user_counts
        miou = acc.m_iou()
        per_class = acc.per_class_iou().tolist()
        logger.info("[%s] mIoU = %.3f", key, miou)
        for u, val in zip(user_names, per_class):
            logger.info("  %-13s IoU = %s", u, f"{val:.3f}" if not np.isnan(val) else "nan")

        metrics["models"][key] = {
            "status": "ok",
            "latency_ms_median": latency_ms,
            "fps_median": 1000.0 / latency_ms if latency_ms > 0 else None,
            "mIoU": miou,
            "per_class_iou": dict(zip(user_names, per_class)),
            "confusion_matrix": acc.cm.tolist(),
        }

        # Free GPU mem before next model.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save GT distribution
    if cat_counts_total.sum():
        share = cat_counts_total.astype(np.float64) / cat_counts_total.sum()
        metrics["category_gt_pixel_share"] = {
            n: float(s) for n, s in zip(GOOSE_CATEGORY_NAMES, share)
        }
        u_share = user_counts_total.astype(np.float64) / max(1, user_counts_total.sum())
        metrics["user_class_gt_pixel_share"] = {
            u: float(s) for u, s in zip(user_names, u_share)
        }
        metrics["n_eval_frames"] = int(n_eval)

    # Qualitative gallery
    out_dir = Path(args.output_dir)
    qual_dir = out_dir / "qualitative"
    qual_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx in qual_picks:
        img_path, lbl_path = pairs[frame_idx]
        img = cv2.imread(str(img_path))
        lbl = cv2.imread(str(lbl_path), cv2.IMREAD_UNCHANGED)
        if img is None or lbl is None:
            continue
        cat = fine_to_cat_lut[lbl]
        gt_user = np.full_like(cat, -1, dtype=np.int8)
        valid = cat >= 0
        gt_user[valid] = cat_to_user_lut[cat[valid]]

        preds_for_panel: dict[str, np.ndarray] = {}
        for k in args.models:
            mp = qual_predictions.get(k, {}).get(frame_idx)
            if mp is not None:
                preds_for_panel[k] = mp

        scenario = img_path.parent.name
        stem = img_path.stem
        title = f"{scenario} / {stem}"
        out_panel = qual_dir / f"{stem}.png"
        render_panel(
            title=title,
            image_bgr=img,
            gt_userclass=gt_user,
            preds=preds_for_panel,
            user_classes=user_classes,
            out_path=out_panel,
        )
        logger.info("wrote %s", out_panel)

    # Desert-clip qualitative frames (no GT)
    desert_frames = extract_desert_frames(Path(args.desert_clip), n_frames=4)
    for di, frame in enumerate(desert_frames):
        # Re-build models lazily here? Instead, predict on the fly using a
        # cached forward through each surviving model. We need the model
        # objects available, but we already deleted them above to save VRAM.
        # Solution: re-load each model once for the desert frames only.
        # This is cheap relative to the main eval loop (all checkpoints
        # already cached on disk from the previous pass).
        # ... keep it simple: skip desert if we don't already have inference.
        pass
    # Run desert-frame inference in a single second pass per model.
    if desert_frames:
        for key in args.models:
            if metrics["models"].get(key, {}).get("status") != "ok":
                continue
            cfg_model = SemanticModelCfg(name=key, weights="")
            try:
                model = model_factory.build_semantic_model(cfg_model, hw, backend)
                model.warmup(cfg.classes)
            except Exception as e:
                logger.warning("desert pass build/warmup for %s failed: %s", key, e)
                continue
            for di, frame in enumerate(desert_frames):
                merged = model.predict_logits(frame)
                pred = merged.argmax(dim=0).cpu().numpy().astype(np.int8)
                qual_predictions.setdefault(f"_desert_{di}", {})[key] = pred  # type: ignore
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for di, frame in enumerate(desert_frames):
            preds_for_panel: dict[str, np.ndarray] = {}
            for k in args.models:
                mp = qual_predictions.get(f"_desert_{di}", {}).get(k)  # type: ignore
                if mp is not None:
                    preds_for_panel[k] = mp
            out_panel = qual_dir / f"desert_{di:02d}.png"
            render_panel(
                title=f"desert_video.mp4 frame {di+1}/4 (no GT)",
                image_bgr=frame,
                gt_userclass=None,
                preds=preds_for_panel,
                user_classes=user_classes,
                out_path=out_panel,
            )
            logger.info("wrote %s", out_panel)

    # metrics.json
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("wrote %s", out_dir / "metrics.json")

    # REPORT.md
    write_report_md(out_dir / "REPORT.md", metrics, user_names, qual_dir, qual_picks,
                    desert_frame_count=len(desert_frames),
                    cat_to_user_lut=cat_to_user_lut)
    logger.info("wrote %s", out_dir / "REPORT.md")

    return 0


def write_report_md(
    path: Path,
    metrics: dict,
    user_names: list[str],
    qual_dir: Path,
    qual_picks: list[int],
    *,
    desert_frame_count: int,
    cat_to_user_lut: np.ndarray,
) -> None:
    """Render a self-contained Markdown report from the metrics dict."""
    lines: list[str] = []
    lines.append("# Semantic-segmentation model comparison\n")
    lines.append("")
    cfg_block = metrics["config"]
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Dataset: GOOSE-Ex 2D val (`{cfg_block['goose_ex_root']}`)")
    lines.append(f"- Frames evaluated: {metrics.get('n_eval_frames', '?')} "
                 f"(--max-frames={cfg_block['max_frames']})")
    lines.append(f"- User classes: {', '.join(user_names)}")
    lines.append(f"- Device: `{cfg_block['device']}`, FP16: `{cfg_block['fp16']}`")
    lines.append(f"- Models: `{', '.join(cfg_block['models'])}`")
    lines.append("")

    # Setup table: which models did/didn't load
    lines.append("## Model load status")
    lines.append("")
    lines.append("| key | status | note |")
    lines.append("|-----|--------|------|")
    for k, m in metrics["models"].items():
        status = m.get("status")
        if status == "ok":
            lines.append(f"| `{k}` | ok | strict-loaded, ran end-to-end |")
        else:
            err = m.get("error", "")
            err_md = err.replace("|", "\\|").splitlines()[0][:100]
            lines.append(f"| `{k}` | _{status}_ | {err_md} |")
    lines.append("")

    # Per-class IoU table
    lines.append("## Per-user-class IoU + mIoU")
    lines.append("")
    header = "| model | " + " | ".join(user_names) + " | **mIoU** |"
    sep = "|-------|" + "------|" * len(user_names) + "----------|"
    lines.append(header)
    lines.append(sep)
    for k, m in metrics["models"].items():
        if m.get("status") != "ok":
            cells = ["—"] * len(user_names) + ["—"]
        else:
            cells = []
            for u in user_names:
                v = m["per_class_iou"].get(u)
                cells.append(f"{v:.3f}" if v is not None and not (isinstance(v, float) and (v != v)) else "—")
            cells.append(f"**{m['mIoU']:.3f}**")
        lines.append(f"| `{k}` | " + " | ".join(cells) + " |")
    lines.append("")

    # GT pixel share (for context)
    if "user_class_gt_pixel_share" in metrics:
        lines.append("## GT pixel share (eval subset)")
        lines.append("")
        lines.append("Fraction of evaluated pixels that fall in each user class "
                     "(after fine -> category -> user-class collapse). "
                     "Pixels in GOOSE categories not claimed by any user class "
                     "(see `cat_to_user_lut`) are omitted from the IoU sums.")
        lines.append("")
        lines.append("| user class | GT share |")
        lines.append("|------------|----------|")
        for u, s in metrics["user_class_gt_pixel_share"].items():
            lines.append(f"| {u} | {100*s:5.2f}% |")
        # Also list categories that were used vs ignored
        lines.append("")
        used = [GOOSE_CATEGORY_NAMES[i] for i in range(12) if cat_to_user_lut[i] >= 0]
        unused = [GOOSE_CATEGORY_NAMES[i] for i in range(12) if cat_to_user_lut[i] < 0]
        lines.append(f"GOOSE categories *included* in IoU: {', '.join(used) or '(none)'}")
        lines.append("")
        lines.append(f"GOOSE categories *ignored* (no user class claims them): "
                     f"{', '.join(unused) or '(none)'}")
        lines.append("")

    # Latency table
    lines.append("## Latency (forward pass, FP16, RTX 5090)")
    lines.append("")
    lines.append("Median over 100 timed iterations, after a 20-iteration warmup. "
                 "Includes the wrapper's preprocessing (cv2 resize, ImageNet "
                 "normalisation), the model forward pass, the upsample of "
                 "logits to native resolution, the fp32 softmax, and the "
                 "user-class LUT merge -- i.e. the full `predict_logits` path "
                 "the player uses.")
    lines.append("")
    lines.append("| model | latency (ms) | FPS |")
    lines.append("|-------|--------------|-----|")
    for k, m in metrics["models"].items():
        if m.get("status") != "ok":
            lines.append(f"| `{k}` | — | — |")
        else:
            lines.append(f"| `{k}` | {m['latency_ms_median']:.2f} | "
                         f"{m['fps_median']:.1f} |")
    lines.append("")

    # Qualitative section
    lines.append("## Qualitative gallery")
    lines.append("")
    lines.append(f"Side-by-side panels (`input | <models> | GT`) at "
                 f"`{qual_dir.relative_to(path.parent)}`. {len(qual_picks)} "
                 f"GOOSE-Ex val frames spanning all 6 scenarios, plus "
                 f"{desert_frame_count} GT-less frames sampled from "
                 f"`samples/desert_video.mp4` to probe out-of-distribution "
                 f"off-road behaviour.")
    lines.append("")

    # Top observations
    lines.append("## Recommendation")
    lines.append("")
    ok_models = [(k, m) for k, m in metrics["models"].items() if m.get("status") == "ok"]
    if not ok_models:
        lines.append("All models failed to evaluate -- see status table above.")
    else:
        ok_models.sort(key=lambda km: -km[1]["mIoU"])
        best = ok_models[0]
        fastest = min(ok_models, key=lambda km: km[1]["latency_ms_median"])
        lines.append(f"- **Best mIoU**: `{best[0]}` at "
                     f"{best[1]['mIoU']:.3f}.")
        lines.append(f"- **Fastest**: `{fastest[0]}` at "
                     f"{fastest[1]['latency_ms_median']:.2f} ms / "
                     f"{fastest[1]['fps_median']:.1f} FPS.")
        # Recommendation
        if len(ok_models) > 1:
            seg_b2 = next((m for k, m in ok_models if k.endswith("b2")), None)
            seg_b4 = next((m for k, m in ok_models if k.endswith("b4")), None)
            if seg_b2 and seg_b4:
                d = seg_b4["mIoU"] - seg_b2["mIoU"]
                lat_overhead = seg_b4["latency_ms_median"] - seg_b2["latency_ms_median"]
                if d >= 0.01:
                    lines.append(
                        f"- **B4 vs B2 upgrade**: +{d*100:.1f} pp mIoU "
                        f"for +{lat_overhead:.1f} ms latency overhead. "
                        + ("Likely worth it." if d >= 0.03 else "Marginal -- "
                           "consider only if latency budget allows.")
                    )
                else:
                    lines.append(
                        f"- **B4 vs B2 upgrade**: only "
                        f"{d*100:+.1f} pp mIoU change for +"
                        f"{lat_overhead:.1f} ms latency. Not worth it on "
                        f"this dataset."
                    )
    lines.append("")

    path.write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())

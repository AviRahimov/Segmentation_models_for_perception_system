#!/usr/bin/env python3
"""One-off visual comparison: top-3 models under argmax (current live-pipeline
behavior) vs a traversable-probability floor of 0.78 (the value found to beat
both argmax and config.yaml's current 0.25 default on the binary-traversable
metric -- see CLAUDE.md). Two rows per sampled image so the effect is visible
directly, not just as a number. Samples across ORFD/zikim/FCDD so the finding
isn't specific to one dataset.

Usage
-----
    PYTHONPATH=src python scripts/segmentation/evaluation/compare_threshold_effect.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "src"))

from _semantic_eval_common import render_panel, resolve_weights  # noqa: E402
from compare_semantic_models import (  # noqa: E402
    _binary_traversable_iou,
    _fcdd_frames,
    _gt_trav_valid,
    _orfd_frames,
    _resize_to_native,
    _road_ground_channel_index,
    _TRAV_CLASSES,
    _zikim_frames,
)
from perception.config.loader import load_config  # noqa: E402
from perception.config.schema import SemanticModelCfg  # noqa: E402
from perception.models.backends.pytorch import PyTorchBackend  # noqa: E402
from perception.models.factory import build_semantic_model  # noqa: E402

TOP3 = ["segformer-b2", "mask2former-large", "mask2former-base"]
LABELS = {"segformer-b2": "distilled-segformer-b2", "mask2former-large": "mask2former-large",
          "mask2former-base": "mask2former-base"}
FLOOR = 0.78


def main() -> int:
    cfg = load_config(str(_REPO / "config" / "config.yaml"))
    hw = cfg.hardware
    backend = PyTorchBackend()
    sem_classes = list(cfg.semantic_classes)
    rg_idx = _road_ground_channel_index(tuple(c.name for c in sem_classes))

    print("Loading models...")
    models = {}
    for key in TOP3:
        weights = resolve_weights(key, "", cfg)
        mdl = build_semantic_model(SemanticModelCfg(name=key, weights=weights), hw, backend)
        mdl.warmup(cfg.classes)
        models[key] = mdl
        print(f"  {key}: {weights}")

    orfd_root = _REPO / "datasets" / "segmentation" / "ORFD"
    zikim_root = _REPO / "datasets" / "segmentation" / "off_road_zikim"
    fcdd_root = _REPO / "datasets" / "segmentation" / "FCDD"

    samples = []
    for fr in _orfd_frames(orfd_root, split="training", samples=2, seed=7):
        samples.append(("ORFD", fr))
    for fr in _zikim_frames(zikim_root, val_subdir="m24", samples=2, seed=7):
        samples.append(("zikim", fr))
    for fr in _fcdd_frames(fcdd_root, split="val", samples=2, seed=7):
        samples.append(("FCDD", fr))

    out_dir = _REPO / "reports" / "segmentation" / "threshold_effect_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    blocks = []
    for ds_name, fr in samples:
        has_gt = fr.gt_user is not None
        preds_argmax: dict[str, np.ndarray] = {}
        preds_floor: dict[str, np.ndarray] = {}
        ious = {}

        for key, mdl in models.items():
            merged = mdl.predict_logits(fr.img_bgr)
            merged = _resize_to_native(merged, fr.img_bgr.shape[:2])
            mnp = merged.float().cpu().numpy()
            pred_mc = mnp.argmax(axis=0).astype(np.int64, copy=False)
            preds_argmax[key] = pred_mc.astype(np.int8, copy=False)

            pred_trav_floor = mnp[rg_idx] >= FLOOR
            pred_vis_floor = pred_mc.astype(np.int8, copy=False).copy()
            pred_vis_floor[pred_trav_floor] = 1
            pred_vis_floor[(~pred_trav_floor) & (pred_mc == 1)] = 0
            preds_floor[key] = pred_vis_floor

            if has_gt:
                gt_trav, valid = _gt_trav_valid(fr.gt_user)
                pred_trav_argmax = pred_mc == rg_idx
                iou_a = _binary_traversable_iou(pred_trav_argmax, gt_trav, valid)
                iou_f = _binary_traversable_iou(pred_trav_floor, gt_trav, valid)
                ious[key] = (iou_a, iou_f)

        gt_vis = None
        if has_gt:
            gt_vis = np.where(fr.gt_user == 255, np.int8(-1), fr.gt_user.astype(np.int8))

        pane_a = out_dir / "_row_a.png"
        pane_f = out_dir / "_row_f.png"
        title_a = f"[{ds_name}] {fr.img_path.name} -- ARGMAX (current)"
        title_f = f"[{ds_name}] {fr.img_path.name} -- FLOOR={FLOOR} (proposed)"
        render_panel(title=title_a, image_bgr=fr.img_bgr, gt_userclass=gt_vis, preds=preds_argmax,
                     user_classes=list(_TRAV_CLASSES), out_path=pane_a, target_w=300)
        render_panel(title=title_f, image_bgr=fr.img_bgr, gt_userclass=gt_vis, preds=preds_floor,
                     user_classes=list(_TRAV_CLASSES), out_path=pane_f, target_w=300)
        row_a = cv2.imread(str(pane_a))
        row_f = cv2.imread(str(pane_f))
        pane_a.unlink(missing_ok=True)
        pane_f.unlink(missing_ok=True)

        if has_gt:
            print(f"[{ds_name}] {fr.img_path.name} trav_IoU argmax->floor: " +
                  ", ".join(f"{k}: {a:.3f}->{f:.3f}" for k, (a, f) in ious.items()))
        else:
            print(f"[{ds_name}] {fr.img_path.name}: qualitative only (no GT)")

        gap = np.full((6, row_a.shape[1], 3), (60, 60, 60), dtype=np.uint8)
        block = np.vstack([row_a, gap, row_f])
        block_path = out_dir / f"{ds_name}_{fr.img_path.stem}.png"
        cv2.imwrite(str(block_path), block)
        print(f"  wrote {block_path}")
        blocks.append((ds_name, fr.img_path.name, block_path, ious if has_gt else None))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

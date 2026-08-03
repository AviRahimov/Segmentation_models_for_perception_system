#!/usr/bin/env python3
"""Qualitative N-way side-by-side model comparison on real, unlabeled video
clips — no ground truth, no metrics, purely for visual inspection of how
each model handles real-world footage the models were never evaluated on.

Usage
-----
    PYTHONPATH=src python scripts/segmentation/evaluation/compare_on_raw_video.py \\
        --videos-dir "datasets/segmentation/Off_Road_ShutterStcok_Videos&Frames" \\
        --models mask2former-large mask2former-base segformer-b2 \\
        --output-dir reports/segmentation/shutterstock_comparison
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "src"))

from perception.config.loader import load_config
from perception.config.schema import ClassDef, SemanticModelCfg
from perception.models.backends.pytorch import PyTorchBackend
from perception.models.factory import SEMANTIC_DEFAULT_WEIGHTS, build_semantic_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("compare_on_raw_video")

_ORFD_CLASSES: list[ClassDef] = [
    ClassDef(name="non_traversable", text_prompt="-", display_mode="mask_only",
             color_rgb=(220, 40, 40), is_semantic=True, native_indices={}),
    ClassDef(name="traversable", text_prompt="-", display_mode="mask_only",
             color_rgb=(40, 255, 140), is_semantic=True, native_indices={}),
    ClassDef(name="sky", text_prompt="-", display_mode="mask_only",
             color_rgb=(80, 160, 255), is_semantic=True, native_indices={}),
]

_PALETTE = np.array([c.color_rgb[::-1] for c in _ORFD_CLASSES] + [(0, 0, 0)], dtype=np.uint8)


def _colorise_overlay(frame_bgr: np.ndarray, pred: np.ndarray) -> np.ndarray:
    seg = pred.copy()
    seg[seg < 0] = len(_ORFD_CLASSES)
    rgb = _PALETTE[seg]
    return cv2.addWeighted(frame_bgr, 0.4, rgb, 0.6, 0.0)


def _n_way_panel(frame_bgr: np.ndarray, preds: dict[str, np.ndarray], panel_w: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    th = int(panel_w * h / w)
    resized = cv2.resize(frame_bgr, (panel_w, th), interpolation=cv2.INTER_AREA)

    panels = [("input", resized)]
    for name, pred in preds.items():
        pred_r = cv2.resize(pred, (panel_w, th), interpolation=cv2.INTER_NEAREST)
        panels.append((name, _colorise_overlay(resized, pred_r)))

    labelled = []
    for name, panel in panels:
        canvas = panel.copy()
        cv2.rectangle(canvas, (0, 0), (panel_w, 24), (0, 0, 0), -1)
        cv2.putText(canvas, name, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        labelled.append(canvas)
    return np.concatenate(labelled, axis=1)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos-dir", required=True)
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--models", nargs="+", default=["mask2former-large", "mask2former-base", "segformer-b2"])
    p.add_argument("--output-dir", default="reports/segmentation/shutterstock_comparison")
    p.add_argument("--panel-w", type=int, default=480)
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args()

    videos_dir = Path(args.videos_dir)
    video_paths = sorted(videos_dir.glob("*.webm")) + sorted(videos_dir.glob("*.mp4"))
    if not video_paths:
        logger.error("No .webm/.mp4 files found in %s", videos_dir)
        return 2
    logger.info("Found %d video clips", len(video_paths))

    cfg = load_config(args.config)
    hw = cfg.hardware
    backend = PyTorchBackend()

    models = {}
    for key in args.models:
        weights = SEMANTIC_DEFAULT_WEIGHTS.get(key.lower().strip(), "")
        if key.lower().strip() == cfg.models.semantic.name.lower().strip() and cfg.models.semantic.weights:
            weights = cfg.models.semantic.weights
        mdl = build_semantic_model(SemanticModelCfg(name=key, weights=weights), hw, backend)
        mdl.warmup(cfg.classes)
        models[key] = mdl
        logger.info("Loaded %s (weights=%s)", key, weights)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for vp in video_paths:
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            logger.warning("Could not open %s — skipping.", vp)
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        writer = None
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames and n >= args.max_frames:
                break
            preds = {}
            for key, mdl in models.items():
                logits = mdl.predict_logits(frame)
                if logits.shape[-2:] != frame.shape[:2]:
                    logits = torch.nn.functional.interpolate(
                        logits.unsqueeze(0), size=frame.shape[:2], mode="bilinear", align_corners=False,
                    )[0]
                preds[key] = logits.argmax(dim=0).cpu().numpy().astype(np.int8)

            panel = _n_way_panel(frame, preds, args.panel_w)
            if writer is None:
                writer = cv2.VideoWriter(
                    str(out_dir / f"{vp.stem}_compare.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"), fps, (panel.shape[1], panel.shape[0]),
                )
            writer.write(panel)
            n += 1
        cap.release()
        if writer is not None:
            writer.release()
        logger.info("Saved %d frames -> %s", n, out_dir / f"{vp.stem}_compare.mp4")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Qualitative N-way side-by-side model comparison on real, unlabeled video
clips — no ground truth, no metrics, purely for visual inspection of how
each model handles real-world footage the models were never evaluated on.

Omit --videos-dir and/or --models to pick interactively from what's on disk.

Usage
-----
    PYTHONPATH=src python scripts/segmentation/evaluation/compare_on_raw_video.py \\
        --videos-dir "datasets/segmentation/Off_Road_ShutterStcok_Videos&Frames" \\
        --models mask2former-large mask2former-base segformer-b2 \\
        --output-dir reports/segmentation/shutterstock_comparison

    # interactive
    PYTHONPATH=src python scripts/segmentation/evaluation/compare_on_raw_video.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "src"))

from _semantic_eval_common import (  # noqa: E402
    BASELINE_CHOICES,
    ask_choice,
    ask_multi_choice,
    parse_model_spec,
    resolve_weights,
    scan_semantic_checkpoints,
)
from perception.config.loader import load_config
from perception.config.schema import ClassDef, SemanticModelCfg
from perception.models.backends.pytorch import PyTorchBackend
from perception.models.factory import build_semantic_model

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


def _scan_video_dirs(seg_root: Path) -> list[tuple[str, Path]]:
    """List datasets/segmentation/* subdirs that contain .webm/.mp4 files."""
    found = []
    if not seg_root.is_dir():
        return found
    for d in sorted(p for p in seg_root.iterdir() if p.is_dir()):
        if list(d.glob("*.webm")) or list(d.glob("*.mp4")):
            found.append((d.name, d))
    return found


def _pick_videos_dir_interactively() -> Path:
    seg_root = _REPO / "datasets" / "segmentation"
    found = _scan_video_dirs(seg_root)
    if not found:
        raise SystemExit(f"No .webm/.mp4 files found under any subdir of {seg_root}. Pass --videos-dir explicitly.")
    options = [(str(d), f"{name} ({len(list(d.glob('*.webm')) + list(d.glob('*.mp4')))} clips)") for name, d in found]
    chosen = ask_choice("Which video folder?", options, default_idx=0)
    return Path(chosen)


def _pick_models_interactively() -> list[str]:
    checkpoints = scan_semantic_checkpoints(_REPO / "weights" / "segmentation" / "orfd")
    all_choices = list(checkpoints) + list(BASELINE_CHOICES)
    options = [
        (f"{c.key}:{c.weights}" if c.weights else c.key, f"{c.key}  [{c.label}]")
        for c in all_choices
    ]
    default_idxs = tuple(range(min(3, len(options))))
    return ask_multi_choice("Which model(s)? (compare 2+ side by side)", options, default_idxs)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos-dir", default=None, help="Omit to pick interactively.")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--models", nargs="+", default=None,
                    help="Model keys, optionally 'key:weights_path'. Omit to pick interactively.")
    p.add_argument("--output-dir", default="reports/segmentation/raw_video_comparison")
    p.add_argument("--panel-w", type=int, default=480)
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args()

    videos_dir = Path(args.videos_dir) if args.videos_dir else _pick_videos_dir_interactively()
    video_paths = sorted(videos_dir.glob("*.webm")) + sorted(videos_dir.glob("*.mp4"))
    if not video_paths:
        logger.error("No .webm/.mp4 files found in %s", videos_dir)
        return 2
    logger.info("Found %d video clips", len(video_paths))

    model_specs = args.models if args.models else _pick_models_interactively()

    cfg = load_config(args.config)
    hw = cfg.hardware
    backend = PyTorchBackend()

    models = {}
    for spec in model_specs:
        key, explicit_w = parse_model_spec(spec)
        weights = resolve_weights(key, explicit_w, cfg)
        mdl = build_semantic_model(SemanticModelCfg(name=key, weights=weights), hw, backend)
        mdl.warmup(cfg.classes)
        models[spec] = mdl
        logger.info("Loaded %s (weights=%s)", spec, weights)

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

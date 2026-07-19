#!/usr/bin/env python3
"""Side-by-side comparison of tracker settings on one video.

Runs instance detection and semantic segmentation ONCE per frame (both are
independent of tracker settings) and feeds the same raw detections into
several independently-configured IoUInstanceTracker instances — one per
--min-hits value (or one per --recovery-floor value). Each variant is
rendered as its own labeled panel and concatenated into a single output
video, so opening one file gives synchronized playback of all variants with
one play/pause button.

Usage
-----
    python scripts/inference/compare_tracker_variants.py \\
        --source samples/off_road_vid_mitvah_24.mp4 \\
        --min-hits 1 2 3 \\
        --output runs/min_hits_compare.mp4

    # Shorter clip, custom panel width:
    python scripts/inference/compare_tracker_variants.py \\
        --source samples/clip.mp4 --min-hits 1 2 --max-frames 300 \\
        --panel-width 640 --output runs/compare.mp4

    # Compare low-confidence recovery floors instead (min-hits held fixed;
    # "none" = recovery disabled for that panel):
    python scripts/inference/compare_tracker_variants.py \\
        --source samples/off_road_vid_mitvah_24.mp4 \\
        --min-hits 3 --recovery-floor none 0.15 0.10 \\
        --output runs/recovery_compare.mp4

    # Compare tracker backends instead (min-hits/recovery-floor held fixed):
    python scripts/inference/compare_tracker_variants.py \\
        --source samples/off_road_vid_mitvah_24.mp4 \\
        --min-hits 3 --backend iou bytetrack \\
        --output runs/backend_compare.mp4
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str((_HERE.parents[1] / "src").resolve()))

from perception.config.loader import load_config, override_source  # noqa: E402
from perception.config.schema import LowConfRecoveryCfg  # noqa: E402
from perception.core.types import FrameResult, SemanticPrediction  # noqa: E402
from perception.io.factory import build_source  # noqa: E402
from perception.models.backends.factory import build_backend  # noqa: E402
from perception.models.factory import build_instance_model, build_semantic_model  # noqa: E402
from perception.models.instance._threshold_gate import gate_confidence  # noqa: E402
from perception.postprocess import filter_duplicates  # noqa: E402
from perception.render.renderer import Renderer  # noqa: E402
from perception.temporal.bytetrack_tracker import ByteTrackInstanceTracker  # noqa: E402
from perception.temporal.factory import build_logits_smoother, build_scene_cut_detector  # noqa: E402
from perception.temporal.iou_tracker import IoUInstanceTracker  # noqa: E402

logger = logging.getLogger("compare_tracker_variants")

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_BAR_H = 30


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--source", required=True, help="Video file (or any FrameSource path)")
    p.add_argument("--min-hits", type=int, nargs="+", default=None,
                   metavar="N",
                   help="One panel per value (default: 1 2 3, unless --backend "
                        "is the compared axis, where it defaults to this "
                        "config's own tuned min_hits as a single shared value)")
    p.add_argument("--recovery-floor", type=str, nargs="+", default=None,
                   metavar="FLOOR",
                   help="One panel per value ('none' = recovery disabled for "
                        "that panel). When given with >1 value, becomes the "
                        "compared axis and --min-hits must be a single value.")
    p.add_argument("--backend", type=str, nargs="+", default=None,
                   choices=["iou", "bytetrack"],
                   help="Tracker backend per panel ('iou' = this project's "
                        "greedy/Hungarian tracker, 'bytetrack' = roboflow/"
                        "trackers' ByteTrackTracker). When given with >1 "
                        "value, becomes the compared axis and --min-hits/"
                        "--recovery-floor must each be a single value.")
    p.add_argument("--output", required=True, help="Output MP4 path")
    p.add_argument("--max-frames", type=int, default=0, help="0 = entire clip")
    p.add_argument("--panel-width", type=int, default=640,
                   help="Each panel resized to this width (default 640)")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _label_panel(img_bgr: np.ndarray, title: str, panel_w: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    scale = panel_w / w
    resized = cv2.resize(img_bgr, (panel_w, int(h * scale)), interpolation=cv2.INTER_LINEAR)
    bar = np.zeros((_BAR_H, panel_w, 3), dtype=np.uint8)
    cv2.putText(bar, title, (8, 21), _FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, resized])


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    import transformers as _tf; _tf.logging.set_verbosity_error()

    cfg = load_config(args.config)
    cfg = override_source(cfg, source_type=None, path=args.source, camera=None)
    if not cfg.runs_yoloe_instance_inference:
        logger.error("models.instance is disabled in this config — nothing to track/compare.")
        return 2

    if args.min_hits is None:
        # Backend is the compared axis -> min_hits must be a single shared
        # value; default to this config's own tuned value rather than the
        # 3-value [1, 2, 3] sweep meant for the min-hits-axis case.
        args.min_hits = ([cfg.temporal.instance_tracker.min_hits]
                         if args.backend and len(args.backend) > 1 else [1, 2, 3])

    # Build the variant list: (label, min_hits, recovery_floor_or_None, backend).
    # recovery_floor is the per-PANEL gate re-applied below to one shared
    # detection pass — it does not require a second model instance.
    if args.backend and len(args.backend) > 1:
        if len(args.min_hits) != 1 or (args.recovery_floor and len(args.recovery_floor) != 1):
            logger.error("--backend with >1 value requires --min-hits/--recovery-floor "
                        "to each be a single value.")
            return 2
        shared_min_hits = args.min_hits[0]
        shared_floor = None
        if args.recovery_floor:
            tok = args.recovery_floor[0]
            shared_floor = None if tok.lower() == "none" else float(tok)
        variants = [(f"backend={b}", shared_min_hits, shared_floor, b) for b in args.backend]
    elif args.recovery_floor:
        if len(args.recovery_floor) > 1 and len(args.min_hits) != 1:
            logger.error("--recovery-floor with >1 value requires exactly one --min-hits value.")
            return 2
        shared_min_hits = args.min_hits[0]
        trk_backend = args.backend[0] if args.backend else "iou"
        variants = []
        for tok in args.recovery_floor:
            floor = None if tok.lower() == "none" else float(tok)
            label = f"recovery={tok}" + (f" [{trk_backend}]" if trk_backend != "iou" else "")
            variants.append((label, shared_min_hits, floor, trk_backend))
    else:
        trk_backend = args.backend[0] if args.backend else "iou"
        variants = [
            (f"min_hits={mh}" + (f" [{trk_backend}]" if trk_backend != "iou" else ""), mh, None, trk_backend)
            for mh in args.min_hits
        ]

    # If any panel requests a numeric recovery floor, the shared detection
    # pass must run at the LOWEST such floor so every panel's gate has boxes
    # to work with — a lower floor only ADDS boxes, it never removes ones
    # above a higher floor.
    numeric_floors = [f for _, _, f, _ in variants if f is not None]
    inst_cfg = cfg.models.instance
    if numeric_floors:
        inst_cfg = dataclasses.replace(
            inst_cfg,
            low_conf_recovery=LowConfRecoveryCfg(enabled=True,
                                                 recovery_conf_floor=min(numeric_floors)),
        )

    backend = build_backend(cfg.hardware.use_tensorrt)
    instance_model = build_instance_model(inst_cfg, cfg.hardware, backend)
    semantic_model = build_semantic_model(cfg.models.semantic, cfg.hardware, backend)
    instance_model.warmup(cfg.classes)
    semantic_model.warmup(cfg.classes)

    smoother = build_logits_smoother(cfg.temporal)
    scene_cut = build_scene_cut_detector(cfg.temporal)
    tc = cfg.temporal.instance_tracker

    def _make_tracker(mh: int, trk_backend: str):
        if trk_backend == "bytetrack":
            return ByteTrackInstanceTracker(
                lost_track_buffer=tc.max_hold_frames,
                frame_rate=tc.frame_rate,
                minimum_consecutive_frames=mh,
                minimum_iou_threshold=tc.iou_threshold,
                hold_score_decay=tc.hold_score_decay,
            )
        return IoUInstanceTracker(
            iou_threshold=tc.iou_threshold,
            max_hold_frames=tc.max_hold_frames,
            hold_score_decay=tc.hold_score_decay,
            bbox_alpha=tc.bbox_alpha,
            score_alpha=tc.score_alpha,
            use_hungarian_matching=tc.use_hungarian_matching,
            min_hits=mh,
        )

    trackers = [
        (label, floor, _make_tracker(mh, trk_backend))
        for label, mh, floor, trk_backend in variants
    ]
    dedup_cfg = cfg.postprocess.duplicate_filter
    renderer = Renderer(cfg.classes, cfg.player, yoloe_prompt_mode=cfg.models.instance.prompt_mode)
    logger.info("Comparing %s on %s", [label for label, _, _ in trackers], args.source)

    src = build_source(cfg.source)
    writer: cv2.VideoWriter | None = None
    n = 0
    t0 = time.perf_counter()
    try:
        while True:
            ok, frame = src.read()
            if not ok or frame is None:
                break
            idx = max(0, src.position - 1)

            cut = scene_cut.update(frame)
            if cut and cfg.temporal.semantic_ema.reset_on_scene_cut:
                smoother.reset()
                for _, _, t in trackers:
                    t.reset()

            # One shared detection pass (at the lowest requested recovery
            # floor, if any) — every panel re-derives its own gate below.
            raw = instance_model.predict(frame)

            logits = semantic_model.predict_logits(frame)
            smoothed = smoother.update(logits)
            sem_pred = SemanticPrediction(logits=smoothed, class_names=semantic_model.class_names)

            elapsed = max(1e-6, time.perf_counter() - t0)
            fps = (n + 1) / elapsed

            panels = []
            for label, floor, tracker in trackers:
                # Re-apply this panel's own gate to the shared raw detections
                # — a no-op (keeps everything) for plain --min-hits panels,
                # since those already satisfy score >= display_threshold.
                panel_dets = [
                    d for d in raw
                    if d.display_threshold is None
                    or gate_confidence(d.score, d.display_threshold, floor)
                ]
                if dedup_cfg.enabled and len(panel_dets) >= 2:
                    panel_dets = filter_duplicates(
                        panel_dets,
                        iou_threshold=dedup_cfg.iou_threshold,
                        containment_threshold=dedup_cfg.containment_threshold,
                        score_margin=dedup_cfg.score_margin,
                    )
                detections = tracker.update(frame, panel_dets)
                result = FrameResult(frame_bgr=frame, detections=detections,
                                     semantic=sem_pred, frame_idx=idx,
                                     inference_ms=elapsed * 1000, scene_cut=cut)
                rendered = renderer.render(result, fps=fps)
                panels.append(_label_panel(rendered, label, args.panel_width))

            max_h = max(p.shape[0] for p in panels)
            padded = []
            for p_img in panels:
                if p_img.shape[0] < max_h:
                    pad = np.zeros((max_h - p_img.shape[0], p_img.shape[1], 3), dtype=np.uint8)
                    p_img = np.vstack([p_img, pad])
                padded.append(p_img)
            grid = np.hstack(padded)

            if writer is None:
                h, w = grid.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fps_out = src.fps() if src.fps() > 0 else 30.0
                writer = cv2.VideoWriter(args.output, fourcc, fps_out, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open output writer for {args.output}")
            writer.write(grid)

            n += 1
            if n % 100 == 0:
                logger.info("  processed %d frames...", n)
            if args.max_frames and n >= args.max_frames:
                break
    finally:
        if writer is not None:
            writer.release()
        src.release()

    elapsed = time.perf_counter() - t0
    logger.info("Processed %d frames in %.2fs (%.1f FPS)", n, elapsed, n / max(1e-6, elapsed))
    logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

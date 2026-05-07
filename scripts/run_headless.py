#!/usr/bin/env python3
"""Headless inference entry point.

Runs the same :class:`PerceptionPipeline` as the GUI but writes the
rendered frames to an MP4 (or simply prints throughput). Useful for
profiling and CI smoke tests, and proves that the inference stack is
fully decoupled from Qt.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str((_HERE.parent / "src").resolve()))

from perception.config.loader import load_config, override_source  # noqa: E402
from perception.io.factory import build_source  # noqa: E402
from perception.pipeline.perception import build_pipeline  # noqa: E402
from perception.render.renderer import Renderer  # noqa: E402

logger = logging.getLogger("headless")


def main() -> int:
    p = argparse.ArgumentParser(description="Headless real-time perception inference")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--source", default=None)
    p.add_argument("--source-type", choices=["video", "camera", "image_dir"], default=None)
    p.add_argument("--camera", type=int, default=None)
    p.add_argument("--output", default=None, help="Optional MP4 output path")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Stop after N frames (0 = entire stream)")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    if args.source is not None or args.source_type is not None or args.camera is not None:
        cfg = override_source(
            cfg,
            source_type=args.source_type,
            path=args.source,
            camera=args.camera,
        )

    src = build_source(cfg.source)
    pipeline = build_pipeline(cfg)
    pipeline.warmup()
    renderer = Renderer(cfg.classes, cfg.player)

    writer: cv2.VideoWriter | None = None
    n = 0
    t0 = time.perf_counter()
    try:
        while True:
            ok, frame = src.read()
            if not ok or frame is None:
                break
            idx = max(0, src.position - 1)
            result = pipeline.process(frame, idx)
            elapsed = max(1e-6, time.perf_counter() - t0)
            rendered = renderer.render(result, fps=(n + 1) / elapsed)

            if args.output:
                if writer is None:
                    h, w = rendered.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    fps_out = src.fps() if src.fps() > 0 else 30.0
                    writer = cv2.VideoWriter(args.output, fourcc, fps_out, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not open output writer for {args.output}")
                writer.write(rendered)

            n += 1
            if args.max_frames and n >= args.max_frames:
                break
    finally:
        if writer is not None:
            writer.release()
        src.release()

    elapsed = time.perf_counter() - t0
    fps = n / max(1e-6, elapsed)
    logger.info("Processed %d frames in %.2fs (%.1f FPS)", n, elapsed, fps)
    if args.output:
        logger.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

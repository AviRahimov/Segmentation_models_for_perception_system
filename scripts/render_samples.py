#!/usr/bin/env python3
"""Batch-render annotated MP4s for every video file in a samples directory.

Loads models once, then processes each input clip sequentially (temporal
state is reset between clips). Default layout::

    samples/foo.mp4 -> samples/annotated/foo_annotated.mp4

With ``--run-all`` the script loops over every registered model and writes
outputs to per-model sub-directories::

    samples/annotated/segformer-b2-orfd/foo_annotated.mp4
    samples/annotated/segformer-b2-final/foo_annotated.mp4
    …

``samples`` is typically gitignored; outputs land alongside under
``samples/annotated`` by default (override with ``--out-dir``).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import torch

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
sys.path.insert(0, str((_REPO_ROOT / "src").resolve()))

from perception.config.loader import load_config, override_source  # noqa: E402
from perception.config.schema import AppConfig                      # noqa: E402
from perception.io.factory import build_source                      # noqa: E402
from perception.pipeline.perception import build_pipeline           # noqa: E402
from perception.render.renderer import Renderer                     # noqa: E402

logger = logging.getLogger("render_samples")

_VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"})


# --------------------------------------------------------------------------- #
# Registered models for --run-all                                              #
# --------------------------------------------------------------------------- #


def _render_model_registry() -> list[dict]:
    """All trained models that can be swapped in for visual comparison."""
    return [
        {
            "key":         "segformer-b2-orfd",
            "name":        "segformer-b2",
            "weights":     str(_REPO_ROOT / "weights" / "orfd" / "segformer-b2" / "best.pth"),
            "num_classes": 3,
        },
        {
            "key":         "segformer-b4-orfd",
            "name":        "segformer-b4",
            "weights":     str(_REPO_ROOT / "weights" / "orfd" / "segformer-b4" / "best.pth"),
            "num_classes": 3,
        },
        {
            "key":         "ddrnet-orfd",
            "name":        "ddrnet",
            "weights":     str(_REPO_ROOT / "weights" / "orfd" / "ddrnet" / "best.pth"),
            "num_classes": 3,
        },
        {
            "key":         "segformer-b2-final",
            "name":        "segformer-b2",
            "weights":     str(_REPO_ROOT / "weights" / "orfd" / "final_dataset" / "segformer-b2" / "best.pth"),
            "num_classes": 3,
        },
    ]


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _with_semantic_model(
    cfg: AppConfig,
    name: str,
    weights: str,
    num_classes: int,
) -> AppConfig:
    """Return a copy of cfg with the semantic model swapped out."""
    sem = replace(cfg.models.semantic, name=name, weights=weights, num_classes=num_classes)
    return replace(cfg, models=replace(cfg.models, semantic=sem))


def _collect_videos(
    samples_dir: Path,
    *,
    recursive: bool,
    skip_under: Path,
    exclude: set[str],
) -> list[Path]:
    skip_root = skip_under.resolve()
    out: list[Path] = []
    candidates = samples_dir.rglob("*") if recursive else samples_dir.iterdir()
    for p in candidates:
        if not p.is_file():
            continue
        if p.suffix.lower() not in _VIDEO_EXTENSIONS:
            continue
        if p.name in exclude:
            continue
        try:
            if p.resolve().is_relative_to(skip_root):
                continue
        except ValueError:
            pass
        out.append(p)
    return sorted(out, key=lambda x: str(x).lower())


def _render_one_clip(
    pipeline,
    renderer,
    cfg_template: AppConfig,
    video_path: Path,
    output_path: Path,
    *,
    max_frames: int,
) -> tuple[int, float]:
    """Return (frame_count, elapsed_seconds)."""
    cfg_vid = override_source(
        cfg_template,
        source_type="video",
        path=str(video_path.resolve()),
    )
    src = build_source(cfg_vid.source)
    pipeline.reset_temporal()

    writer: cv2.VideoWriter | None = None
    n = 0
    t0 = time.perf_counter()
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            ok, frame = src.read()
            if not ok or frame is None:
                break
            idx = max(0, src.position - 1)
            result = pipeline.process(frame, idx)
            elapsed = max(1e-6, time.perf_counter() - t0)
            rendered = renderer.render(result, fps=(n + 1) / elapsed)

            if writer is None:
                h, w = rendered.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fps_out = src.fps() if src.fps() > 0 else 30.0
                writer = cv2.VideoWriter(str(output_path), fourcc, fps_out, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open VideoWriter for {output_path}")
            writer.write(rendered)

            n += 1
            if max_frames and n >= max_frames:
                break
    finally:
        if writer is not None:
            writer.release()
        src.release()

    return n, time.perf_counter() - t0


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #


def main() -> int:
    p = argparse.ArgumentParser(
        description="Render annotated MP4 for each video under samples/",
    )
    p.add_argument("--config", default=str(_REPO_ROOT / "config/config.yaml"))
    p.add_argument(
        "--samples-dir",
        type=Path,
        default=_REPO_ROOT / "samples",
        help="Directory containing source videos",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <samples-dir>/annotated)",
    )
    p.add_argument(
        "--suffix",
        default="_annotated",
        help="Stem suffix before .mp4 (e.g. foo -> foo_annotated.mp4)",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Discover videos recursively under samples-dir",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Cap frames per clip (0 = full length)",
    )
    p.add_argument(
        "--exclude",
        nargs="*",
        default=["recording.mp4"],
        help="Video filenames to skip (default: recording.mp4)",
    )
    p.add_argument(
        "--run-all",
        action="store_true",
        help=(
            "Loop over all registered trained models. "
            "Outputs go to <out-dir>/<model-key>/. "
            "Skips any model whose checkpoint file is missing."
        ),
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    samples_dir = Path(args.samples_dir).resolve()
    if not samples_dir.is_dir():
        logger.error("Samples directory not found: %s", samples_dir)
        return 1

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (samples_dir / "annotated")
    cfg_template = load_config(args.config)

    exclude_set = set(args.exclude or [])
    videos = _collect_videos(
        samples_dir,
        recursive=args.recursive,
        skip_under=out_dir,
        exclude=exclude_set,
    )
    if not videos:
        logger.warning("No video files found under %s", samples_dir)
        return 0

    if exclude_set:
        logger.info("Excluding: %s", ", ".join(sorted(exclude_set)))
    logger.info("Found %d video(s) to render.", len(videos))

    # ── Determine which models to run ──────────────────────────────────────
    if args.run_all:
        registry = [
            m for m in _render_model_registry()
            if Path(m["weights"]).exists()
        ]
        if not registry:
            logger.error(
                "No checkpoint files found for any registered model. "
                "Train first with scripts/train_orfd.py."
            )
            return 1
        missing = [m["key"] for m in _render_model_registry() if not Path(m["weights"]).exists()]
        if missing:
            logger.warning("Skipping models with missing checkpoints: %s", ", ".join(missing))
        models_to_run = registry
        logger.info("Running %d model(s): %s", len(registry), ", ".join(m["key"] for m in registry))
    else:
        models_to_run = []  # empty → single-model mode below

    # ── Run ────────────────────────────────────────────────────────────────
    failed: list[tuple[str, str, str]] = []  # (model_key, video_name, error)

    def _run_model(cfg: AppConfig, model_out_dir: Path, model_label: str) -> None:
        pipeline = build_pipeline(cfg)
        pipeline.warmup()
        renderer = Renderer(
            cfg.classes,
            cfg.player,
            yoloe_prompt_mode=cfg.models.instance.prompt_mode,
        )
        logger.info("=== Model: %s → %s ===", model_label, model_out_dir)
        try:
            for vid in videos:
                out_name = f"{vid.stem}{args.suffix}.mp4"
                output_path = model_out_dir / out_name
                logger.info("  %s -> %s", vid.name, output_path.relative_to(out_dir.parent))
                try:
                    n, elapsed = _render_one_clip(
                        pipeline,
                        renderer,
                        cfg_template,
                        vid,
                        output_path,
                        max_frames=args.max_frames,
                    )
                    fps = n / max(1e-6, elapsed)
                    logger.info("    %d frames in %.2fs (%.1f FPS avg)", n, elapsed, fps)
                except Exception as e:
                    logger.exception("    FAILED: %s", e)
                    failed.append((model_label, vid.name, str(e)))
        finally:
            del pipeline
            torch.cuda.empty_cache()

    if models_to_run:
        for model_def in models_to_run:
            cfg = _with_semantic_model(
                cfg_template,
                name=model_def["name"],
                weights=model_def["weights"],
                num_classes=model_def["num_classes"],
            )
            _run_model(cfg, out_dir / model_def["key"], model_def["key"])
    else:
        # Single-model mode: use config as-is, flat output directory.
        _run_model(cfg_template, out_dir, "config")

    # ── Summary ────────────────────────────────────────────────────────────
    if failed:
        logger.error("%d render(s) failed:", len(failed))
        for model_key, vid_name, err in failed:
            logger.error("  [%s] %s — %s", model_key, vid_name, err)
        return 1

    logger.info("Done. All renders completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Annotate every image in a folder with YOLOE + SegFormer and save PNGs.

Default layout::

    <input-dir>/foo.jpg  ->  <input-dir>/annotated/foo_annotated.png

Override the output directory with ``--out-dir``.
The model used is whatever is active in ``config/config.yaml``
(or override with ``--config``).

Usage::

    python scripts/annotate_images.py
    python scripts/annotate_images.py --input-dir /path/to/imgs
    python scripts/annotate_images.py --input-dir /path/to/imgs --out-dir /tmp/out
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
sys.path.insert(0, str((_REPO_ROOT / "src").resolve()))

from perception.config.loader import load_config          # noqa: E402
from perception.pipeline.perception import build_pipeline  # noqa: E402
from perception.render.renderer import Renderer            # noqa: E402

logger = logging.getLogger("annotate_images")

_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

_DEFAULT_INPUT_DIR = Path("/home/avi/Documents/subset_dataset/img")


def _collect_images(folder: Path, skip_under: Path) -> list[Path]:
    skip_root = skip_under.resolve()
    out: list[Path] = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        try:
            if p.resolve().is_relative_to(skip_root):
                continue
        except ValueError:
            pass
        out.append(p)
    return sorted(out, key=lambda x: x.name.lower())


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run YOLOE + SegFormer on all images in a folder and save annotated PNGs.",
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=_DEFAULT_INPUT_DIR,
        help="Folder containing input images (default: %(default)s)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output folder (default: <input-dir>/annotated)",
    )
    p.add_argument(
        "--config",
        default=str(_REPO_ROOT / "config" / "config.yaml"),
        help="Path to config.yaml (default: %(default)s)",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        logger.error("Input directory not found: %s", input_dir)
        return 1

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (input_dir / "annotated")
    out_dir.mkdir(parents=True, exist_ok=True)

    images = _collect_images(input_dir, skip_under=out_dir)
    if not images:
        logger.warning("No image files found in %s", input_dir)
        return 0

    logger.info("Found %d image(s) in %s", len(images), input_dir)
    logger.info("Output → %s", out_dir)

    cfg = load_config(args.config)
    pipeline = build_pipeline(cfg)
    pipeline.warmup()
    renderer = Renderer(
        cfg.classes,
        cfg.player,
        yoloe_prompt_mode=cfg.models.instance.prompt_mode,
    )

    failed = 0
    t_start = time.perf_counter()

    for idx, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning("[%d/%d] Could not read %s — skipping", idx + 1, len(images), img_path.name)
            failed += 1
            continue

        try:
            result = pipeline.process(frame, idx)
            rendered = renderer.render(result, fps=0.0)
            out_path = out_dir / f"{img_path.stem}_annotated.png"
            cv2.imwrite(str(out_path), rendered)
            logger.info("[%d/%d] %s -> %s", idx + 1, len(images), img_path.name, out_path.name)
        except Exception:
            logger.exception("[%d/%d] FAILED: %s", idx + 1, len(images), img_path.name)
            failed += 1

    elapsed = time.perf_counter() - t_start
    total = len(images)
    ok = total - failed
    logger.info(
        "Done: %d/%d annotated in %.1fs (%.1f img/s)",
        ok, total, elapsed, ok / max(elapsed, 1e-6),
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

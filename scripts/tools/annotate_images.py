#!/usr/bin/env python3
"""Annotate every image in a folder with the instance + semantic models and save PNGs.

Default layout::

    <input-dir>/foo.jpg  ->  <input-dir>/annotated/foo_annotated.png

Override the output directory with ``--out-dir``.
The models used are whatever is active in ``config/config.yaml``
(or override with ``--config``), unless overridden per-run with
``--instance-model``/``--semantic-model``/``--pick-models``.

Usage::

    python scripts/tools/annotate_images.py --input-dir /path/to/imgs
    python scripts/tools/annotate_images.py --input-dir /path/to/imgs --out-dir /tmp/out
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str((_REPO_ROOT / "src").resolve()))
sys.path.insert(0, str(_HERE))

from _model_picker_common import (  # noqa: E402
    parse_model_spec,
    pick_instance_and_semantic_model_specs,
    resolve_instance_weights,
    resolve_weights,
)
from perception.config.loader import load_config, override_models  # noqa: E402
from perception.pipeline.perception import build_pipeline  # noqa: E402
from perception.render.renderer import Renderer            # noqa: E402

logger = logging.getLogger("annotate_images")

_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


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
        description="Run the instance + semantic models on all images in a folder and save annotated PNGs.",
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Folder containing input images",
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
    p.add_argument(
        "--instance-model", default=None, metavar="KEY[:weights_path]",
        help="Override the detection model (default: whatever config.yaml has pinned).",
    )
    p.add_argument(
        "--semantic-model", default=None, metavar="KEY[:weights_path]",
        help="Override the segmentation model (default: whatever config.yaml has pinned).",
    )
    p.add_argument(
        "--pick-models", action="store_true",
        help="Interactively choose both models from what's on disk (ignores "
             "--instance-model/--semantic-model if also given).",
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

    if args.pick_models:
        instance_spec, semantic_spec = pick_instance_and_semantic_model_specs(cfg, _REPO_ROOT)
        args.instance_model, args.semantic_model = instance_spec, semantic_spec
        logger.info("Picked instance=%s semantic=%s", instance_spec, semantic_spec)

    if args.instance_model:
        i_key, i_w = parse_model_spec(args.instance_model)
        cfg = override_models(cfg, instance_name=i_key, instance_weights=resolve_instance_weights(i_key, i_w, cfg))
    if args.semantic_model:
        s_key, s_w = parse_model_spec(args.semantic_model)
        cfg = override_models(cfg, semantic_name=s_key, semantic_weights=resolve_weights(s_key, s_w, cfg))

    pipeline = build_pipeline(cfg)
    pipeline.warmup()
    renderer = Renderer(cfg.classes, cfg.player)

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

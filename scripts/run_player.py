#!/usr/bin/env python3
"""GUI entry point: PyQt5 video player with real-time perception overlay."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow ``python scripts/run_player.py`` from a fresh checkout.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str((_HERE.parent / "src").resolve()))

from PyQt5.QtWidgets import QApplication  # noqa: E402

from perception.config.loader import load_config, override_source  # noqa: E402
from perception.io.factory import build_source  # noqa: E402
from perception.pipeline.perception import build_pipeline  # noqa: E402
from perception.render.renderer import Renderer  # noqa: E402
from perception.ui.main_window import MainWindow  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-time off-road perception player")
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config")
    parser.add_argument("--source", default=None,
                        help="Override source.path (video file or image directory)")
    parser.add_argument("--source-type", choices=["video", "camera", "image_dir"], default=None,
                        help="Override source.type")
    parser.add_argument("--camera", type=int, default=None, help="Override camera index")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("rf-detr").setLevel(logging.WARNING)
    import transformers as _tf; _tf.logging.set_verbosity_error()

    logging.info(f"Using model: {args.config}")

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
    renderer = Renderer(cfg.classes, cfg.player, yoloe_prompt_mode=cfg.models.instance.prompt_mode)

    app = QApplication(sys.argv)
    win = MainWindow(src, pipeline, renderer, cfg)
    win.resize(1280, 800)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

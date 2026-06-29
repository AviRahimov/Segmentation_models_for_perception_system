#!/usr/bin/env python3
"""Headless YOLOE discovery logging: JSONL per frame + TSV prompt counts.

Requires ``models.instance.prompt_mode: discovery`` (and vocabulary path)
in the YAML config. Runs the same pipeline as the player without Qt.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str((_HERE.parents[2] / "src").resolve()))

from perception.config.loader import load_config, override_source  # noqa: E402
from perception.io.factory import build_source  # noqa: E402
from perception.pipeline.perception import build_pipeline  # noqa: E402

logger = logging.getLogger("yoloe_discovery_dump")


def main() -> int:
    p = argparse.ArgumentParser(description="Log YOLOE discovery detections to JSONL + TSV")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--source", default=None, help="Override source.path")
    p.add_argument("--source-type", choices=["video", "camera", "image_dir"], default=None)
    p.add_argument("--camera", type=int, default=None)
    p.add_argument("--max-frames", type=int, default=100)
    p.add_argument("--jsonl", type=Path, default=None, help="Append one JSON object per frame")
    p.add_argument("--summary-tsv", type=Path, default=None, help="Write prompt,count,max_conf")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(message)s")

    cfg = load_config(args.config)
    if cfg.models.instance.prompt_mode != "discovery":
        logger.error(
            "Config must set models.instance.prompt_mode to 'discovery' (got %r).",
            cfg.models.instance.prompt_mode,
        )
        return 2

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

    counts: dict[str, list[float | int]] = defaultdict(lambda: [0, 0.0])  # n, max_conf
    jsonl_f = args.jsonl.open("a", encoding="utf-8") if args.jsonl else None
    try:
        n = 0
        while True:
            ok, frame = src.read()
            if not ok or frame is None:
                break
            idx = max(0, src.position - 1)
            result = pipeline.process(frame, idx)
            rec = {
                "frame_idx": int(idx),
                "detections": [
                    {
                        "prompt": d.class_name,
                        "confidence": round(float(d.score), 6),
                        "bbox_xyxy": [int(v) for v in d.bbox_xyxy],
                    }
                    for d in result.detections
                ],
            }
            if jsonl_f:
                jsonl_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                jsonl_f.flush()
            for d in result.detections:
                agg = counts[d.class_name]
                agg[0] = int(agg[0]) + 1
                agg[1] = max(float(agg[1]), float(d.score))
            n += 1
            if args.max_frames and n >= args.max_frames:
                break
    finally:
        if jsonl_f:
            jsonl_f.close()
        src.release()

    logger.info("Processed %d frames; %d unique prompts seen", n, len(counts))

    if args.summary_tsv:
        args.summary_tsv.parent.mkdir(parents=True, exist_ok=True)
        lines = ["prompt\tcount\tmax_confidence\n"]
        for prompt in sorted(counts.keys(), key=lambda k: (-int(counts[k][0]), k)):
            c, mx = counts[prompt]
            lines.append(f"{prompt}\t{int(c)}\t{float(mx):.6f}\n")
        args.summary_tsv.write_text("".join(lines), encoding="utf-8")
        logger.info("Wrote %s", args.summary_tsv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

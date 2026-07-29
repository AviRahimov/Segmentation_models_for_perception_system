#!/usr/bin/env python3
"""Stage 4: Side-by-side visual comparison of N model specs (any mix, any precision).

Standalone (uses ``_rfdetr_trt_common.load_model``, not
``scripts/detection/evaluation/compare_detection_models.py``'s dispatch, which
is Ultralytics/RF-DETR-PyTorch-specific and not worth risking a deep extension
for onnx:/engine: specs — see the plan for why). Supports ``pytorch:``/
``onnx:``/``engine:`` (RF-DETR) and ``ultralytics:`` (YOLO ``.pt``/``.onnx``/
``.engine`` uniformly) specs, mixed freely — e.g. rfdetr-s vs rfdetr-m vs the
best YOLO checkpoint, at whatever precision each turned out usable. A single
``--models`` spec also works (no comparison, just one annotated video) — used
to render separate per-model videos rather than side-by-side panels.

``--mode video`` overlays each panel's own live rolling-average FPS (real
per-model inference latency measured during the run, not the aggregate
benchmark number) in the top-right corner.

For a PyTorch-checkpoint-only comparison (no onnx:/engine: needed), prefer
``scripts/detection/evaluation/compare_detection_models.py --mode images/video``
instead — it already supports N-way mixed YOLO+RF-DETR PyTorch checkpoints
with zero new code (this tool exists specifically for onnx:/engine:/
ultralytics: specs that script doesn't handle).

Usage
-----
    python scripts/detection/optimization/compare_models.py --mode image \\
        --models pytorch:weights/detection/rfdetr-s/detection_dataset_hardneg/conservative_aug/best.pt \\
                 pytorch:weights/detection/rfdetr-m/detection_dataset_hardneg/conservative_aug/best.pt \\
                 ultralytics:weights/detection/yolo11m/yolo_dataset_auto_labeled/freeze21/best.pt \\
        --image datasets/detection/Detection_Dataset/valid/images/some_frame.jpg

    python scripts/detection/optimization/compare_models.py --mode video \\
        --models engine:weights/detection/optimization/rfdetr-s_fp32.engine \\
                 engine:weights/detection/optimization/rfdetr-m_fp32.engine \\
                 ultralytics:weights/detection/optimization/yolo11m_freeze21_fp32.engine \\
        --source /home/avi/Music/gaza_road_videos/tzir-driving.mp4 \\
        --output reports/detection/optimization/video_compare_3way.mp4

    # Single-model annotated video (no comparison), with a live FPS overlay:
    python scripts/detection/optimization/compare_models.py --mode video \\
        --models engine:weights/detection/optimization/rfdetr-s_fp32.engine \\
        --labels "rfdetr-s FP32" \\
        --source /home/avi/Music/gaza_road_videos/tzir-driving.mp4 \\
        --output results/videos_by_model/rfdetr-s/tzir-driving.mp4
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_HERE))

from _rfdetr_trt_common import load_model  # noqa: E402

_PALETTE = [(60, 200, 60), (60, 60, 230), (230, 160, 40), (200, 60, 200), (60, 200, 200)]
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FPS_WINDOW = 30  # rolling-average window, in frames


class _FPSTracker:
    """Rolling-average FPS over the last _FPS_WINDOW inference calls."""

    def __init__(self, window: int = _FPS_WINDOW) -> None:
        self._latencies_s: deque[float] = deque(maxlen=window)

    def tick(self, elapsed_s: float) -> None:
        self._latencies_s.append(elapsed_s)

    @property
    def fps(self) -> float | None:
        if not self._latencies_s:
            return None
        mean_s = sum(self._latencies_s) / len(self._latencies_s)
        return 1.0 / mean_s if mean_s > 0 else None


def _draw(frame_bgr: np.ndarray, dets, color: tuple, label: str, fps: float | None) -> np.ndarray:
    out = frame_bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        text = f"{d.class_id}:{d.score:.2f}"
        cv2.putText(out, text, (x1 + 1, max(0, y1 - 4)), _FONT, 0.45, color, 1, cv2.LINE_AA)
    cv2.putText(out, f"{label} ({len(dets)} dets)", (10, 25), _FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    if fps is not None:
        fps_text = f"{fps:.1f} FPS"
        (tw, _), _ = cv2.getTextSize(fps_text, _FONT, 0.7, 2)
        cv2.putText(out, fps_text, (out.shape[1] - tw - 12, 25), _FONT, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def _n_way(frame_bgr: np.ndarray, models: list, labels: list[str], threshold: float,
           trackers: list[_FPSTracker] | None = None) -> np.ndarray:
    panels = []
    for i, (model, label) in enumerate(zip(models, labels)):
        t0 = time.perf_counter()
        dets = model.infer(frame_bgr, threshold=threshold)
        if trackers is not None:
            trackers[i].tick(time.perf_counter() - t0)
        color = _PALETTE[i % len(_PALETTE)]
        fps = trackers[i].fps if trackers is not None else None
        panels.append(_draw(frame_bgr, dets, color, label, fps))
    return np.hstack(panels)


def _resolve_labels(args) -> list[str]:
    if args.labels:
        if len(args.labels) != len(args.models):
            raise SystemExit(f"--labels needs exactly {len(args.models)} entries (one per --models spec)")
        return args.labels
    return args.models


def run_image(args) -> int:
    models = [load_model(spec) for spec in args.models]
    labels = _resolve_labels(args)
    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"Could not read image: {args.image}")
    combined = _n_way(frame, models, labels, args.conf)
    out_path = Path(args.output) if args.output else _ROOT / "reports" / "detection" / "optimization" / "compare_image.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), combined)
    print(f"Saved: {out_path}")
    return 0


def run_video(args) -> int:
    models = [load_model(spec) for spec in args.models]
    labels = _resolve_labels(args)
    trackers = [_FPSTracker() for _ in models]
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(args.output) if args.output else _ROOT / "reports" / "detection" / "optimization" / "compare_video.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * len(models), h))

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and n >= args.max_frames:
            break
        combined = _n_way(frame, models, labels, args.conf, trackers)
        writer.write(combined)
        n += 1
    cap.release()
    writer.release()
    print(f"Saved {n} frames -> {out_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["image", "video"], required=True)
    p.add_argument("--models", nargs="+", required=True,
                   help="One or more pytorch:/onnx:/engine:/ultralytics: specs "
                        "(one spec = single annotated video, no comparison)")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Optional short on-screen label per model (default: the raw spec string)")
    p.add_argument("--image", help="Required for --mode image")
    p.add_argument("--source", help="Required for --mode video")
    p.add_argument("--output", default=None)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args()

    if args.mode == "image":
        if not args.image:
            raise SystemExit("--image is required for --mode image")
        return run_image(args)
    if not args.source:
        raise SystemExit("--source is required for --mode video")
    return run_video(args)


if __name__ == "__main__":
    raise SystemExit(main())

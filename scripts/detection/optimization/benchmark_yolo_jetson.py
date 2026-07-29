#!/usr/bin/env python3
"""Stage 3: Best-YOLO TensorRT export + Jetson benchmark.

Run this ON THE JETSON. Separate from benchmark_jetson.py (RF-DETR) —
different framework, and Ultralytics' own ``YOLO(path)`` already loads
``.pt``/``.onnx``/``.engine`` interchangeably through the same ``.export()``/
``.predict()`` calls, so there's no hand-rolled TensorRT decode logic needed
here the way RF-DETR's NMS-free output required. Shares
``_video_bench_common.py``'s FPS/accuracy measurement code with
benchmark_jetson.py so both model families are measured the same way.

Needs ``ultralytics`` + a matching ``torchvision`` (built from source against
this Jetson's exact NVIDIA torch build — no prebuilt wheel exists for this
JetPack/torch combo; see JETSON.md's "Best-YOLO Optimization" section for the
build steps) installed on this venv, on top of the RF-DETR pipeline's
torch/tensorrt/cusparselt setup.

Usage
-----
    python3 benchmark_yolo_jetson.py \\
        --weights weights/yolo11m_freeze21.pt \\
        --videos-dir gaza_road_videos/ \\
        --val-images val_images --val-labels val_labels \\
        --output benchmark_results_yolo.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from _rfdetr_trt_common import UltralyticsModel  # noqa: E402 — shared wrapper, not RF-DETR-specific
from _video_bench_common import benchmark_videos, evaluate_accuracy  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("benchmark_yolo_jetson")

CSV_FIELDS = [
    "model_name", "precision", "engine_build_ok", "fps_video_mean",
    "latency_ms_p50", "latency_ms_p99",
    "precision_at_0.4", "recall_at_0.4", "fp_per_image", "n_gt_boxes", "n_images", "notes",
]


def _patch_torch_onnx_export_for_old_nvidia_torch() -> None:
    """Ultralytics detects "torch 2.4.x" by version string and unconditionally passes
    ``dynamo=False`` to ``torch.onnx.export`` (``ultralytics/utils/export/engine.py``).
    NVIDIA's Jetson-specific torch build (``2.4.0a0+...nv24.07``) reports as 2.4.x but
    was snapshotted before upstream torch actually added the ``dynamo`` parameter to
    ``torch.onnx.export`` — so the real function rejects it. Strip any kwarg the
    installed torch doesn't actually accept, rather than patching ultralytics itself.
    """
    import inspect

    import torch

    real_export = torch.onnx.export
    accepted = set(inspect.signature(real_export).parameters)
    unsupported = {"dynamo"} - accepted
    if not unsupported:
        return  # this torch build already supports it — nothing to patch

    def _export_compat(*args, **kwargs):
        for k in unsupported:
            kwargs.pop(k, None)
        return real_export(*args, **kwargs)

    torch.onnx.export = _export_compat
    logger.info("Patched torch.onnx.export to drop unsupported kwarg(s) %s "
                "(NVIDIA Jetson torch build predates upstream torch.onnx.export's "
                "own support for them).", unsupported)


def export_engine(weights: Path, half: bool, imgsz: int) -> Path:
    from ultralytics import YOLO

    _patch_torch_onnx_export_for_old_nvidia_torch()
    model = YOLO(str(weights))
    exported = model.export(format="engine", half=half, device=0, imgsz=imgsz)
    return Path(exported)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="Path to the transferred YOLO .pt checkpoint")
    p.add_argument("--model-name", default="yolo11m_freeze21")
    p.add_argument("--imgsz", type=int, default=640, help="Matches the checkpoint's training imgsz")
    p.add_argument("--videos-dir", required=True)
    p.add_argument("--val-images", default=None)
    p.add_argument("--val-labels", default=None)
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--fps-threshold", type=float, default=0.35)
    p.add_argument("--output", default="benchmark_results_yolo.csv")
    args = p.parse_args()

    weights = Path(args.weights)
    if not weights.is_file():
        raise SystemExit(f"Weights not found: {weights}")

    videos_dir = Path(args.videos_dir)
    video_paths = sorted(videos_dir.glob("*.mp4"))
    logger.info("Found %d video clips in %s", len(video_paths), videos_dir)

    rows = []
    for precision, half in [("FP32", False), ("FP16", True)]:
        logger.info("=== %s ===", precision)
        engine_path = weights.parent / f"{weights.stem}_{precision.lower()}.engine"
        if engine_path.is_file():
            logger.info("Engine already exists, skipping export: %s", engine_path.name)
        else:
            try:
                exported = export_engine(weights, half=half, imgsz=args.imgsz)
                exported.rename(engine_path) if exported != engine_path else None
            except Exception as e:  # noqa: BLE001
                logger.error("Ultralytics export FAILED for %s: %s", precision, e)
                rows.append({"model_name": args.model_name, "precision": precision,
                             "engine_build_ok": False, "notes": f"EXPORT FAILED: {e}"})
                continue

        model = UltralyticsModel(engine_path, imgsz=args.imgsz)
        video_stats = benchmark_videos(model, video_paths, threshold=args.fps_threshold)

        acc = {"precision": None, "recall": None, "fp_per_image": None, "n_gt": 0, "n_images": 0}
        if args.val_images and args.val_labels:
            acc = evaluate_accuracy(model, Path(args.val_images), Path(args.val_labels), threshold=args.conf)

        notes = ""
        if acc["recall"] is not None and acc["n_gt"] > 0 and acc["recall"] < 0.3:
            notes = f"WARNING: recall collapsed to {acc['recall']:.2f} at this precision."

        rows.append({
            "model_name": args.model_name,
            "precision": precision,
            "engine_build_ok": True,
            "fps_video_mean": round(video_stats["mean_fps"], 1),
            "latency_ms_p50": round(video_stats["p50_ms"], 2),
            "latency_ms_p99": round(video_stats["p99_ms"], 2),
            "precision_at_0.4": round(acc["precision"], 3) if acc["precision"] is not None else "",
            "recall_at_0.4": round(acc["recall"], 3) if acc["recall"] is not None else "",
            "fp_per_image": round(acc["fp_per_image"], 3) if acc["fp_per_image"] is not None else "",
            "n_gt_boxes": acc["n_gt"],
            "n_images": acc["n_images"],
            "notes": notes,
        })
        del model

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    logger.info("Results saved: %s", out_path)

    print("\n" + "=" * 90)
    print(f"{'Precision':<8}  {'Built':>6}  {'FPS':>8}  {'p50ms':>7}  {'P@0.4':>6}  {'R@0.4':>6}  {'FP/img':>7}")
    print("-" * 90)
    for r in rows:
        print(f"{r['precision']:<8}  {str(r.get('engine_build_ok')):>6}  "
              f"{str(r.get('fps_video_mean', '')):>8}  {str(r.get('latency_ms_p50', '')):>7}  "
              f"{str(r.get('precision_at_0.4', '')):>6}  {str(r.get('recall_at_0.4', '')):>6}  "
              f"{str(r.get('fp_per_image', '')):>7}")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

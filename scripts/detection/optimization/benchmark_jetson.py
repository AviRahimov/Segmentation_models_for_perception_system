#!/usr/bin/env python3
"""Stage 2: RF-DETR-m TensorRT engine build + Jetson benchmark.

Run this ON THE JETSON, in the minimal optimization venv (no full repo
checkout needed — only this file + ``_rfdetr_trt_common.py`` + the
transferred ``.onnx`` + video clips + a small labeled validation subset).

Pre-flight (run manually once, see JETSON.md)
----------------------------------------------
    sudo nvpmodel -m 0
    sudo jetson_clocks
    nvpmodel -q     # confirm MAXN

What it does
------------
For FP32 and FP16 (this round's confirmed scope — no INT8, see the plan):
  1. Build a TRT engine from the transferred .onnx via trtexec.
  2. Real-video FPS + latency (p50/p99) over every clip in --videos-dir —
     decode+preprocess+infer, matching actual deployment conditions rather
     than a synthetic-tensor-only number.
  3. A coarse accuracy sanity check at a fixed 0.4 confidence operating point
     (precision/recall/FP-per-image against real YOLO-seg-polygon ground
     truth) — not a full mAP curve (that machinery lives in
     scripts/detection/evaluation/_ap_utils.py, which needs the full repo +
     ultralytics/rfdetr installed; deliberately not transplanted onto this
     minimal Jetson venv). This is a regression guard against a badly broken
     engine, not a publishable accuracy number — re-verify anything
     promising with the real leaderboard.py machinery back on the dev PC.

Results are written to reports/detection/optimization/benchmark_results.csv
(create the directory yourself before scp'ing this back, or run
generate_report.py on the dev PC once the CSV is copied over).

Usage
-----
    python3 benchmark_jetson.py \\
        --onnx weights/detection/optimization/rfdetr-m.onnx \\
        --videos-dir gaza_road_videos/ \\
        --val-images Detection_Dataset_valid/images \\
        --val-labels Detection_Dataset_valid/labels \\
        --output benchmark_results.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from _rfdetr_trt_common import RFDETRTensorRTEngine  # noqa: E402
from _video_bench_common import benchmark_videos, evaluate_accuracy  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("benchmark_jetson")

CSV_FIELDS = [
    "model_name", "precision", "harness", "engine_build_ok", "fps_video_mean",
    "latency_ms_p50", "latency_ms_p99",
    "precision_at_0.4", "recall_at_0.4", "fp_per_image", "n_gt_boxes", "n_images", "notes",
]
# harness="naive": cv2/numpy CPU preprocess, full stream sync every frame, no CUDA graph.
# harness="optimized": GPU-side preprocess + CUDA-graph replay (falls back to no-graph if
# capture fails on this TRT build — see _rfdetr_trt_common.RFDETRTensorRTEngine's docstring).
_HARNESS_CONFIGS = {
    "naive":     {"gpu_preprocess": False, "cuda_graph": False},
    "optimized": {"gpu_preprocess": True,  "cuda_graph": True},
}
_CLASS_NAMES = ["Military Vehicle", "person"]  # matches Detection_Dataset's data.yaml order


# --------------------------------------------------------------------------- #
# Engine build                                                                  #
# --------------------------------------------------------------------------- #

def _trtexec_path() -> str | None:
    p = shutil.which("trtexec")
    if p:
        return p
    for candidate in ["/usr/src/tensorrt/bin/trtexec", "/usr/local/bin/trtexec"]:
        if Path(candidate).is_file():
            return candidate
    return None


def build_engine(onnx_path: Path, engine_path: Path, fp16: bool,
                  workspace_mib: int = 4096) -> tuple[bool, str]:
    trtexec = _trtexec_path()
    if trtexec is None:
        return False, "trtexec binary not found"

    # No --shapes flag: the ONNX was exported with a static (non-dynamic-axes)
    # input shape (see export_onnx.py), and trtexec rejects --shapes entirely
    # for static models ("Static model does not take explicit shapes").
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{workspace_mib}MiB",
        "--useCudaGraph",
    ]
    if fp16:
        cmd.append("--fp16")
    logger.info("Building %s engine:\n  %s", "FP16" if fp16 else "FP32", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return False, "trtexec timed out (30 min)"

    for line in result.stderr.splitlines():
        low = line.lower()
        if ("fallback" in low and "fp32" in low) or "insufficient workspace" in low or "myelin" in low:
            logger.warning("TRT BUILD WARNING: %s", line.strip())

    if result.returncode != 0 or not engine_path.is_file():
        tail = (result.stderr or result.stdout)[-3000:]
        logger.error("trtexec FAILED:\n%s", tail)
        return False, f"trtexec exit={result.returncode}"

    logger.info("Engine built: %s (%.1f MB)", engine_path.name, engine_path.stat().st_size / 1e6)
    return True, ""


# --------------------------------------------------------------------------- #
# Real-video FPS/latency + accuracy — see _video_bench_common.py (shared with  #
# benchmark_yolo_jetson.py so both model families use the same measurement     #
# methodology).                                                                #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", required=True, help="Path to the transferred rfdetr-m.onnx")
    p.add_argument("--model-name", default="rfdetr-m", help="Used for the .engine filename stem (e.g. rfdetr-s)")
    p.add_argument("--engine-dir", default=None, help="Where to write .engine files (default: alongside --onnx)")
    p.add_argument("--videos-dir", required=True, help="Directory of gaza-road clips for FPS benchmarking")
    p.add_argument("--val-images", default=None, help="Directory of labeled validation images (optional)")
    p.add_argument("--val-labels", default=None, help="Directory of matching YOLO-seg label .txt files (optional)")
    p.add_argument("--shape", type=int, nargs=2, default=[576, 576], metavar=("H", "W"))
    p.add_argument("--conf", type=float, default=0.4, help="Operating-point confidence for the accuracy check")
    p.add_argument("--fps-threshold", type=float, default=0.35,
                   help="Confidence threshold used for the FPS-only video pass (production default)")
    p.add_argument("--harness", choices=["naive", "optimized", "both"], default="both",
                   help="Which harness configuration(s) to benchmark (see _HARNESS_CONFIGS)")
    p.add_argument("--output", default="benchmark_results.csv")
    args = p.parse_args()

    harness_names = list(_HARNESS_CONFIGS) if args.harness == "both" else [args.harness]

    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        raise SystemExit(f"ONNX file not found: {onnx_path}")
    engine_dir = Path(args.engine_dir) if args.engine_dir else onnx_path.parent
    engine_dir.mkdir(parents=True, exist_ok=True)

    videos_dir = Path(args.videos_dir)
    video_paths = sorted(videos_dir.glob("*.mp4"))
    logger.info("Found %d video clips in %s", len(video_paths), videos_dir)

    rows = []
    for precision, fp16 in [("FP32", False), ("FP16", True)]:
        logger.info("=== %s ===", precision)
        engine_path = engine_dir / f"{args.model_name}_{precision.lower()}.engine"
        if engine_path.is_file():
            logger.info("Engine already exists, skipping build: %s", engine_path.name)
            ok, note = True, ""
        else:
            ok, note = build_engine(onnx_path, engine_path, fp16)

        if not ok:
            rows.append({"model_name": args.model_name, "precision": precision,
                         "engine_build_ok": False, "notes": f"BUILD FAILED: {note}"})
            continue

        for harness_name in harness_names:
            harness_kwargs = _HARNESS_CONFIGS[harness_name]
            logger.info("--- %s / harness=%s ---", precision, harness_name)
            engine = RFDETRTensorRTEngine(engine_path, input_size=tuple(args.shape), **harness_kwargs)
            video_stats = benchmark_videos(engine, video_paths, threshold=args.fps_threshold)

            acc = {"precision": None, "recall": None, "fp_per_image": None, "n_gt": 0, "n_images": 0}
            if args.val_images and args.val_labels:
                acc = evaluate_accuracy(engine, Path(args.val_images), Path(args.val_labels), threshold=args.conf)

            notes = ""
            if acc["recall"] is not None and acc["n_gt"] > 0 and acc["recall"] < 0.3:
                notes = (
                    f"WARNING: recall collapsed to {acc['recall']:.2f} at this precision — likely "
                    "degraded/suppressed logits (verify with a raw-output probe before trusting FPS "
                    "numbers for deployment; see plan's FP16 accuracy risk note)."
                )

            rows.append({
                "model_name": args.model_name,
                "precision": precision,
                "harness": harness_name,
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
            del engine

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    logger.info("Results saved: %s", out_path)

    print("\n" + "=" * 100)
    print(f"{'Precision':<8}  {'Harness':<10}  {'Built':>6}  {'FPS':>8}  {'p50ms':>7}  "
          f"{'P@0.4':>6}  {'R@0.4':>6}  {'FP/img':>7}")
    print("-" * 100)
    for r in rows:
        print(f"{r['precision']:<8}  {str(r.get('harness', '')):<10}  {str(r.get('engine_build_ok')):>6}  "
              f"{str(r.get('fps_video_mean', '')):>8}  {str(r.get('latency_ms_p50', '')):>7}  "
              f"{str(r.get('precision_at_0.4', '')):>6}  {str(r.get('recall_at_0.4', '')):>6}  "
              f"{str(r.get('fp_per_image', '')):>7}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

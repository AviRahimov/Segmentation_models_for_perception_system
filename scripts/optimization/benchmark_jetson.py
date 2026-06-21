#!/usr/bin/env python3
"""Stage 4: Jetson AGX Orin engine build + authoritative benchmark.

Run this script ON the Jetson after transferring all .onnx files from the
dev PC.

Pre-flight checks (run manually once before benchmarking)
---------------------------------------------------------
    sudo nvpmodel -m 0              # MAXN power mode (max performance)
    sudo jetson_clocks               # lock all clocks to maximum
    dpkg -l | grep tensorrt          # confirm TensorRT 10.x is installed
    ls /usr/local/cuda/lib64/libcusparse_lt.so*  # check cuSPARSELt for sparsity

What it does
------------
For each .onnx file found in --onnx-dir:
  1. Build a TRT engine using trtexec with appropriate flags.
  2. Benchmark latency with trtexec --avgRuns=100 --iterations=200.
  3. Run real-harness timing via TensorRTBackend (p50, p99).
  4. Run 30-minute soak test to detect thermal throttling.
  5. Validate mIoU from engine output against the ORFD validation set.

Results are written to reports/optimization/benchmark_results.csv.

Usage
-----
    python scripts/optimization/benchmark_jetson.py \\
        --onnx-dir weights/optimization/ \\
        --val-data datasets/Final_Dataset \\
        --output reports/optimization/benchmark_results.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts" / "training"))

import train_orfd as _t

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark_jetson")

CSV_FIELDS = [
    "variant_name", "backbone", "precision", "sparsity", "resolution",
    "miou_pytorch", "miou_engine", "latency_ms_p50", "latency_ms_p99",
    "fps", "sustained_fps_30min", "notes",
]

# Jetson Orin workspace: 8 GB is safe for B2.
_TRT_WORKSPACE_GB = 8


# --------------------------------------------------------------------------- #
# Engine build                                                                  #
# --------------------------------------------------------------------------- #

def _trtexec_path() -> str:
    p = shutil.which("trtexec")
    if p is None:
        for candidate in ["/usr/src/tensorrt/bin/trtexec", "/usr/local/bin/trtexec"]:
            if Path(candidate).is_file():
                return candidate
        raise RuntimeError(
            "trtexec not found. Ensure TensorRT is installed and on PATH.\n"
            "On Jetson: sudo ln -s /usr/src/tensorrt/bin/trtexec /usr/local/bin/trtexec"
        )
    return p


def _parse_variant_flags(onnx_name: str) -> dict:
    """Infer TRT flags and metadata from the ONNX filename."""
    name = onnx_name.lower()
    is_int8   = "int8" in name or "qat" in name
    is_sparse = "sparse" in name
    is_fp16   = True  # always enable fp16 as a fallback layer

    # Resolution from filename e.g. baseline_fp16_256x256.onnx → 256
    resolution = 256
    for part in name.replace("_", "x").split("x"):
        if part.isdigit() and 128 <= int(part) <= 1024:
            resolution = int(part)
            break

    return {
        "is_int8":    is_int8,
        "is_sparse":  is_sparse,
        "is_fp16":    is_fp16,
        "resolution": resolution,
        "precision":  "INT8" if is_int8 else "FP16",
        "sparsity":   "2:4" if is_sparse else "none",
    }


def _build_engine(onnx_path: Path, engine_path: Path, flags: dict) -> bool:
    """Call trtexec to build a TRT engine.  Returns True on success."""
    trtexec = _trtexec_path()
    res = flags["resolution"]

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--shapes=pixel_values:1x3x{res}x{res}",
        f"--memPoolSize=workspace:{_TRT_WORKSPACE_GB * 1024}MiB",
        "--useCudaGraph",
        "--noDataTransfers",
    ]
    if flags["is_fp16"]:
        cmd.append("--fp16")
    if flags["is_int8"]:
        cmd.append("--int8")
    if flags["is_sparse"]:
        cmd.append("--sparsity=enable")

    logger.info("Building TRT engine:\n  %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        logger.error("trtexec timed out (30 min) for %s", onnx_path.name)
        return False

    # Scan for FP32 fallback warnings — logged but not fatal.
    for line in result.stderr.splitlines():
        if "fallback" in line.lower() and "fp32" in line.lower():
            logger.warning("TRT FALLBACK WARNING: %s", line.strip())

    if result.returncode != 0:
        logger.error("trtexec FAILED:\n%s", result.stderr[-3000:])
        return False

    if not engine_path.is_file():
        logger.error("trtexec succeeded but engine not found at %s", engine_path)
        return False

    logger.info("Engine built: %s  (%.1f MB)", engine_path, engine_path.stat().st_size / 1e6)
    return True


# --------------------------------------------------------------------------- #
# Latency benchmark                                                             #
# --------------------------------------------------------------------------- #

def _trtexec_latency(engine_path: Path, flags: dict) -> dict:
    """Run trtexec latency benchmark.  Returns p50/p99 dict from trtexec output."""
    trtexec = _trtexec_path()
    res = flags["resolution"]
    cmd = [
        trtexec,
        f"--loadEngine={engine_path}",
        f"--shapes=pixel_values:1x3x{res}x{res}",
        "--avgRuns=100",
        "--iterations=200",
        "--useCudaGraph",
        "--noDataTransfers",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    p50 = p99 = None
    for line in result.stdout.splitlines() + result.stderr.splitlines():
        # trtexec prints: "mean: 5.123 ms" "percentile: 5.456 ms at 99%"
        if "mean" in line.lower() and "ms" in line:
            parts = line.split()
            for i, tok in enumerate(parts):
                if tok.lower() in ("mean:", "mean") and i + 1 < len(parts):
                    try:
                        p50 = float(parts[i + 1].rstrip("ms"))
                    except ValueError:
                        pass
        if "99%" in line and "ms" in line:
            parts = line.split()
            for i, tok in enumerate(parts):
                if "ms" in tok:
                    try:
                        p99 = float(tok.rstrip("ms"))
                    except ValueError:
                        pass
    return {"latency_ms_p50_trtexec": p50, "latency_ms_p99_trtexec": p99}


def _harness_latency(engine_path: Path, resolution: int, n_frames: int = 300) -> dict:
    """Real-harness latency via TensorRTBackend.  Returns p50/p99 in ms."""
    try:
        from perception.models.backends.tensorrt import TensorRTBackend
    except ImportError:
        logger.warning("TensorRTBackend not importable — skipping harness latency.")
        return {"latency_ms_p50": None, "latency_ms_p99": None}

    backend = TensorRTBackend(str(engine_path))
    dummy = torch.zeros(1, 3, resolution, resolution, dtype=torch.float32).cuda()

    # Warm-up
    for _ in range(20):
        backend.infer(dummy)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_frames):
        t0 = time.perf_counter()
        backend.infer(dummy)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    p50 = float(np.percentile(times, 50))
    p99 = float(np.percentile(times, 99))
    fps = 1000.0 / p50 if p50 > 0 else 0.0
    logger.info("Harness latency: p50=%.2f ms  p99=%.2f ms  fps=%.1f", p50, p99, fps)
    return {"latency_ms_p50": p50, "latency_ms_p99": p99, "fps": fps}


def _soak_test(engine_path: Path, resolution: int, duration_s: int = 1800) -> float:
    """Run sustained inference for duration_s seconds.  Returns sustained FPS."""
    try:
        from perception.models.backends.tensorrt import TensorRTBackend
    except ImportError:
        logger.warning("TensorRTBackend not importable — skipping soak test.")
        return 0.0

    logger.info("Starting %d-minute soak test ...", duration_s // 60)
    backend = TensorRTBackend(str(engine_path))
    dummy = torch.zeros(1, 3, resolution, resolution, dtype=torch.float32).cuda()

    t_end = time.perf_counter() + duration_s
    count = 0
    t_start = time.perf_counter()
    while time.perf_counter() < t_end:
        backend.infer(dummy)
        torch.cuda.synchronize()
        count += 1

    elapsed = time.perf_counter() - t_start
    fps = count / elapsed
    logger.info("Soak test done: %.1f FPS sustained over %.1f min", fps, elapsed / 60)
    return fps


# --------------------------------------------------------------------------- #
# mIoU validation from engine                                                  #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _engine_miou(engine_path: Path, val_data: str, resolution: int) -> float:
    """Validate mIoU from TRT engine output on the ORFD validation set."""
    try:
        from perception.models.backends.tensorrt import TensorRTBackend
    except ImportError:
        logger.warning("TensorRTBackend not importable — skipping engine mIoU.")
        return float("nan")

    from perception.datasets.orfd_torch import ORFDDataset

    backend = TensorRTBackend(str(engine_path))
    val_ds = ORFDDataset(val_data, split="validation", augment=False, input_size=resolution)
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    all_preds, all_labels = [], []
    for images, labels in loader:
        images_cuda = images.cuda()
        # Engine output: (1, C, H/4, W/4) — upsample to label size.
        raw = backend.infer(images_cuda)
        if isinstance(raw, torch.Tensor):
            logits = raw
        else:
            logits = torch.from_numpy(raw).cuda()
        logits = torch.nn.functional.interpolate(
            logits.float(), size=(resolution, resolution),
            mode="bilinear", align_corners=False,
        )
        preds = logits.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)

    preds_cat  = torch.cat(all_preds,  dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    miou, per_class = _t.compute_miou(preds_cat, labels_cat)
    logger.info("Engine mIoU: %.4f  per-class: %s", miou,
                [f"{v:.3f}" if not (isinstance(v, float) and v != v) else "nan" for v in per_class])
    return miou


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser(description="Stage 4: Jetson TRT engine build + benchmark")
    p.add_argument("--onnx-dir",     default="weights/optimization",
                   help="Directory containing .onnx files to benchmark")
    p.add_argument("--engine-dir",   default=None,
                   help="Where to save .engine files (default: same as --onnx-dir)")
    p.add_argument("--val-data",     default="datasets/Final_Dataset")
    p.add_argument("--output",       default="reports/optimization/benchmark_results.csv")
    p.add_argument("--soak",         action="store_true",
                   help="Run 30-minute soak test per variant (adds ~2h total)")
    p.add_argument("--soak-duration",type=int, default=1800,
                   help="Soak test duration in seconds (default: 1800 = 30 min)")
    p.add_argument("--pytorch-ref",  default="weights/orfd/frozen_backbone/segformer-b2/best.pth",
                   help="PyTorch baseline for reference mIoU (optional)")
    args = p.parse_args()

    onnx_dir = Path(args.onnx_dir)
    if not onnx_dir.is_absolute():
        onnx_dir = _ROOT / onnx_dir
    engine_dir = Path(args.engine_dir) if args.engine_dir else onnx_dir
    if not engine_dir.is_absolute():
        engine_dir = _ROOT / engine_dir
    engine_dir.mkdir(parents=True, exist_ok=True)

    val_data = Path(args.val_data)
    if not val_data.is_absolute():
        val_data = _ROOT / val_data

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = _ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    onnx_files = sorted(onnx_dir.glob("*.onnx"))
    if not onnx_files:
        logger.error("No .onnx files found in %s", onnx_dir)
        return 1

    logger.info("Found %d ONNX files: %s", len(onnx_files), [f.name for f in onnx_files])

    # ---- Optional: compute PyTorch reference mIoU ----
    pytorch_ref_miou: dict[int, float] = {}
    ref_ckpt = _ROOT / args.pytorch_ref
    if ref_ckpt.is_file():
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        from perception.datasets.orfd_torch import ORFDDataset

        ckpt_data = torch.load(str(ref_ckpt), map_location="cpu", weights_only=True)
        state_dict = ckpt_data.get("net", ckpt_data) if isinstance(ckpt_data, dict) else ckpt_data
        from perception.models.semantic.segformer import _remap_segformer_keys
        state_dict = _remap_segformer_keys(state_dict)
        n_labels = state_dict["decode_head.classifier.weight"].shape[0]
        hf_base = "nvidia/segformer-b2-finetuned-ade-512-512"

        for onnx_f in onnx_files:
            flags = _parse_variant_flags(onnx_f.stem)
            res = flags["resolution"]
            if res in pytorch_ref_miou:
                continue
            processor = SegformerImageProcessor.from_pretrained(hf_base)
            processor.size = {"height": res, "width": res}
            model = SegformerForSemanticSegmentation.from_pretrained(
                hf_base, num_labels=n_labels, ignore_mismatched_sizes=True
            )
            model.load_state_dict(state_dict, strict=True)
            model = model.eval().cuda()
            val_ds = ORFDDataset(str(val_data), split="validation", augment=False, input_size=res)
            loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)
            criterion = lambda logits, labels: _t._dice_ce_loss(logits, labels)
            _, ref_miou = _t.evaluate(model, processor, loader, criterion, "cuda", fp16=False)
            pytorch_ref_miou[res] = ref_miou
            logger.info("PyTorch reference mIoU @ %d px: %.4f", res, ref_miou)
            del model

    # ---- Process each ONNX ----
    rows = []
    for onnx_path in onnx_files:
        flags = _parse_variant_flags(onnx_path.stem)
        engine_path = engine_dir / onnx_path.with_suffix(".engine").name
        res = flags["resolution"]

        logger.info("=== Processing: %s ===", onnx_path.name)

        # Build engine
        if not _build_engine(onnx_path, engine_path, flags):
            rows.append({
                "variant_name": onnx_path.stem,
                "backbone": "segformer-b2",
                "precision": flags["precision"],
                "sparsity": flags["sparsity"],
                "resolution": res,
                "miou_pytorch": pytorch_ref_miou.get(res, ""),
                "miou_engine": "",
                "latency_ms_p50": "",
                "latency_ms_p99": "",
                "fps": "",
                "sustained_fps_30min": "",
                "notes": "ENGINE BUILD FAILED",
            })
            continue

        # Latency from trtexec
        trt_lat = _trtexec_latency(engine_path, flags)

        # Latency from harness
        harness_lat = _harness_latency(engine_path, res)

        # Soak test
        sustained_fps = ""
        if args.soak:
            sustained_fps = round(_soak_test(engine_path, res, args.soak_duration), 1)

        # mIoU from engine
        eng_miou = _engine_miou(engine_path, str(val_data), res)

        # Reference mIoU
        ref_miou = pytorch_ref_miou.get(res, "")

        # Flag mIoU drop > 1% vs reference
        notes = ""
        if ref_miou and not (isinstance(eng_miou, float) and eng_miou != eng_miou):
            drop = ref_miou - eng_miou
            if drop > 0.01:
                notes = f"WARNING: engine mIoU drop {drop:.3f} > 0.01"

        rows.append({
            "variant_name":       onnx_path.stem,
            "backbone":           "segformer-b2",
            "precision":          flags["precision"],
            "sparsity":           flags["sparsity"],
            "resolution":         res,
            "miou_pytorch":       round(ref_miou, 4) if ref_miou else "",
            "miou_engine":        round(eng_miou, 4) if eng_miou == eng_miou else "",
            "latency_ms_p50":     round(harness_lat.get("latency_ms_p50") or
                                        trt_lat.get("latency_ms_p50_trtexec") or 0, 2),
            "latency_ms_p99":     round(harness_lat.get("latency_ms_p99") or
                                        trt_lat.get("latency_ms_p99_trtexec") or 0, 2),
            "fps":                round(harness_lat.get("fps") or 0, 1),
            "sustained_fps_30min": sustained_fps,
            "notes":              notes,
        })

    # ---- Write CSV ----
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Benchmark results saved: %s", out_path)

    # ---- Print summary ----
    print("\n" + "=" * 90)
    print(f"{'Variant':<40}  {'Prec':>5}  {'mIoU_eng':>9}  {'p50ms':>7}  {'FPS':>6}")
    print("-" * 90)
    for r in rows:
        print(
            f"{r['variant_name'][:40]:<40}  {r['precision']:>5}  "
            f"{str(r['miou_engine']):>9}  {str(r['latency_ms_p50']):>7}  {str(r['fps']):>6}"
        )
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

def _trtexec_path() -> str | None:
    """Return path to trtexec binary, or None if not available."""
    p = shutil.which("trtexec")
    if p:
        return p
    for candidate in ["/usr/src/tensorrt/bin/trtexec", "/usr/local/bin/trtexec"]:
        if Path(candidate).is_file():
            return candidate
    return None


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
    """Build a TRT engine from ONNX using the Python TensorRT API.

    Falls back to trtexec subprocess if available, otherwise uses tensorrt Python
    bindings directly (works when trtexec binary is absent, e.g. on Jetson where
    only the runtime libraries are installed without the samples package).
    """
    trtexec = _trtexec_path()
    res = flags["resolution"]

    if trtexec:
        # Use trtexec when available — produces identical engines.
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
        logger.info("Building TRT engine via trtexec:\n  %s", " ".join(cmd))
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        except subprocess.TimeoutExpired:
            logger.error("trtexec timed out (30 min) for %s", onnx_path.name)
            return False
        for line in result.stderr.splitlines():
            if "fallback" in line.lower() and "fp32" in line.lower():
                logger.warning("TRT FALLBACK: %s", line.strip())
        if result.returncode != 0:
            logger.error("trtexec FAILED:\n%s", result.stderr[-3000:])
            return False
    else:
        # trtexec not available — build via Python TensorRT API.
        logger.info("trtexec not found; building engine via Python TRT API ...")
        try:
            import tensorrt as trt  # type: ignore
        except ImportError:
            logger.error("tensorrt Python package not found.")
            return False

        trt_logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(trt_logger)
        try:
            # TRT < 10: EXPLICIT_BATCH flag required. TRT 10+: default, flag deprecated.
            nf = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        except AttributeError:
            nf = 0
        network = builder.create_network(nf)
        parser = trt.OnnxParser(network, trt_logger)

        with open(str(onnx_path), "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error("ONNX parse error: %s", parser.get_error(i))
                return False

        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, _TRT_WORKSPACE_GB * (1 << 30)
        )
        if flags["is_fp16"]:
            config.set_flag(trt.BuilderFlag.FP16)
        if flags["is_int8"]:
            config.set_flag(trt.BuilderFlag.INT8)
        if flags["is_sparse"]:
            try:
                config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
            except AttributeError:
                logger.warning("SPARSE_WEIGHTS flag not available in this TRT version.")

        logger.info(
            "Building %s  fp16=%s int8=%s sparse=%s — may take 5-15 min ...",
            onnx_path.name, flags["is_fp16"], flags["is_int8"], flags["is_sparse"],
        )
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            logger.error("TRT engine build failed for %s", onnx_path.name)
            return False

        engine_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(engine_path), "wb") as f:
            f.write(serialized)

    if not engine_path.is_file():
        logger.error("Engine file not found after build: %s", engine_path)
        return False

    logger.info("Engine built: %s  (%.1f MB)", engine_path.name, engine_path.stat().st_size / 1e6)
    return True


# --------------------------------------------------------------------------- #
# TRT engine helpers (shared by latency, soak, and mIoU sections)              #
# --------------------------------------------------------------------------- #

def _load_trt_context(engine_path: Path):
    """Deserialize a TRT engine and return (context, out_buf).

    out_buf dtype is inferred from the engine's logits binding;
    defaults to FP16 if detection fails.
    Input binding is always FP32 — matches the ONNX export dtype.
    """
    import tensorrt as trt  # type: ignore
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    context = engine.create_execution_context()
    out_shape = tuple(context.get_tensor_shape("logits"))
    try:
        trt_dt = engine.get_tensor_dtype("logits")
        out_dtype = torch.float32 if trt_dt == trt.DataType.FLOAT else torch.float16
    except Exception:
        out_dtype = torch.float16
    out_buf = torch.empty(out_shape, dtype=out_dtype, device="cuda")
    return context, out_buf


def _trt_infer(context, pixel_values: torch.Tensor, out_buf: torch.Tensor) -> torch.Tensor:
    """Run one synchronous TRT inference pass; returns a clone of out_buf."""
    stream = torch.cuda.current_stream().cuda_stream
    context.set_tensor_address("pixel_values", pixel_values.data_ptr())
    context.set_tensor_address("logits", out_buf.data_ptr())
    context.execute_async_v3(stream)
    torch.cuda.current_stream().synchronize()
    return out_buf.clone()


# --------------------------------------------------------------------------- #
# Latency benchmark                                                             #
# --------------------------------------------------------------------------- #

def _trtexec_latency(engine_path: Path, flags: dict) -> dict:
    """Run trtexec latency benchmark.  Returns p50/p99 dict from trtexec output."""
    trtexec = _trtexec_path()
    if trtexec is None:
        logger.info("trtexec not available — skipping trtexec latency (harness latency used).")
        return {"latency_ms_p50_trtexec": None, "latency_ms_p99_trtexec": None}
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
    """Real-harness latency via direct TRT API.  Returns p50/p99 in ms."""
    try:
        context, out_buf = _load_trt_context(engine_path)
    except Exception as e:
        logger.warning("Failed to load TRT engine for harness latency: %s", e)
        return {"latency_ms_p50": None, "latency_ms_p99": None, "fps": None}

    dummy = torch.zeros(1, 3, resolution, resolution, dtype=torch.float32, device="cuda")

    for _ in range(20):
        _trt_infer(context, dummy, out_buf)

    times = []
    for _ in range(n_frames):
        t0 = time.perf_counter()
        _trt_infer(context, dummy, out_buf)
        times.append((time.perf_counter() - t0) * 1000)

    p50 = float(np.percentile(times, 50))
    p99 = float(np.percentile(times, 99))
    fps = 1000.0 / p50 if p50 > 0 else 0.0
    logger.info("Harness latency: p50=%.2f ms  p99=%.2f ms  fps=%.1f", p50, p99, fps)
    return {"latency_ms_p50": p50, "latency_ms_p99": p99, "fps": fps}


def _soak_test(engine_path: Path, resolution: int, duration_s: int = 1800) -> float:
    """Run sustained inference for duration_s seconds.  Returns sustained FPS."""
    try:
        context, out_buf = _load_trt_context(engine_path)
    except Exception as e:
        logger.warning("Failed to load TRT engine for soak test: %s", e)
        return 0.0

    logger.info("Starting %d-minute soak test ...", duration_s // 60)
    dummy = torch.zeros(1, 3, resolution, resolution, dtype=torch.float32, device="cuda")

    t_end = time.perf_counter() + duration_s
    count = 0
    t_start = time.perf_counter()
    while time.perf_counter() < t_end:
        _trt_infer(context, dummy, out_buf)
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
        context, out_buf = _load_trt_context(engine_path)
    except Exception as e:
        logger.warning("Failed to load TRT engine for mIoU: %s", e)
        return float("nan")

    from perception.datasets.orfd_torch import ORFDDataset

    val_ds = ORFDDataset(val_data, split="validation", augment=False, input_size=resolution)
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    all_preds, all_labels = [], []
    for images, labels in loader:
        # Input is always FP32 — matches the ONNX input binding.
        images_cuda = images.cuda().float()
        logits = _trt_infer(context, images_cuda, out_buf)
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

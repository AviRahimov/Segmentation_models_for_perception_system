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
  3. Real decode+preprocess+infer FPS/latency over actual video clips, for
     each requested --harness config (naive: CPU preprocess, no CUDA graph;
     optimized: GPU preprocess + CUDA-graph replay) — see
     ``_segformer_trt_common.py``.
  4. Run 30-minute soak test to detect thermal throttling.
  5. Validate mIoU from engine output against the ORFD validation set.

Results are written to reports/segmentation/optimization/benchmark_results.csv.

Usage
-----
    python scripts/segmentation/optimization/benchmark_jetson.py \\
        --onnx-dir weights/segmentation/optimization/ \\
        --val-data datasets/segmentation/ORFD \\
        --videos-dir data/videos \\
        --output reports/segmentation/optimization/benchmark_results.csv
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

_ROOT = Path(__file__).resolve().parents[3]
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts" / "segmentation" / "training"))
sys.path.insert(0, str(_ROOT / "scripts" / "segmentation"))
sys.path.insert(0, str(_HERE))

from _orfd_common import _dice_ce_loss, compute_miou, evaluate
from _segformer_checkpoint_common import load_remapped_state_dict
from _segformer_trt_common import SegformerTensorRTEngine, benchmark_videos

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark_jetson")

CSV_FIELDS = [
    "variant_name", "backbone", "precision", "sparsity", "resolution", "harness",
    "miou_pytorch", "miou_engine", "latency_ms_p50", "latency_ms_p99",
    "fps", "sustained_fps_30min", "notes",
]

# harness="naive": cv2/numpy CPU preprocess, full stream sync every frame, no CUDA graph.
# harness="optimized": GPU-side preprocess + CUDA-graph replay (falls back to no-graph if
# capture fails — see _segformer_trt_common.SegformerTensorRTEngine's docstring).
_HARNESS_CONFIGS = {
    "naive":     {"gpu_preprocess": False, "cuda_graph": False},
    "optimized": {"gpu_preprocess": True,  "cuda_graph": True},
}

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


def _has_qdq_nodes(onnx_path: Path) -> bool:
    """Return True if the ONNX has QuantizeLinear nodes (embedded INT8 scale factors).

    TRT requires either QDQ nodes or an external calibration cache to build an
    INT8 engine.  Filenames may say 'int8' or 'qat' but the ONNX may lack the
    nodes if the modelopt QDQ export failed and a plain torch.onnx fallback was
    used (e.g. sparse model exported before QAT).
    """
    try:
        import onnx  # type: ignore
        model = onnx.load(str(onnx_path))
        return any(n.op_type == "QuantizeLinear" for n in model.graph.node)
    except Exception:
        return False


def _parse_variant_flags(onnx_name: str, onnx_path: Path | None = None) -> dict:
    """Infer TRT flags and metadata from the ONNX filename (+ optional graph scan)."""
    name = onnx_name.lower()
    is_sparse = "sparse" in name
    is_fp16   = True  # always enable fp16 as a fallback layer

    # Only use INT8 if the ONNX actually has embedded quantization scale nodes.
    # Filename alone is unreliable: sparse model is exported before QAT so it
    # has no QDQ nodes even though the name contains "int8".
    name_suggests_int8 = "int8" in name or "qat" in name
    if name_suggests_int8 and onnx_path is not None:
        is_int8 = _has_qdq_nodes(onnx_path)
        if not is_int8:
            logger.warning(
                "%s: filename suggests INT8 but ONNX has no QDQ nodes — "
                "building FP16 engine to avoid calibration failure.",
                onnx_path.name,
            )
    else:
        is_int8 = name_suggests_int8

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
        "precision":  "INT8" if is_int8 else "FP32",
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
        # No --shapes flag: export_onnx.py exports with a static (non-dynamic-axes)
        # input shape, and trtexec rejects --shapes entirely for static models
        # ("Static model does not take explicit shapes") — same lesson already
        # applied in scripts/detection/optimization/benchmark_jetson.py.
        cmd = [
            trtexec,
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
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
        # FP16 is only safe alongside INT8: the fake-quant ops in the QAT ONNX
        # force FP32 output, anchoring precision. Pure FP16 causes NaN in
        # SegFormer's attention softmax (overflow on Jetson Ampere).
        if flags["is_fp16"] and flags["is_int8"]:
            config.set_flag(trt.BuilderFlag.FP16)
        elif flags["is_fp16"]:
            logger.warning(
                "Pure FP16 skipped in Python TRT path (NaN in SegFormer attention) "
                "— building FP32 engine instead."
            )
        if flags["is_int8"]:
            config.set_flag(trt.BuilderFlag.INT8)
        # SPARSE_WEIGHTS via Python TRT API produces degenerate logits on this
        # TRT build (all-class-0 predictions). The 2:4 pattern is already in the
        # ONNX weights; TRT runs correct dense kernels without this flag.
        if flags["is_sparse"]:
            logger.warning(
                "SPARSE_WEIGHTS flag skipped in Python TRT path — causes degenerate "
                "output. Sparse weight pattern in ONNX is preserved; dense kernels used."
            )

        logger.info(
            "Building %s  fp16=%s int8=%s sparse=%s — may take 5-15 min ...",
            onnx_path.name, flags["is_fp16"] and flags["is_int8"],
            flags["is_int8"], False,
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
# Latency benchmark                                                             #
# --------------------------------------------------------------------------- #

def _trtexec_latency(engine_path: Path, flags: dict) -> dict:
    """Run trtexec latency benchmark.  Returns p50/p99 dict from trtexec output."""
    trtexec = _trtexec_path()
    if trtexec is None:
        logger.info("trtexec not available — skipping trtexec latency (harness latency used).")
        return {"latency_ms_p50_trtexec": None, "latency_ms_p99_trtexec": None}
    cmd = [
        trtexec,
        f"--loadEngine={engine_path}",
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


def _video_harness_fps(engine_path: Path, resolution: int, harness_name: str,
                        video_paths: list[Path]) -> dict:
    """Real decode+preprocess+infer FPS/latency over --videos-dir clips.

    Replaces the old synthetic-dummy-tensor timing loop: that never
    exercised preprocessing at all, so it structurally couldn't show any
    gain from GPU-side preprocessing. See _segformer_trt_common.py's
    module docstring.
    """
    if not video_paths:
        logger.warning("No videos found — skipping real-video FPS for harness=%s", harness_name)
        return {"latency_ms_p50": None, "latency_ms_p99": None, "fps": None}

    engine = SegformerTensorRTEngine(engine_path, input_size=resolution,
                                      **_HARNESS_CONFIGS[harness_name])
    stats = benchmark_videos(engine, video_paths)
    del engine
    logger.info("harness=%s: p50=%.2f ms  p99=%.2f ms  fps=%.1f",
                harness_name, stats["p50_ms"], stats["p99_ms"], stats["mean_fps"])
    return {"latency_ms_p50": stats["p50_ms"], "latency_ms_p99": stats["p99_ms"],
            "fps": stats["mean_fps"]}


def _soak_test(engine_path: Path, resolution: int, duration_s: int = 1800) -> float:
    """Run sustained inference for duration_s seconds.  Returns sustained FPS.

    Naive (no CUDA graph) tensor-in/tensor-out loop, matching the pre-existing
    soak methodology — this measures raw sustained engine throughput /
    thermal behavior, not harness-optimization gains.
    """
    try:
        engine = SegformerTensorRTEngine(engine_path, input_size=resolution,
                                          gpu_preprocess=False, cuda_graph=False)
    except Exception as e:
        logger.warning("Failed to load TRT engine for soak test: %s", e)
        return 0.0

    logger.info("Starting %d-minute soak test ...", duration_s // 60)
    dummy = torch.zeros(1, 3, resolution, resolution, dtype=torch.float32, device="cuda")

    t_end = time.perf_counter() + duration_s
    count = 0
    t_start = time.perf_counter()
    while time.perf_counter() < t_end:
        engine.infer_tensor(dummy)
        count += 1

    elapsed = time.perf_counter() - t_start
    fps = count / elapsed
    logger.info("Soak test done: %.1f FPS sustained over %.1f min", fps, elapsed / 60)
    del engine
    return fps


# --------------------------------------------------------------------------- #
# mIoU validation from engine                                                  #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _engine_miou(engine_path: Path, val_data: str, resolution: int) -> float:
    """Validate mIoU from TRT engine output on the ORFD validation set."""
    try:
        engine = SegformerTensorRTEngine(engine_path, input_size=resolution,
                                          gpu_preprocess=False, cuda_graph=False)
    except Exception as e:
        logger.warning("Failed to load TRT engine for mIoU: %s", e)
        return float("nan")

    from perception.datasets.orfd_torch import ORFDDataset

    val_ds = ORFDDataset(val_data, split="validation", augment=False, input_size=resolution)
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    all_preds, all_labels = [], []
    for images, labels in loader:
        # Input is always FP32 — matches the ONNX input binding.
        images_cuda = images.cuda().float()
        logits = engine.infer_tensor(images_cuda)
        logits = torch.nn.functional.interpolate(
            logits.float(), size=(resolution, resolution),
            mode="bilinear", align_corners=False,
        )
        preds = logits.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)
    del engine

    preds_cat  = torch.cat(all_preds,  dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    miou, per_class = compute_miou(preds_cat, labels_cat)
    logger.info("Engine mIoU: %.4f  per-class: %s", miou,
                [f"{v:.3f}" if not (isinstance(v, float) and v != v) else "nan" for v in per_class])
    return miou


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser(description="Stage 4: Jetson TRT engine build + benchmark")
    p.add_argument("--onnx-dir",     default="weights/segmentation/optimization",
                   help="Directory containing .onnx files to benchmark")
    p.add_argument("--engine-dir",   default=None,
                   help="Where to save .engine files (default: same as --onnx-dir)")
    p.add_argument("--val-data",     default="datasets/segmentation/ORFD")
    p.add_argument("--output",       default="reports/segmentation/optimization/benchmark_results.csv")
    p.add_argument("--soak",         action="store_true",
                   help="Run 30-minute soak test per variant (adds ~2h total)")
    p.add_argument("--soak-duration",type=int, default=1800,
                   help="Soak test duration in seconds (default: 1800 = 30 min)")
    p.add_argument("--pytorch-ref",  default="weights/segmentation/orfd/frozen_backbone/segformer-b2/best.pth",
                   help="PyTorch baseline for reference mIoU (optional)")
    p.add_argument("--videos-dir",   default="data/videos",
                   help="Directory of real video clips for FPS benchmarking (real decode+"
                        "preprocess+infer, not a synthetic dummy tensor)")
    p.add_argument("--harness",      choices=["naive", "optimized", "both"], default="both",
                   help="Which harness configuration(s) to benchmark (see _HARNESS_CONFIGS)")
    args = p.parse_args()

    harness_names = list(_HARNESS_CONFIGS) if args.harness == "both" else [args.harness]

    videos_dir = Path(args.videos_dir)
    if not videos_dir.is_absolute():
        videos_dir = _ROOT / videos_dir
    video_paths = sorted(videos_dir.glob("*.mp4")) if videos_dir.is_dir() else []
    logger.info("Found %d video clips in %s", len(video_paths), videos_dir)

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

        state_dict = load_remapped_state_dict(ref_ckpt)
        n_labels = state_dict["decode_head.classifier.weight"].shape[0]
        hf_base = "nvidia/segformer-b2-finetuned-ade-512-512"

        for onnx_f in onnx_files:
            flags = _parse_variant_flags(onnx_f.stem, onnx_f)
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
            criterion = lambda logits, labels: _dice_ce_loss(logits, labels)
            _, ref_miou = evaluate(model, processor, loader, criterion, "cuda", fp16=False)
            pytorch_ref_miou[res] = ref_miou
            logger.info("PyTorch reference mIoU @ %d px: %.4f", res, ref_miou)
            del model

    # ---- Process each ONNX ----
    rows = []
    for onnx_path in onnx_files:
        flags = _parse_variant_flags(onnx_path.stem, onnx_path)
        engine_path = engine_dir / onnx_path.with_suffix(".engine").name
        res = flags["resolution"]

        logger.info("=== Processing: %s ===", onnx_path.name)

        # Build engine (skip if already exists)
        if engine_path.is_file():
            logger.info("Engine already exists, skipping build: %s", engine_path.name)
        elif not _build_engine(onnx_path, engine_path, flags):
            rows.append({
                "variant_name": onnx_path.stem,
                "backbone": "segformer-b2",
                "precision": flags["precision"],
                "sparsity": flags["sparsity"],
                "resolution": res,
                "harness": "",
                "miou_pytorch": pytorch_ref_miou.get(res, ""),
                "miou_engine": "",
                "latency_ms_p50": "",
                "latency_ms_p99": "",
                "fps": "",
                "sustained_fps_30min": "",
                "notes": "ENGINE BUILD FAILED",
            })
            continue

        # Latency from trtexec (independent of --harness — trtexec's own timer)
        trt_lat = _trtexec_latency(engine_path, flags)

        # Soak test — once per variant, naive raw-executor loop regardless of --harness
        sustained_fps = ""
        if args.soak:
            sustained_fps = round(_soak_test(engine_path, res, args.soak_duration), 1)

        # mIoU from engine — once per variant, independent of --harness
        eng_miou = _engine_miou(engine_path, str(val_data), res)
        ref_miou = pytorch_ref_miou.get(res, "")

        notes = ""
        if ref_miou and not (isinstance(eng_miou, float) and eng_miou != eng_miou):
            drop = ref_miou - eng_miou
            if drop > 0.01:
                notes = f"WARNING: engine mIoU drop {drop:.3f} > 0.01"

        for harness_name in harness_names:
            logger.info("--- %s / harness=%s ---", onnx_path.name, harness_name)
            harness_lat = _video_harness_fps(engine_path, res, harness_name, video_paths)

            rows.append({
                "variant_name":       onnx_path.stem,
                "backbone":           "segformer-b2",
                "precision":          flags["precision"],
                "sparsity":           flags["sparsity"],
                "resolution":         res,
                "harness":            harness_name,
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
    print("\n" + "=" * 100)
    print(f"{'Variant':<40}  {'Prec':>5}  {'Harness':>10}  {'mIoU_eng':>9}  {'p50ms':>7}  {'FPS':>6}")
    print("-" * 100)
    for r in rows:
        print(
            f"{r['variant_name'][:40]:<40}  {r['precision']:>5}  {str(r.get('harness', '')):>10}  "
            f"{str(r['miou_engine']):>9}  {str(r['latency_ms_p50']):>7}  {str(r['fps']):>6}"
        )
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

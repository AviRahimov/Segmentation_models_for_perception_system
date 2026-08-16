#!/usr/bin/env python3
"""Stage 1: Export SegFormer-B2 baseline to FP16 ONNX with numerical validation.

Produces a validated ONNX file ready to be transferred to the Jetson for
TRT engine build.  Does NOT build TRT engines — that happens on Jetson via
benchmark_jetson.py.

Usage
-----
    # Baseline FP16 export (dev PC):
    python scripts/segmentation/optimization/export_onnx.py \\
        --checkpoint weights/segmentation/orfd/frozen_backbone/segformer-b2/best.pth \\
        --resolution 256

    # Also called internally by train_qat.py and train_sparse.py after
    # their modelopt export step (those scripts call _validate_onnx directly).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts" / "segmentation"))

from _segformer_checkpoint_common import build_segformer_from_checkpoint  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("export_onnx")

# Tolerance for ONNX-vs-PyTorch numerical validation.
_MAX_ABS_DIFF = 1e-2  # fp16 rounding can introduce ~1e-3; generous but not silent


_HF_BASES = {
    "segformer-b0": "nvidia/segformer-b0-finetuned-ade-512-512",
    "segformer-b1": "nvidia/segformer-b1-finetuned-ade-512-512",
    "segformer-b2": "nvidia/segformer-b2-finetuned-ade-512-512",
    "segformer-b3": "nvidia/segformer-b3-finetuned-ade-512-512",
    "segformer-b4": "nvidia/segformer-b4-finetuned-ade-512-512",
}


def _load_model(checkpoint: str, resolution: int, fp16: bool, device: str, variant: str = "segformer-b2"):
    """Load a SegFormer checkpoint from a local .pth, set processor to target resolution."""
    model, processor, n_labels = build_segformer_from_checkpoint(
        checkpoint, device, resolution=resolution, hf_base=_HF_BASES[variant], fp16=fp16
    )
    logger.info("Loaded: %s  variant=%s  res=%d  fp16=%s  classes=%d",
                Path(checkpoint).name, variant, resolution, fp16, n_labels)
    return model, processor


def export_fp16_onnx(
    checkpoint: str,
    resolution: int,
    output_dir: Path,
    device: str = "cuda",
    fp16: bool = True,
    variant: str = "segformer-b2",
) -> Path:
    """Export baseline checkpoint to FP16 ONNX.  Returns the saved .onnx path."""
    import onnx

    model, _ = _load_model(checkpoint, resolution, fp16, device, variant)

    output_dir.mkdir(parents=True, exist_ok=True)
    precision_tag = "fp16" if fp16 else "fp32"
    onnx_path = output_dir / f"baseline_{precision_tag}_{resolution}x{resolution}.onnx"

    dtype = torch.float16 if fp16 else torch.float32
    dummy = torch.zeros(1, 3, resolution, resolution, device=device, dtype=dtype)

    logger.info("Exporting → %s", onnx_path)
    torch.onnx.export(
        model,
        (dummy,),           # positional tuple — required in torch 2.x
        str(onnx_path),
        input_names=["pixel_values"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,       # force TorchScript exporter (stable, dict args broken in 2.11)
    )

    logger.info("Checking ONNX graph validity ...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX checker: OK")

    _validate_onnx(onnx_path, model, resolution, fp16, device)

    logger.info("Saved: %s", onnx_path)
    return onnx_path


def _validate_onnx(
    onnx_path: Path,
    pytorch_model: torch.nn.Module,
    resolution: int,
    fp16: bool,
    device: str,
) -> None:
    """Compare ONNX and PyTorch outputs on a random batch.  Raises if diff > tolerance."""
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed — skipping numerical validation.")
        return

    dtype_np = np.float16 if fp16 else np.float32
    np.random.seed(42)
    sample_np = np.random.randn(1, 3, resolution, resolution).astype(dtype_np)
    sample_pt = torch.from_numpy(sample_np).to(device)

    with torch.no_grad():
        logits_pt = pytorch_model(pixel_values=sample_pt).logits.float().cpu().numpy()

    # Try GPU EP first; fall back to CPU (covers Jetson and machines without ort-gpu).
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(str(onnx_path), providers=providers)
    except Exception:
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    logits_ort = sess.run(["logits"], {"pixel_values": sample_np})[0]  # (1, C, H/4, W/4)
    logits_ort_f32 = logits_ort.astype(np.float32)

    max_diff = float(np.abs(logits_pt - logits_ort_f32).max())
    mean_diff = float(np.abs(logits_pt - logits_ort_f32).mean())
    logger.info("ONNX validation — max_abs_diff=%.2e  mean_abs_diff=%.2e", max_diff, mean_diff)

    if max_diff > _MAX_ABS_DIFF:
        raise RuntimeError(
            f"ONNX numerical validation failed: max_abs_diff={max_diff:.2e} > {_MAX_ABS_DIFF:.2e}.\n"
            "Check the export pipeline for precision issues."
        )
    logger.info("ONNX validation: PASSED  (tolerance=%.2e)", _MAX_ABS_DIFF)


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 1: FP16 ONNX export + validation")
    p.add_argument("--checkpoint", default="weights/segmentation/orfd/frozen_backbone/segformer-b2/best.pth")
    p.add_argument("--variant", default="segformer-b2", choices=list(_HF_BASES),
                   help="SegFormer variant matching --checkpoint's architecture (default segformer-b2, "
                        "the only variant this pipeline supported before). Must match --checkpoint's own "
                        "architecture or PyTorch will fail to strict-load its state dict.")
    p.add_argument("--resolution", type=int, default=256,
                   help="Input resolution (square). Choose from Stage 0 sweep.")
    p.add_argument("--output-dir", default=None,
                   help="Directory to save the .onnx file (default: weights/segmentation/optimization "
                        "for segformer-b2, weights/segmentation/optimization_<variant> otherwise -- "
                        "kept separate per variant since benchmark_jetson.py globs *.onnx in one "
                        "directory per invocation and assumes a single backbone for the whole run).")
    p.add_argument("--no-fp16", dest="fp16", action="store_false", default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = _ROOT / ckpt
    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif args.variant == "segformer-b2":
        out_dir = Path("weights/segmentation/optimization")
    else:
        out_dir = Path(f"weights/segmentation/optimization_{args.variant}")
    if not out_dir.is_absolute():
        out_dir = _ROOT / out_dir

    onnx_path = export_fp16_onnx(str(ckpt), args.resolution, out_dir, args.device, args.fp16, args.variant)
    print(f"\nONNX ready: {onnx_path}")
    print("Transfer this file to the Jetson, then run:")
    print(f"  python scripts/segmentation/optimization/benchmark_jetson.py --onnx-dir {out_dir} --backbone {args.variant}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

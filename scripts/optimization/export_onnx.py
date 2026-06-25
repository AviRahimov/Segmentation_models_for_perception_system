#!/usr/bin/env python3
"""Stage 1: Export SegFormer-B2 baseline to FP16 ONNX with numerical validation.

Produces a validated ONNX file ready to be transferred to the Jetson for
TRT engine build.  Does NOT build TRT engines — that happens on Jetson via
benchmark_jetson.py.

Usage
-----
    # Baseline FP16 export (dev PC):
    python scripts/optimization/export_onnx.py \\
        --checkpoint weights/orfd/frozen_backbone/segformer-b2/best.pth \\
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

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("export_onnx")

# Tolerance for ONNX-vs-PyTorch numerical validation.
_MAX_ABS_DIFF = 1e-2  # fp16 rounding can introduce ~1e-3; generous but not silent


def _load_model(checkpoint: str, resolution: int, fp16: bool, device: str):
    """Load SegFormer-B2 from a local .pth, set processor to target resolution."""
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    ckpt_path = Path(checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    state_dict = ckpt.get("net", ckpt) if isinstance(ckpt, dict) else ckpt

    from perception.models.semantic.segformer import _remap_segformer_keys
    state_dict = _remap_segformer_keys(state_dict)

    n_labels = state_dict["decode_head.classifier.weight"].shape[0]

    hf_base = "nvidia/segformer-b2-finetuned-ade-512-512"
    processor = SegformerImageProcessor.from_pretrained(hf_base)
    processor.size = {"height": resolution, "width": resolution}

    model = SegformerForSemanticSegmentation.from_pretrained(
        hf_base, num_labels=n_labels, ignore_mismatched_sizes=True
    )
    model.load_state_dict(state_dict, strict=True)
    model = model.eval().to(device)
    if fp16:
        model = model.half()

    logger.info("Loaded: %s  res=%d  fp16=%s  classes=%d", ckpt_path.name, resolution, fp16, n_labels)
    return model, processor


def export_fp16_onnx(
    checkpoint: str,
    resolution: int,
    output_dir: Path,
    device: str = "cuda",
    fp16: bool = True,
) -> Path:
    """Export baseline checkpoint to FP16 ONNX.  Returns the saved .onnx path."""
    import onnx

    model, _ = _load_model(checkpoint, resolution, fp16, device)

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
    p.add_argument("--checkpoint", default="weights/orfd/frozen_backbone/segformer-b2/best.pth")
    p.add_argument("--resolution", type=int, default=256,
                   help="Input resolution (square). Choose from Stage 0 sweep.")
    p.add_argument("--output-dir", default="weights/optimization",
                   help="Directory to save the .onnx file")
    p.add_argument("--no-fp16", dest="fp16", action="store_false", default=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = _ROOT / ckpt
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = _ROOT / out_dir

    onnx_path = export_fp16_onnx(str(ckpt), args.resolution, out_dir, args.device, args.fp16)
    print(f"\nONNX ready: {onnx_path}")
    print("Transfer this file to the Jetson, then run:")
    print(f"  python scripts/optimization/benchmark_jetson.py --onnx-dir weights/optimization/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Stage 0: Resolution sweep for SegFormer-B2 ORFD checkpoint.

Evaluates mIoU and median latency at 256, 384, and 512 px on the ORFD
validation set.  Writes reports/optimization/resolution_sweep.json.

Usage
-----
    python scripts/optimization/resolution_sweep.py \\
        --checkpoint weights/orfd/frozen_backbone/segformer-b2/best.pth \\
        --data datasets/Final_Dataset

Choose the resolution for downstream stages based on the printed table.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts" / "training"))

import train_orfd as _t

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("resolution_sweep")

# Label size kept fixed so mIoU comparisons across resolutions are consistent.
_LABEL_SIZE = 512
_WARMUP_ITERS = 20
_TIMING_ITERS = 100


def _load_model(checkpoint: str, device: str):
    """Load SegFormer-B2 from a local .pth checkpoint."""
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
    model = SegformerForSemanticSegmentation.from_pretrained(
        hf_base, num_labels=n_labels, ignore_mismatched_sizes=True
    )
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    logger.info("Loaded checkpoint: %s  (%d classes)", ckpt_path, n_labels)
    return model, processor


@torch.no_grad()
def _sweep_resolution(
    model,
    processor,
    data_root: str,
    resolution: int,
    device: str,
    batch_size: int,
    num_workers: int,
) -> dict:
    from perception.datasets.orfd_torch import ORFDDataset

    # Override processor to the sweep resolution.
    processor.size = {"height": resolution, "width": resolution}

    val_ds = ORFDDataset(data_root, split="validation", augment=False, input_size=_LABEL_SIZE)
    loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device == "cuda"),
    )

    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    batch_times: list[float] = []

    # Warm-up pass (first batch only to avoid cold-start bias).
    images_warm, _ = next(iter(loader))
    images_warm = images_warm.to(device)
    for _ in range(3):
        _t.segformer_forward(model, processor, images_warm, device, fp16=False)
    if device == "cuda":
        torch.cuda.synchronize()

    for images, labels in tqdm(loader, desc=f"res={resolution}", leave=False):
        images = images.to(device)
        t0 = time.perf_counter()
        logits = _t.segformer_forward(model, processor, images, device, fp16=False)
        if device == "cuda":
            torch.cuda.synchronize()
        batch_times.append((time.perf_counter() - t0) * 1000)

        preds = logits.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels)

    preds_cat  = torch.cat(all_preds,  dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    miou, per_class = _t.compute_miou(preds_cat, labels_cat)

    # Per-image latency estimate (batch_time / batch_size for a rough p50).
    per_image_ms = [t / batch_size for t in batch_times]
    p50 = float(np.percentile(per_image_ms, 50))
    p99 = float(np.percentile(per_image_ms, 99))
    fps_approx = 1000.0 / p50 if p50 > 0 else 0.0

    logger.info(
        "res=%d  mIoU=%.4f  per-class=%s  p50=%.1fms  p99=%.1fms  fps≈%.1f",
        resolution, miou,
        [f"{v:.3f}" if not (isinstance(v, float) and v != v) else "nan" for v in per_class],
        p50, p99, fps_approx,
    )

    return {
        "resolution": resolution,
        "miou": round(miou, 4),
        "miou_per_class": [
            round(v, 4) if not (isinstance(v, float) and v != v) else None
            for v in per_class
        ],
        "latency_ms_p50": round(p50, 2),
        "latency_ms_p99": round(p99, 2),
        "fps_approx": round(fps_approx, 1),
        "label_size": _LABEL_SIZE,
        "device": device,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Stage 0: resolution mIoU + latency sweep")
    p.add_argument("--checkpoint", default="weights/orfd/frozen_backbone/segformer-b2/best.pth")
    p.add_argument("--data",        default="datasets/Final_Dataset",
                   help="ORFD root (must contain validation/)")
    p.add_argument("--resolutions", nargs="+", type=int, default=[256, 384, 512])
    p.add_argument("--batch",       type=int, default=8)
    p.add_argument("--workers",     type=int, default=4)
    p.add_argument("--output",      default="reports/optimization/resolution_sweep.json")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # Resolve relative paths from repo root.
    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = _ROOT / ckpt
    data = Path(args.data)
    if not data.is_absolute():
        data = _ROOT / data
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = _ROOT / out_path

    model, processor = _load_model(str(ckpt), args.device)

    results = []
    for res in sorted(set(args.resolutions)):
        logger.info("--- Sweeping resolution: %d px ---", res)
        row = _sweep_resolution(
            model, processor, str(data), res,
            args.device, args.batch, args.workers,
        )
        results.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Saved: %s", out_path)

    # Print summary table.
    print("\n" + "=" * 70)
    print(f"{'Res':>6}  {'mIoU':>7}  {'p50 (ms)':>10}  {'p99 (ms)':>10}  {'FPS≈':>6}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['resolution']:>6}  {r['miou']:>7.4f}  "
            f"{r['latency_ms_p50']:>10.1f}  {r['latency_ms_p99']:>10.1f}  "
            f"{r['fps_approx']:>6.1f}"
        )
    print("=" * 70)
    print(f"\nNote: latency measured on {results[0]['device'] if results else 'N/A'}.")
    print("Jetson AGX Orin numbers will differ — use benchmark_jetson.py for authoritative FPS.")
    print(f"\nResults saved to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

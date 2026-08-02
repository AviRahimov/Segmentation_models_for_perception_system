"""Stage 4 prep: real params + dev-PC latency for the 6 new architecture
candidates trained in the SegFormer-B2 model-comparison phase (AurigaNet,
DINOv2-Base/Large, UPerNet, Mask2Former-Base/Large). Complements the mIoU
numbers already recorded in each weights/segmentation/orfd/<arch>/train_log.json.

Usage
-----
    python scripts/segmentation/evaluation/benchmark_new_architectures.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts" / "segmentation"))

logging.basicConfig(level=logging.WARNING)

WEIGHTS_DIR = _ROOT / "weights" / "segmentation" / "orfd"
N_WARMUP = 10
N_ITERS = 50


def _params_m(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def _latency_ms(fn, *args) -> float:
    for _ in range(N_WARMUP):
        fn(*args)
    torch.cuda.synchronize()
    times = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def bench_auriganet():
    from _auriganet_common import build_auriganet
    model, _ = build_auriganet("cuda", fp16=False, weights=str(WEIGHTS_DIR / "auriganet/best.pth"))
    model.eval()
    x = torch.randn(1, 3, 640, 640, device="cuda")
    with torch.no_grad():
        lat = _latency_ms(lambda: model(x))
    return _params_m(model), lat


def bench_dinov2(variant_dir: str, backbone_id: str):
    from _dinov2_common import DINOV2_INPUT_SIZE, build_dinov2
    model, _ = build_dinov2("cuda", fp16=False, weights=str(WEIGHTS_DIR / variant_dir / "best.pth"),
                             backbone_id=backbone_id)
    model.eval()
    x = torch.randn(1, 3, DINOV2_INPUT_SIZE, DINOV2_INPUT_SIZE, device="cuda")
    with torch.no_grad():
        lat = _latency_ms(lambda: model(x))
    return _params_m(model), lat


def bench_upernet():
    from _upernet_common import build_upernet
    model, _ = build_upernet("cuda", fp16=False, weights=str(WEIGHTS_DIR / "upernet/best.pth"))
    model.eval()
    x = torch.randn(1, 3, 512, 512, device="cuda")
    with torch.no_grad():
        lat = _latency_ms(lambda: model(pixel_values=x))
    return _params_m(model), lat


def bench_mask2former(variant_dir: str, backbone_id: str):
    from _mask2former_common import build_mask2former
    model, _ = build_mask2former("cuda", fp16=False, weights=str(WEIGHTS_DIR / variant_dir / "best.pth"),
                                  backbone_id=backbone_id)
    model.eval()
    x = torch.randn(1, 3, 384, 384, device="cuda")
    with torch.no_grad():
        lat = _latency_ms(lambda: model(pixel_values=x))
    return _params_m(model), lat


def main() -> int:
    results = {}

    print("Benchmarking AurigaNet ...")
    results["auriganet"] = bench_auriganet()

    print("Benchmarking DINOv2-Base ...")
    results["dinov2-base"] = bench_dinov2("dinov2", "facebook/dinov2-base")

    print("Benchmarking DINOv2-Large ...")
    results["dinov2-large"] = bench_dinov2("dinov2-large", "facebook/dinov2-large")

    print("Benchmarking UPerNet (ConvNeXt-Base) ...")
    results["upernet"] = bench_upernet()

    print("Benchmarking Mask2Former-Base (Swin-Base) ...")
    results["mask2former-base"] = bench_mask2former("mask2former", "facebook/mask2former-swin-base-ade-semantic")

    print("Benchmarking Mask2Former-Large (Swin-Large) ...")
    results["mask2former-large"] = bench_mask2former("mask2former-large", "facebook/mask2former-swin-large-ade-semantic")

    out = {k: {"params_m": round(p, 2), "latency_ms": round(l, 2), "fps": round(1000.0 / l, 1)}
           for k, (p, l) in results.items()}
    print(json.dumps(out, indent=2))

    out_path = _ROOT / "reports" / "segmentation" / "new_architectures_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

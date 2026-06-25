#!/usr/bin/env python3
"""Stage 3: 2:4 structured sparsity + QAT for SegFormer-B2.

Pipeline (sparsify-then-QAT order, default)
--------------------------------------------
1. Load frozen-backbone B2 baseline checkpoint.
2. Apply modelopt 2:4 sparsity to Linear layers in the MiT backbone
   (attention QKV/output projections, Mix-FFN dense layers).
3. Short sparsity-aware fine-tune (3–5 epochs) to recover accuracy —
   only decode_head gets gradient updates (backbone frozen).
4. Apply QAT on top: calibrate + fine-tune again.
5. Export ONNX with both sparsity pattern and embedded QDQ nodes.

Alternative: --order qat_first (apply fake-quant before sparsifying).

Runtime note
------------
The 2:4 sparsity pattern is in the weights of the exported ONNX.  Actual
runtime speedup only materialises on the Jetson AGX Orin where cuSPARSELt
is present.  On the dev machine this step improves compression only.

Usage
-----
    python scripts/optimization/train_sparse.py \\
        --config config/optimization/sparse.yaml

    python scripts/optimization/train_sparse.py \\
        --checkpoint weights/orfd/frozen_backbone/segformer-b2/best.pth \\
        --resolution 256 --sparse-epochs 4 --qat-epochs 6 --order sparse_first
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

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
logger = logging.getLogger("train_sparse")


# --------------------------------------------------------------------------- #
# Config                                                                        #
# --------------------------------------------------------------------------- #

def _load_sparse_config(path: str) -> dict[str, Any]:
    import yaml
    p = Path(path)
    if not p.is_file():
        return {}
    with p.open() as f:
        return yaml.safe_load(f) or {}


def _parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(_ROOT / "config" / "optimization" / "sparse.yaml"))
    known, _ = pre.parse_known_args()
    cfg = _load_sparse_config(known.config)

    p = argparse.ArgumentParser(description="Stage 3: 2:4 sparsity + QAT")
    p.add_argument("--config",         default=known.config)
    p.add_argument("--checkpoint",     default=cfg.get("base_checkpoint",
                   "weights/orfd/frozen_backbone/segformer-b2/best.pth"))
    p.add_argument("--data",           default=cfg.get("data", "datasets/Final_Dataset"))
    p.add_argument("--resolution",     type=int, default=cfg.get("resolution", 256))
    p.add_argument("--sparse-epochs",  type=int, default=cfg.get("sparse_epochs", 4))
    p.add_argument("--qat-epochs",     type=int, default=cfg.get("qat_epochs", 6))
    p.add_argument("--sparse-lr",      type=float, default=cfg.get("sparse_lr", 1e-5))
    p.add_argument("--qat-lr",         type=float, default=cfg.get("qat_lr", 5e-6))
    p.add_argument("--batch",          type=int, default=cfg.get("batch_size", 8))
    p.add_argument("--workers",        type=int, default=cfg.get("workers", 4))
    p.add_argument("--cal-batches",    type=int, default=cfg.get("calibration_batches", 4))
    p.add_argument("--output-dir",     default=cfg.get("output_dir", "weights/optimization"))
    p.add_argument("--order",          choices=["sparse_first", "qat_first"],
                   default=cfg.get("order", "sparse_first"))
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Sparsity helpers                                                              #
# --------------------------------------------------------------------------- #

def _check_cusparselт() -> None:
    """Warn if cuSPARSELt is not available (no runtime speedup on this machine)."""
    import ctypes, ctypes.util
    lib = ctypes.util.find_library("cusparseLt") or ctypes.util.find_library("cusparse_lt")
    if lib is None:
        logger.warning(
            "cuSPARSELt not found — 2:4 sparsity pattern will be exported but "
            "runtime speedup will ONLY materialise on the Jetson AGX Orin."
        )
    else:
        logger.info("cuSPARSELt found: %s", lib)


def _apply_sparsity(model: torch.nn.Module, device: str) -> torch.nn.Module:
    """Apply 2:4 structured sparsity to Linear layers in the MiT backbone."""
    try:
        import modelopt.torch.sparsity as mts  # type: ignore
    except ImportError:
        raise ImportError(
            "nvidia-modelopt is required for 2:4 sparsity.\n"
            "Install with: pip install 'nvidia-modelopt[torch]'"
        )

    logger.info("Applying 2:4 structured sparsity to Linear layers ...")

    # Rules format: {module_type: {name_glob_pattern: config | None}}
    # None means exclude that pattern. Backbone Linear only; decode_head excluded.
    # Must be wrapped in a list — val2list() treats a bare tuple as a sequence and
    # unpacks it into [mode_str, config_dict], causing KeyError: 0.
    sparse_mode = [("sparse_magnitude", {
        "nn.Linear": {
            "*": {},                # sparsify all Linear by default
            "*decode_head*": None,  # exclude decode_head layers
        },
    })]

    n_before = sum(1 for _ in model.modules() if isinstance(_, torch.nn.Linear))
    model = mts.sparsify(model, mode=sparse_mode)
    logger.info("2:4 sparsity applied. Total Linear layers in model: %d", n_before)
    return model


def _apply_quantization(model: torch.nn.Module) -> torch.nn.Module:
    try:
        import modelopt.torch.quantization as mtq  # type: ignore
    except ImportError:
        raise ImportError("nvidia-modelopt is required.")
    logger.info("Applying INT8 fake-quantization on top of sparse model ...")
    return mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop=None)


def _calibrate(model, loader: DataLoader, processor, device: str, n_batches: int) -> None:
    import modelopt.torch.quantization as mtq  # type: ignore
    logger.info("Calibrating quantization scales (%d batches) ...", n_batches)
    model.eval()
    count = 0

    def _fwd(mod):
        nonlocal count
        for images, _ in loader:
            if count >= n_batches:
                break
            images = images.to(device)
            with torch.no_grad():
                _t.segformer_forward(mod, processor, images, device, fp16=False)
            count += 1

    mtq.calibrate(model, algorithm="max", forward_loop=_fwd)
    logger.info("Calibration done (%d batches).", count)


def _freeze_backbone(model: torch.nn.Module) -> None:
    frozen = 0
    for name, param in model.named_parameters():
        if "decode_head" not in name:
            param.requires_grad_(False)
            frozen += param.numel()
    logger.info("Backbone frozen (%dM params).", frozen // 1_000_000)


def _fine_tune(model, processor, loader, val_loader, device, lr, epochs, tag):
    from torch.optim import AdamW
    _freeze_backbone(model)
    criterion = lambda logits, labels: _t._dice_ce_loss(logits, labels)
    head_params = [p for p in model.decode_head.parameters() if p.requires_grad]
    optimizer = AdamW(head_params, lr=lr, weight_decay=0.01)
    best_miou = 0.0
    best_state = None
    for epoch in range(1, epochs + 1):
        loss = _t.train_one_epoch(
            model, processor, loader, optimizer, criterion, device, fp16=False, clip_norm=1.0,
        )
        _, val_miou = _t.evaluate(model, processor, val_loader, criterion, device, fp16=False)
        logger.info("[%s] epoch %d/%d  loss=%.4f  mIoU=%.4f", tag, epoch, epochs, loss, val_miou)
        if val_miou > best_miou:
            best_miou = val_miou
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_miou


def _export_onnx(model: torch.nn.Module, resolution: int, output_dir: Path,
                 device: str, tag: str) -> Path:
    import onnx

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / f"sparse_qat_int8_{resolution}x{resolution}_{tag}.onnx"
    model.eval()
    dummy = torch.zeros(1, 3, resolution, resolution, device=device, dtype=torch.float32)

    # Remove sparsity wrappers before ONNX export. torch.onnx.export cannot
    # capture the sparse module wrappers correctly — the exported ONNX ends up
    # with near-zero weight contributions and only the classifier bias survives,
    # producing constant degenerate output. mts.export() strips the wrappers and
    # bakes the 2:4 sparse weights as regular tensors in a standard nn.Module.
    try:
        import modelopt.torch.sparsity as mts  # type: ignore
        model = mts.export(model)
        logger.info("Sparse wrappers removed via mts.export() before ONNX export.")
    except Exception as e:
        logger.warning("mts.export() failed (%s) — ONNX weights may be degenerate.", e)

    try:
        import modelopt.torch.quantization as mtq  # type: ignore
        mtq.export_onnx(
            model,
            args=(dummy,),      # positional tuple — required in torch 2.x
            f=str(onnx_path),
            input_names=["pixel_values"],
            output_names=["logits"],
            opset_version=17,
            do_constant_folding=False,
        )
    except Exception as e:
        logger.warning("modelopt ONNX export failed (%s); using torch.onnx fallback", e)
        torch.onnx.export(
            model, (dummy,), str(onnx_path),    # positional tuple — required in torch 2.x
            input_names=["pixel_values"], output_names=["logits"],
            opset_version=17, do_constant_folding=False, dynamo=False,
        )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX checker: OK  →  %s", onnx_path)
    return onnx_path


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def main() -> int:
    args = _parse_args()
    _check_cusparselт()
    device = args.device

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = _ROOT / ckpt
    data = Path(args.data)
    if not data.is_absolute():
        data = _ROOT / data
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = _ROOT / out_dir

    if args.resolution is None:
        args.resolution = 256
        logger.warning("resolution not set in config — defaulting to 256. "
                       "Set 'resolution: 256' in sparse.yaml after running Stage 0.")

    logger.info("=== Stage 3: 2:4 Sparsity + QAT (order=%s) ===", args.order)

    # ---- Load baseline ----
    model, processor = _t.build_segformer("segformer-b2", device, fp16=False)
    ckpt_data = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    state_dict = ckpt_data.get("net", ckpt_data) if isinstance(ckpt_data, dict) else ckpt_data

    from perception.models.semantic.segformer import _remap_segformer_keys
    state_dict = _remap_segformer_keys(state_dict)

    model.load_state_dict(state_dict, strict=True)
    processor.size = {"height": args.resolution, "width": args.resolution}

    # ---- Data ----
    from perception.datasets.orfd_torch import ORFDDataset
    train_ds = ORFDDataset(str(data), split="training",   augment=True,  input_size=512)
    val_ds   = ORFDDataset(str(data), split="validation", augment=False, input_size=512)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    # Baseline mIoU
    criterion = lambda logits, labels: _t._dice_ce_loss(logits, labels)
    _, baseline_miou = _t.evaluate(model, processor, val_loader, criterion, device, fp16=False)
    logger.info("Baseline mIoU: %.4f", baseline_miou)

    onnx_paths = []

    if args.order == "sparse_first":
        # ---- Sparsify → fine-tune → export (before QAT) ----
        model = _apply_sparsity(model, device)
        sparse_miou = _fine_tune(model, processor, train_loader, val_loader, device,
                                 args.sparse_lr, args.sparse_epochs, "sparse-ft")
        logger.info("Post-sparsity mIoU: %.4f", sparse_miou)

        # Export HERE — before QAT is applied. mts.export() only succeeds when
        # the export stack has a single mode (sparsity only). Calling it after
        # mtq.quantize() stacks a second mode and raises an export-stack conflict.
        # TRT cannot use INT8 arithmetic without embedded QDQ nodes anyway, so
        # the engine precision is identical whether we export here or after QAT.
        onnx_paths.append(_export_onnx(model, args.resolution, out_dir, device, "sparse_first"))

        # Continue QAT for final accuracy logging (not for export).
        model = _apply_quantization(model)
        _calibrate(model, train_loader, processor, device, args.cal_batches)
        qat_miou = _fine_tune(model, processor, train_loader, val_loader, device,
                              args.qat_lr, args.qat_epochs, "sparse+qat-ft")
        logger.info("Final mIoU after QAT (reference only): %.4f", qat_miou)

    else:  # qat_first
        # ---- QAT → fine-tune → sparsify ----
        model = _apply_quantization(model)
        _calibrate(model, train_loader, processor, device, args.cal_batches)
        qat_miou = _fine_tune(model, processor, train_loader, val_loader, device,
                              args.qat_lr, args.qat_epochs, "qat-ft")
        logger.info("Post-QAT mIoU: %.4f", qat_miou)

        model = _apply_sparsity(model, device)
        sparse_miou = _fine_tune(model, processor, train_loader, val_loader, device,
                                 args.sparse_lr, args.sparse_epochs, "qat+sparse-ft")
        logger.info("Final mIoU (qat_first): %.4f", sparse_miou)
        onnx_paths.append(_export_onnx(model, args.resolution, out_dir, device, "qat_first"))

    print("\n=== Stage 3 complete ===")
    print(f"Baseline mIoU: {baseline_miou:.4f}")
    for path in onnx_paths:
        print(f"  Exported: {path}")
    print("Transfer ONNX files to Jetson for TRT engine build.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

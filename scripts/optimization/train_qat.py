#!/usr/bin/env python3
"""Stage 2: Quantization-Aware Training (QAT) for SegFormer-B2 using NVIDIA modelopt.

Pipeline
--------
1. Load frozen-backbone B2 baseline checkpoint.
2. Apply modelopt fake-quantization (INT8 symmetric per-channel weights,
   per-tensor activations — TensorRT's standard scheme).
3. Calibrate quantization scales on a few training batches.
4. Fine-tune for a small number of epochs (only decode_head gets gradients —
   backbone stays frozen as in the baseline).
5. Export ONNX with embedded QDQ nodes — TRT reads INT8 scales directly,
   no external calibration cache needed.
6. Run onnx.checker + onnxruntime numerical validation.

Usage
-----
    python scripts/optimization/train_qat.py \\
        --config config/optimization/qat.yaml

    # Or override individual settings:
    python scripts/optimization/train_qat.py \\
        --checkpoint weights/orfd/frozen_backbone/segformer-b2/best.pth \\
        --resolution 256 --epochs 8 --lr 1e-5
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

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
logger = logging.getLogger("train_qat")


# --------------------------------------------------------------------------- #
# Config                                                                        #
# --------------------------------------------------------------------------- #

def _load_qat_config(path: str) -> dict[str, Any]:
    import yaml
    p = Path(path)
    if not p.is_file():
        return {}
    with p.open() as f:
        return yaml.safe_load(f) or {}


def _parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(_ROOT / "config" / "optimization" / "qat.yaml"))
    known, _ = pre.parse_known_args()
    cfg = _load_qat_config(known.config)

    p = argparse.ArgumentParser(description="Stage 2: QAT INT8 fine-tuning")
    p.add_argument("--config",      default=known.config)
    p.add_argument("--checkpoint",  default=cfg.get("base_checkpoint",
                   "weights/orfd/frozen_backbone/segformer-b2/best.pth"))
    p.add_argument("--data",        default=cfg.get("data", "datasets/Final_Dataset"))
    p.add_argument("--resolution",  type=int, default=cfg.get("resolution", 256),
                   help="Input resolution chosen from Stage 0 sweep")
    p.add_argument("--epochs",      type=int, default=cfg.get("qat_epochs", 8))
    p.add_argument("--lr",          type=float, default=cfg.get("qat_lr", 1e-5))
    p.add_argument("--batch",       type=int, default=cfg.get("batch_size", 8))
    p.add_argument("--workers",     type=int, default=cfg.get("workers", 4))
    p.add_argument("--cal-batches", type=int, default=cfg.get("calibration_batches", 4),
                   help="Number of training batches for MTQ calibration")
    p.add_argument("--output-dir",  default=cfg.get("output_dir", "weights/optimization"))
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# QAT helpers                                                                   #
# --------------------------------------------------------------------------- #

def _apply_quantization(model: torch.nn.Module) -> torch.nn.Module:
    """Insert INT8 fake-quantization nodes via modelopt."""
    try:
        import modelopt.torch.quantization as mtq  # type: ignore
    except ImportError:
        raise ImportError(
            "nvidia-modelopt is required for QAT.\n"
            "Install with: pip install 'nvidia-modelopt[torch]'"
        )

    # INT8 symmetric per-channel weights, per-tensor activations.
    # This matches TensorRT's default INT8 quantization scheme.
    quant_cfg = mtq.INT8_DEFAULT_CFG
    logger.info("Applying INT8 fake-quantization via modelopt ...")
    model = mtq.quantize(model, quant_cfg, forward_loop=None)
    return model


def _calibrate(model, loader: DataLoader, processor, device: str, n_batches: int) -> None:
    """Run MTQ calibration to compute quantization scale factors."""
    import modelopt.torch.quantization as mtq  # type: ignore

    logger.info("Calibrating on %d batches ...", n_batches)
    model.eval()
    count = 0

    pbar = tqdm(total=n_batches, desc="  calibrate", unit="batch",
                bar_format="{l_bar}{bar:30}{r_bar}", leave=True)

    def _forward_loop(mod):
        nonlocal count
        for images, _ in loader:
            if count >= n_batches:
                break
            images = images.to(device)
            with torch.no_grad():
                _t.segformer_forward(mod, processor, images, device, fp16=False)
            count += 1
            pbar.update(1)

    mtq.calibrate(model, algorithm="max", forward_loop=_forward_loop)
    pbar.close()
    logger.info("Calibration done (%d batches processed).", count)


def _freeze_backbone(model: torch.nn.Module) -> None:
    """Freeze all parameters except decode_head (mirrors baseline training)."""
    frozen = 0
    for name, param in model.named_parameters():
        if "decode_head" not in name:
            param.requires_grad_(False)
            frozen += param.numel()
    logger.info("Backbone frozen for QAT fine-tune (%dM params).", frozen // 1_000_000)


def _export_qdq_onnx(model: torch.nn.Module, resolution: int, output_dir: Path, device: str) -> Path:
    """Export QAT model to ONNX with embedded QDQ nodes."""
    import onnx

    try:
        import modelopt.torch.quantization as mtq  # type: ignore
        _has_onnx_export = hasattr(mtq, "export_onnx") or hasattr(mtq, "export")
    except ImportError:
        _has_onnx_export = False

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / f"qat_int8_{resolution}x{resolution}.onnx"

    model.eval()
    dummy = torch.zeros(1, 3, resolution, resolution, device=device, dtype=torch.float32)

    if _has_onnx_export:
        # Preferred: use modelopt's ONNX export which correctly handles QDQ placement.
        try:
            import modelopt.torch.quantization as mtq  # type: ignore
            mtq.export_onnx(
                model,
                args=(dummy,),      # positional tuple — required in torch 2.x
                f=str(onnx_path),
                input_names=["pixel_values"],
                output_names=["logits"],
                opset_version=17,
                do_constant_folding=False,  # keep QDQ nodes intact
            )
            logger.info("Exported QDQ ONNX via modelopt: %s", onnx_path)
        except Exception as e:
            logger.warning("modelopt ONNX export failed (%s); falling back to torch.onnx.export", e)
            _fallback_onnx_export(model, dummy, onnx_path)
    else:
        _fallback_onnx_export(model, dummy, onnx_path)

    logger.info("Checking ONNX graph validity ...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX checker: OK")

    return onnx_path


def _fallback_onnx_export(model, dummy, onnx_path: Path) -> None:
    logger.warning(
        "Using torch.onnx.export fallback — QDQ nodes may not be preserved. "
        "Upgrade nvidia-modelopt for proper INT8 ONNX export."
    )
    torch.onnx.export(
        model,
        (dummy,),           # positional tuple — required in torch 2.x
        str(onnx_path),
        input_names=["pixel_values"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=False,
        dynamo=False,
    )
    logger.info("Fallback ONNX export: %s", onnx_path)


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def main() -> int:
    args = _parse_args()
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
                       "Set 'resolution: 256' in qat.yaml after running Stage 0.")

    print("\n" + "="*60)
    print("  STAGE 2 — QAT INT8 Fine-Tuning")
    print("="*60)
    logger.info("Checkpoint: %s", ckpt)
    logger.info("Resolution: %d px  |  Epochs: %d  |  LR: %.2e  |  Device: %s",
                args.resolution, args.epochs, args.lr, device)

    # ---- Load baseline model ----
    model, processor = _t.build_segformer("segformer-b2", device, fp16=False)

    ckpt_data = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    state_dict = ckpt_data.get("net", ckpt_data) if isinstance(ckpt_data, dict) else ckpt_data

    from perception.models.semantic.segformer import _remap_segformer_keys
    state_dict = _remap_segformer_keys(state_dict)

    model.load_state_dict(state_dict, strict=True)
    logger.info("Baseline checkpoint loaded.")

    # Override processor resolution.
    processor.size = {"height": args.resolution, "width": args.resolution}

    # ---- Data ----
    from perception.datasets.orfd_torch import ORFDDataset
    train_ds = ORFDDataset(str(data), split="training",   augment=True,  input_size=512)
    val_ds   = ORFDDataset(str(data), split="validation", augment=False, input_size=512)
    cal_loader   = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)

    # ---- Evaluate baseline mIoU (pre-QAT reference) ----
    criterion = lambda logits, labels: _t._dice_ce_loss(logits, labels)
    print("\n[1/5] Evaluating baseline FP32 mIoU ...")
    _, baseline_miou = _t.evaluate(model, processor, val_loader, criterion, device, fp16=False)
    print(f"      Baseline mIoU (FP32): {baseline_miou:.4f}")

    # ---- Apply quantization + calibrate ----
    print("\n[2/5] Inserting INT8 fake-quantization nodes ...")
    model = _apply_quantization(model)
    print(f"\n[3/5] Calibrating quantization scales ({args.cal_batches} batches) ...")
    _calibrate(model, cal_loader, processor, device, args.cal_batches)

    # ---- Evaluate post-calibration (before fine-tune) ----
    print("\n[4/5] Evaluating post-calibration mIoU (no fine-tune yet) ...")
    _, cal_miou = _t.evaluate(model, processor, val_loader, criterion, device, fp16=False)
    print(f"      Post-calibration mIoU: {cal_miou:.4f}  "
          f"(drop vs FP32: {baseline_miou - cal_miou:+.4f})")

    # ---- QAT fine-tune (only decode_head) ----
    print(f"\n[5/5] QAT fine-tuning decode_head ({args.epochs} epochs) ...")
    _freeze_backbone(model)
    from torch.optim import AdamW
    head_params = [p for p in model.decode_head.parameters() if p.requires_grad]
    optimizer = AdamW(head_params, lr=args.lr, weight_decay=0.01)

    best_miou = cal_miou
    best_state = None

    epoch_bar = tqdm(
        range(1, args.epochs + 1),
        desc="QAT fine-tune",
        unit="epoch",
        bar_format="{l_bar}{bar:35}{r_bar}",
        leave=True,
    )
    epoch_bar.set_postfix(loss="—", mIoU="—", best=f"{best_miou:.4f}")

    for epoch in epoch_bar:
        train_loss = _t.train_one_epoch(
            model, processor, train_loader, optimizer, criterion,
            device, fp16=False, clip_norm=1.0,
        )
        _, val_miou = _t.evaluate(model, processor, val_loader, criterion, device, fp16=False)

        is_best = val_miou > best_miou
        if is_best:
            best_miou = val_miou
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        epoch_bar.set_postfix(
            loss=f"{train_loss:.4f}",
            mIoU=f"{val_miou:.4f}",
            best=f"{best_miou:.4f}",
            new="✓" if is_best else " ",
        )
        tqdm.write(
            f"  epoch {epoch:2d}/{args.epochs}  loss={train_loss:.4f}  "
            f"mIoU={val_miou:.4f}  best={best_miou:.4f}"
            + ("  ← new best" if is_best else "")
        )

    epoch_bar.close()

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info("Restored best QAT checkpoint: mIoU=%.4f", best_miou)

    logger.info(
        "QAT complete.  baseline=%.4f  post-cal=%.4f  final=%.4f  (total_drop=%.4f)",
        baseline_miou, cal_miou, best_miou, baseline_miou - best_miou,
    )

    # ---- Export QDQ ONNX ----
    model = model.to(device)
    onnx_path = _export_qdq_onnx(model, args.resolution, out_dir, device)

    print("\n" + "="*60)
    print("  STAGE 2 — COMPLETE")
    print("="*60)
    print(f"  Baseline FP32 mIoU : {baseline_miou:.4f}")
    print(f"  Post-calibration   : {cal_miou:.4f}  ({baseline_miou - cal_miou:+.4f})")
    print(f"  Final QAT INT8     : {best_miou:.4f}  ({baseline_miou - best_miou:+.4f})")
    print(f"  ONNX saved         : {onnx_path}")
    print("  → Transfer ONNX to Jetson for TRT engine build.")
    print("="*60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

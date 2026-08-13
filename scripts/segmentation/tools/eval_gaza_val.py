#!/usr/bin/env python3
"""Evaluate one or more SegFormer-B2 checkpoints on the Gaza-domain held-out
val split (and, for a regression-guard comparison, on ORFD-val too).

Exists because the two pre-Gaza-work production checkpoints (frozen_backbone,
distilled) were never scored against the Gaza-val split introduced in
split_gaza_domain.py -- their Gaza-domain mIoU has to be *measured*, not
assumed, before they can be fairly ranked against the Gaza-only fine-tunes
that were selected using that same split.

Usage
-----
    python scripts/segmentation/tools/eval_gaza_val.py \\
        --weights weights/segmentation/orfd/frozen_backbone/segformer-b2/best.pth \\
                  weights/segmentation/orfd/distilled_segformer-b2/best.pth \\
        --labels frozen_backbone distilled
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts" / "segmentation"))
sys.path.insert(0, str(_ROOT / "scripts" / "segmentation" / "training"))

from perception.datasets.orfd_torch import ORFDDataset  # noqa: E402
from perception.datasets.gaza_domain_torch import GazaDomainDataset  # noqa: E402
from _orfd_common import _dice_ce_loss, build_segformer, evaluate  # noqa: E402
from _segformer_checkpoint_common import load_remapped_state_dict  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("eval_gaza_val")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", nargs="+", required=True, help="One or more SegFormer .pth checkpoints")
    p.add_argument("--labels", nargs="+", default=None, help="Optional short label per checkpoint (must match --weights 1:1)")
    p.add_argument("--variants", nargs="+", default=None,
                   help="Optional model variant per checkpoint (e.g. segformer-b2 segformer-b3), must match "
                        "--weights 1:1. Defaults to segformer-b2 for every checkpoint (unchanged behavior).")
    p.add_argument("--gaza-data", default="datasets/segmentation/gaza_domain")
    p.add_argument("--orfd-data", default="datasets/segmentation/ORFD")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--json-out", default=None, help="Optional path to also dump results as JSON")
    args = p.parse_args()

    if args.labels and len(args.labels) != len(args.weights):
        raise SystemExit(f"--labels needs exactly {len(args.weights)} entries (one per --weights)")
    labels = args.labels if args.labels else args.weights
    if args.variants and len(args.variants) != len(args.weights):
        raise SystemExit(f"--variants needs exactly {len(args.weights)} entries (one per --weights)")
    variants = args.variants if args.variants else ["segformer-b2"] * len(args.weights)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = device == "cuda"

    gaza_path = (Path(args.gaza_data) if Path(args.gaza_data).is_absolute() else _ROOT / args.gaza_data)
    val_stems = set((gaza_path / "splits" / "val.txt").read_text().split())
    gaza_val_ds = GazaDomainDataset(str(gaza_path), augment=False, stems=val_stems)
    gaza_val_loader = DataLoader(gaza_val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    orfd_path = (Path(args.orfd_data) if Path(args.orfd_data).is_absolute() else _ROOT / args.orfd_data)
    orfd_val_ds = ORFDDataset(str(orfd_path), split="validation", augment=False)
    orfd_val_loader = DataLoader(orfd_val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    logger.info("Gaza-val: %d samples  ORFD-val: %d samples", len(gaza_val_ds), len(orfd_val_ds))

    def criterion(logits, lbl):
        return _dice_ce_loss(logits, lbl)

    results: dict[str, dict[str, float]] = {}
    for weights_path, label, variant in zip(args.weights, labels, variants):
        wpath = Path(weights_path)
        if not wpath.is_absolute():
            wpath = _ROOT / wpath
        model, processor = build_segformer(variant, device, fp16)
        state_dict = load_remapped_state_dict(wpath)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        gaza_loss, gaza_miou = evaluate(model, processor, gaza_val_loader, criterion, device, fp16)
        orfd_loss, orfd_miou = evaluate(model, processor, orfd_val_loader, criterion, device, fp16)
        logger.info("%-22s  gaza_val_mIoU=%.4f  orfd_val_mIoU=%.4f  (%s)", label, gaza_miou, orfd_miou, wpath)
        results[label] = {"gaza_val_miou": gaza_miou, "orfd_val_miou": orfd_miou, "weights": str(wpath)}
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print(f"{'Label':<24} {'Gaza-val mIoU':>15} {'ORFD-val mIoU':>15}")
    print("-" * 70)
    for label, r in results.items():
        print(f"{label:<24} {r['gaza_val_miou']:>15.4f} {r['orfd_val_miou']:>15.4f}")
    print("=" * 70)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))
        logger.info("Wrote %s", args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

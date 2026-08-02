"""Shared AurigaNet build/forward helpers — mirrors ``_segformer_checkpoint_common.py``'s
role for SegFormer, so training/eval/comparison scripts share one implementation.

AurigaNet is a lightweight (~6.4M param) multi-task backbone+neck+seg-head
architecture (vendored under ``src/perception/models/semantic/_vendored/auriganet/``),
resurrected from git history (commit range 9edf9a7..e8a6847) where it was fine-tuned
on ORFD once and scored 0.8831 mIoU before being deliberately removed
("chore: remove AurigaNet model and all references"). Unlike SegFormer it has no
pretrained backbone — every fine-tuning run starts from random initialisation.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger("train_orfd")

NUM_CLASSES = 3


def build_auriganet(device: str, fp16: bool, weights: str = "") -> tuple[nn.Module, None]:
    """Return (model, None) — the None keeps the (model, processor) tuple shape
    used elsewhere in this codebase, since AurigaNet has no HF processor."""
    import sys
    _ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_ROOT / "src"))
    from perception.models.semantic._vendored.auriganet import AurigaNetArch

    logger.info("Building AurigaNet (num_seg_classes=%d, with_detection=False) ...", NUM_CLASSES)
    model = AurigaNetArch(num_seg_classes=NUM_CLASSES, with_detection=False)

    if weights and Path(weights).is_file():
        ckpt = torch.load(weights, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("net", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("AurigaNet resume: %d missing keys", len(missing))
        if unexpected:
            logger.warning("AurigaNet resume: %d unexpected keys", len(unexpected))
        logger.info("AurigaNet loaded from %s", weights)

    model = model.to(device)
    return model, None


def auriganet_forward(
    model: nn.Module,
    images_chw: torch.Tensor,  # (B, 3, H, W) float32, ImageNet-normalised
    device: str,
    fp16: bool,
) -> torch.Tensor:
    """Return (B, NUM_CLASSES, H, W) upsampled logits from AurigaNet."""
    _, _, h, w = images_chw.shape
    x = images_chw.to(device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=fp16):
        seg_logits, _embed, _det = model(x)  # (B, C, H/4, W/4)

    seg_logits = torch.nn.functional.interpolate(
        seg_logits.float(), size=(h, w), mode="bilinear", align_corners=False,
    )
    return seg_logits  # (B, C, H, W)

"""Shared UPerNet (ConvNeXt-Base backbone) build helper.

UPerNet's HF wrapper (``UperNetForSemanticSegmentation``) returns dense
per-pixel logits via ``model(pixel_values=...).logits`` — the exact same
calling convention as SegFormer — and its ``AutoImageProcessor`` resolves to
the same ``SegformerImageProcessor`` class under the hood. This means
``_orfd_common.segformer_forward`` (despite the name) works unmodified as
UPerNet's forward function too; no UPerNet-specific forward pass is needed.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("train_orfd")

NUM_CLASSES = 3
UPERNET_HF_BASE = "openmmlab/upernet-convnext-base"


def build_upernet(device: str, fp16: bool, weights: str = ""):
    """Return (model, processor) for UPerNet fine-tuning. Mirrors build_segformer's shape."""
    from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

    logger.info("Loading UPerNet base weights from %s ...", UPERNET_HF_BASE)
    processor = AutoImageProcessor.from_pretrained(UPERNET_HF_BASE)
    model = UperNetForSemanticSegmentation.from_pretrained(
        UPERNET_HF_BASE,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )

    if weights:
        import torch
        from pathlib import Path
        if Path(weights).is_file():
            ckpt = torch.load(weights, map_location="cpu", weights_only=True)
            state_dict = ckpt.get("net", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(state_dict, strict=True)
            logger.info("UPerNet loaded from local checkpoint %s", weights)

    model = model.to(device)
    return model, processor

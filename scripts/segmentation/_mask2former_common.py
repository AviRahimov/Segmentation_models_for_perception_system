"""Shared Mask2Former (Swin-Base backbone) build + batch-conversion helpers.

Mask2Former is structurally different from SegFormer/UPerNet: it predicts
``num_queries`` (mask, class) pairs via Hungarian-matched set prediction, not
dense per-pixel logits, and its own ``forward(..., mask_labels, class_labels)``
computes the training loss internally. This means it does NOT fit
_orfd_common.py's train_one_epoch/evaluate (dense-logit + external criterion)
loop — train_mask2former.py implements its own loop using this module's
helpers instead.

For inference-shaped output (a dense per-user-class score map, matching what
SegFormer/UPerNet return via .logits), see mask2former_semantic_logits()
below — the same einsum HF's own post_process_semantic_segmentation uses
internally, minus the final argmax so callers can defer that decision.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("train_orfd")

NUM_CLASSES = 3
IGNORE_INDEX = 255
MASK2FORMER_HF_BASE = "facebook/mask2former-swin-base-ade-semantic"

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_mask2former(device: str, fp16: bool, weights: str = "",
                       backbone_id: str = MASK2FORMER_HF_BASE):
    """Return (model, processor)."""
    from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

    logger.info("Loading Mask2Former base weights from %s ...", backbone_id)
    processor = Mask2FormerImageProcessor.from_pretrained(backbone_id)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        backbone_id,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )

    if weights:
        from pathlib import Path
        if Path(weights).is_file():
            ckpt = torch.load(weights, map_location="cpu", weights_only=True)
            state_dict = ckpt.get("net", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(state_dict, strict=True)
            logger.info("Mask2Former loaded from local checkpoint %s", weights)

    model = model.to(device)
    return model, processor


def denormalize_to_uint8_rgb(images_chw: torch.Tensor) -> list[np.ndarray]:
    """(B,3,H,W) ImageNet-normalised float tensor -> list of HWC uint8 RGB ndarrays."""
    mean = _IMAGENET_MEAN.to(images_chw.device)
    std  = _IMAGENET_STD.to(images_chw.device)
    rgb_01 = images_chw * std + mean
    rgb_u8 = (rgb_01.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return [rgb_u8[i] for i in range(rgb_u8.shape[0])]


def prepare_batch(processor, images_chw: torch.Tensor, labels_hw: torch.Tensor, device: str):
    """Convert a (images, labels) batch from ORFDDataset into Mask2Former's
    processor-encoded inputs (pixel_values, mask_labels, class_labels)."""
    images = denormalize_to_uint8_rgb(images_chw)
    seg_maps = [labels_hw[i].cpu().numpy().astype(np.int64) for i in range(labels_hw.shape[0])]

    encoded = processor(
        images=images, segmentation_maps=seg_maps,
        ignore_index=IGNORE_INDEX, return_tensors="pt",
    )
    pixel_values = encoded["pixel_values"].to(device)
    mask_labels  = [m.to(device) for m in encoded["mask_labels"]]
    class_labels = [c.to(device) for c in encoded["class_labels"]]
    return pixel_values, mask_labels, class_labels


@torch.no_grad()
def mask2former_semantic_logits(model: nn.Module, pixel_values: torch.Tensor,
                                 target_size: tuple[int, int]) -> torch.Tensor:
    """Dense (num_classes, H, W) per-pixel class scores — same computation HF's
    post_process_semantic_segmentation uses internally, minus the final argmax,
    so callers (predict_logits / temporal smoothing) can defer that decision."""
    outputs = model(pixel_values=pixel_values)
    class_queries_logits = outputs.class_queries_logits[0]   # (Q, num_classes+1)
    masks_queries_logits = outputs.masks_queries_logits[0]    # (Q, h, w)

    masks_queries_logits = torch.nn.functional.interpolate(
        masks_queries_logits.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False,
    )[0]

    masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]  # drop "no object", (Q, C)
    masks_probs = masks_queries_logits.sigmoid()                    # (Q, H, W)
    segmentation = torch.einsum("qc,qhw->chw", masks_classes, masks_probs)  # (C, H, W)
    return segmentation

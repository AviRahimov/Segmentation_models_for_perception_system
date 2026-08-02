"""DINOv2 (frozen ViT-B/14 backbone) + lightweight decode head for ORFD.

Research finding behind this design (see the model-comparison plan): a frozen
DINOv2 backbone with only a *linear* head reaches just ~49 mIoU on ADE20K
(arXiv:2304.07193) — clearly weaker than a fully fine-tuned specialist. We use
a slightly-more-than-linear head (two 1x1 convs) while still keeping the
backbone frozen, matching the "cheap foundation-model candidate" role this
model plays in the comparison (train only a small head, not the whole network).

DINOv2 requires input dimensions divisible by its patch_size (14) — images are
resized to 518x518 (37x37 patches) before the backbone forward pass.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger("train_orfd")

NUM_CLASSES = 3
DINOV2_HF_BASE = "facebook/dinov2-base"
DINOV2_INPUT_SIZE = 518  # 37 * 14 (patch_size) — divisible, per DINOv2's requirement


class Dinov2SegHead(nn.Module):
    """Two 1x1-conv head on top of frozen DINOv2 patch tokens.

    Deliberately small — the point of this candidate is "how far do frozen
    foundation features get with minimal task-specific training," not to
    build a heavy decoder that would defeat the "cheap head" comparison.
    """

    def __init__(self, in_channels: int, num_classes: int = NUM_CLASSES, hidden: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.bn1   = nn.BatchNorm2d(hidden)
        self.act   = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        return self.conv2(x)


class Dinov2SegModel(nn.Module):
    """Frozen DINOv2 backbone + trainable Dinov2SegHead.

    forward(pixel_values) -> (B, num_classes, H_in, W_in) logits, already
    upsampled from the patch grid back to the model's own input resolution
    (DINOV2_INPUT_SIZE) — callers upsample again to the frame's native size.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, backbone_id: str = DINOV2_HF_BASE):
        super().__init__()
        from transformers import Dinov2Model

        self.backbone = Dinov2Model.from_pretrained(backbone_id)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

        self.patch_size = self.backbone.config.patch_size
        self.head = Dinov2SegHead(self.backbone.config.hidden_size, num_classes)

    def train(self, mode: bool = True):
        # Keep the frozen backbone in eval() (no dropout/stochastic depth)
        # even when the wrapping module is in train() for the head.
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        h, w = pixel_values.shape[-2:]
        gh, gw = h // self.patch_size, w // self.patch_size

        with torch.no_grad():
            out = self.backbone(pixel_values=pixel_values)
            tokens = out.last_hidden_state[:, 1:, :]  # drop CLS token, (B, gh*gw, C)

        b, _, c = tokens.shape
        grid = tokens.transpose(1, 2).reshape(b, c, gh, gw)  # (B, C, gh, gw)
        logits = self.head(grid)  # (B, num_classes, gh, gw)
        logits = torch.nn.functional.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False,
        )
        return logits


def build_dinov2(device: str, fp16: bool, weights: str = "",
                  backbone_id: str = DINOV2_HF_BASE) -> tuple[nn.Module, None]:
    """Return (model, None) — mirrors build_auriganet's (model, processor) shape."""
    from pathlib import Path

    logger.info("Building DINOv2 (%s, frozen backbone) + lightweight head ...", backbone_id)
    model = Dinov2SegModel(num_classes=NUM_CLASSES, backbone_id=backbone_id)

    if weights and Path(weights).is_file():
        ckpt = torch.load(weights, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("net", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("DINOv2 resume: %d missing keys", len(missing))
        logger.info("DINOv2 head loaded from %s", weights)

    model = model.to(device)
    return model, None


def dinov2_forward(
    model: nn.Module,
    images_chw: torch.Tensor,  # (B, 3, H, W) float32, ImageNet-normalised
    device: str,
    fp16: bool,
) -> torch.Tensor:
    """Return (B, NUM_CLASSES, H, W) upsampled logits from the DINOv2 head."""
    _, _, h, w = images_chw.shape
    x = images_chw.to(device)
    x = torch.nn.functional.interpolate(
        x, size=(DINOV2_INPUT_SIZE, DINOV2_INPUT_SIZE), mode="bilinear", align_corners=False,
    )

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=fp16):
        logits = model(x)  # (B, C, DINOV2_INPUT_SIZE, DINOV2_INPUT_SIZE)

    logits = torch.nn.functional.interpolate(
        logits.float(), size=(h, w), mode="bilinear", align_corners=False,
    )
    return logits

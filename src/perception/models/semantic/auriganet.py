"""AurigaNet semantic segmentation wrapper.

Fine-tuned mode only: the model is expected to output ``num_seg_classes``
channels that map directly to the configured user classes (no ADE20K LUT).

Inference: input is resized to 640×640 (AurigaNet's native size), forwarded,
then the seg logits are upsampled back to the original frame resolution before
returning probabilities.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ...config.schema import ClassDef
from ..backends.base import InferenceBackend
from .base import SemanticModel

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

INPUT_SIZE = 640  # AurigaNet native input resolution


class AurigaNetSemanticModel(SemanticModel):
    """AurigaNet wrapper implementing the SemanticModel interface.

    Loads the vendored architecture with a modified segmentation head that
    outputs ``num_seg_classes`` channels.  The detection and lane heads are
    preserved in the model graph for future use but are frozen on init and
    not called during inference.

    Args:
        weights:     Path to a ``.pth`` checkpoint (``{"net": state_dict}``)
                     or empty string for random initialisation.
        backend:     Inference backend (passed through; currently unused).
        device:      ``"cuda"`` or ``"cpu"``.
        fp16:        Enable bfloat16 mixed precision (CUDA only).
        num_classes: Number of segmentation output channels (default 3 for ORFD).
    """

    def __init__(
        self,
        weights: str = "",
        backend: InferenceBackend | None = None,
        device: str = "cpu",
        fp16: bool = False,
        num_classes: int | None = None,
        tta: bool = False,
    ) -> None:
        from ._vendored.auriganet import AurigaNetArch

        self._device = device
        self._fp16 = bool(fp16) and isinstance(device, str) and device.startswith("cuda")
        self._num_classes = num_classes if num_classes is not None else 3

        self._model = AurigaNetArch(
            num_seg_classes=self._num_classes,
            with_detection=False,
        )

        if weights and Path(weights).is_file():
            ckpt = torch.load(weights, map_location="cpu", weights_only=True)
            state_dict = ckpt.get("net", ckpt) if isinstance(ckpt, dict) else ckpt
            missing, unexpected = self._model.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning("AurigaNet: %d missing keys in checkpoint", len(missing))
            if unexpected:
                logger.warning("AurigaNet: %d unexpected keys in checkpoint", len(unexpected))
            logger.info("AurigaNet loaded from %s (%d classes)", weights, self._num_classes)
        elif weights:
            logger.warning("AurigaNet: weights path %r not found; using random init.", weights)
        else:
            logger.info("AurigaNet: no weights provided, using random init.")

        # Freeze lane head — it's BDD100K-specific and never trained on ORFD.
        for p in self._model.Seg.lane.parameters():
            p.requires_grad_(False)

        self._model.eval()
        self._model = self._model.to(device)
        if self._fp16:
            self._model = self._model.half()

        self._semantic_classes: list[ClassDef] = []
        self._tta: bool = tta

    # ------------------------------------------------------------------ #

    def warmup(self, classes: Sequence[ClassDef]) -> None:
        sem = [c for c in classes if c.is_semantic]
        if sem and len(sem) != self._num_classes:
            logger.warning(
                "AurigaNet: config has %d semantic classes but model outputs %d channels; "
                "classes beyond index %d will never be predicted.",
                len(sem), self._num_classes, self._num_classes - 1,
            )
        self._semantic_classes = sem
        logger.info(
            "AurigaNet warmed up: %d model channels, %d config classes.",
            self._num_classes, len(sem),
        )

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self._semantic_classes)

    # ------------------------------------------------------------------ #

    def _forward_single(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """Single forward pass → probs (C, H, W) at original resolution."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        x = resized.astype(np.float32) / 255.0
        x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
        x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)  # (1, 3, 640, 640)
        x = x.to(self._device)
        if self._fp16:
            x = x.half()
        seg_logits, _embed, _det = self._model(x)  # (1, C, H/4, W/4)
        seg_logits = F.interpolate(
            seg_logits, size=(h, w), mode="bilinear", align_corners=False,
        )[0]  # (C, H, W)
        return torch.softmax(seg_logits.float(), dim=0).to(seg_logits.dtype)

    @torch.inference_mode()
    def predict_logits(self, frame_bgr: np.ndarray) -> torch.Tensor:
        if not self._semantic_classes:
            raise RuntimeError(
                "AurigaNetSemanticModel.predict_logits called before warmup()."
            )
        probs = self._forward_single(frame_bgr)
        if self._tta:
            probs_flip = self._forward_single(np.fliplr(frame_bgr))
            probs_flip = torch.flip(probs_flip, dims=[-1])
            probs = (probs + probs_flip) * 0.5
        return probs

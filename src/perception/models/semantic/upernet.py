"""UPerNet (ConvNeXt-Base backbone) semantic segmentation wrapper.

Fine-tuned mode only (no ADE20K LUT). UPerNet's HF wrapper returns dense
per-pixel logits via ``model(pixel_values=...).logits`` — the same
convention as SegFormer's fine-tuned mode — so this wrapper mirrors
segformer.py's simpler (non-LUT) code path directly.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

from ...config.schema import ClassDef
from ..backends.base import InferenceBackend
from .base import SemanticModel

logger = logging.getLogger(__name__)

_HF_BASE = "openmmlab/upernet-convnext-base"


class UPerNetSemanticModel(SemanticModel):
    def __init__(
        self,
        weights: str = "",
        backend: InferenceBackend | None = None,
        device: str = "cuda",
        fp16: bool = True,
        num_classes: int | None = None,
    ) -> None:
        from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

        self._device = device
        self._fp16 = bool(fp16) and isinstance(device, str) and device.startswith("cuda")
        self._num_classes = num_classes if num_classes is not None else 3

        self._processor = AutoImageProcessor.from_pretrained(_HF_BASE)

        _is_local = Path(weights).suffix == ".pth"
        self._model = UperNetForSemanticSegmentation.from_pretrained(
            _HF_BASE, num_labels=self._num_classes, ignore_mismatched_sizes=True,
        )
        if _is_local and Path(weights).is_file():
            ckpt = torch.load(weights, map_location="cpu", weights_only=True)
            state_dict = ckpt.get("net", ckpt) if isinstance(ckpt, dict) else ckpt
            self._model.load_state_dict(state_dict, strict=True)
            logger.info("UPerNet loaded from local checkpoint %s (%d classes)",
                        weights, self._num_classes)
        elif weights:
            logger.warning("UPerNet: weights path %r not found; using ADE20K-pretrained init.", weights)

        self._model.eval()

        if backend is not None:
            self._model = backend.prepare(self._model, device=self._device, fp16=self._fp16, engine_path="")
        else:
            self._model = self._model.to(self._device)
            if self._fp16:
                self._model = self._model.half()

        self._semantic_classes: list[ClassDef] = []

    # ------------------------------------------------------------------ #
    def warmup(self, classes: Sequence[ClassDef]) -> None:
        sem = [c for c in classes if c.is_semantic]
        if sem and len(sem) != self._num_classes:
            logger.warning(
                "UPerNet: config has %d semantic classes but model outputs %d channels; "
                "classes beyond index %d will never be predicted.",
                len(sem), self._num_classes, self._num_classes - 1,
            )
        self._semantic_classes = sem
        logger.info("UPerNet warmed up: %d model channels, %d config classes.",
                    self._num_classes, len(sem))

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self._semantic_classes)

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def predict_logits(self, frame_bgr: np.ndarray) -> torch.Tensor:
        if not self._semantic_classes:
            raise RuntimeError("UPerNetSemanticModel.predict_logits called before warmup().")

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=rgb, return_tensors="pt")
        pixel_values: torch.Tensor = inputs["pixel_values"].to(self._device)
        if self._fp16:
            pixel_values = pixel_values.half()

        outputs = self._model(pixel_values=pixel_values)
        logits = outputs.logits  # (1, C, H, W) — UPerNet's decode head already upsamples internally
        logits = torch.nn.functional.interpolate(
            logits.float(), size=(h, w), mode="bilinear", align_corners=False,
        )[0]
        probs = torch.softmax(logits, dim=0)
        return probs  # (C_user, H, W)

"""Mask2Former (Swin-Base backbone) semantic segmentation wrapper.

Fine-tuned mode only (no ADE20K LUT — unlike segformer.py's dual-mode
design, this class only supports a checkpoint fine-tuned directly on the
project's user classes). Mask2Former predicts (mask, class) query pairs
rather than dense per-pixel logits; predict_logits() converts them into a
dense per-user-class score map via the same computation
transformers.Mask2FormerImageProcessor.post_process_semantic_segmentation
uses internally, minus its final argmax (temporal smoothing needs the raw
scores, not a pre-argmaxed map — see SemanticModel.predict_logits's contract).
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

_HF_BASES: dict[str, str] = {
    "mask2former":       "facebook/mask2former-swin-base-ade-semantic",
    "mask2former-base":  "facebook/mask2former-swin-base-ade-semantic",
    "mask2former-large": "facebook/mask2former-swin-large-ade-semantic",
}


class Mask2FormerSemanticModel(SemanticModel):
    def __init__(
        self,
        weights: str = "",
        name: str = "mask2former",
        backend: InferenceBackend | None = None,
        device: str = "cuda",
        fp16: bool = True,
        num_classes: int | None = None,
    ) -> None:
        from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

        self._device = device
        self._fp16 = bool(fp16) and isinstance(device, str) and device.startswith("cuda")
        self._num_classes = num_classes if num_classes is not None else 3
        _hf_base = _HF_BASES.get(name.lower().strip(), _HF_BASES["mask2former"])

        _is_local = Path(weights).suffix == ".pth"
        self._processor = Mask2FormerImageProcessor.from_pretrained(_hf_base)

        if _is_local and Path(weights).is_file():
            self._model = Mask2FormerForUniversalSegmentation.from_pretrained(
                _hf_base, num_labels=self._num_classes, ignore_mismatched_sizes=True,
            )
            ckpt = torch.load(weights, map_location="cpu", weights_only=True)
            state_dict = ckpt.get("net", ckpt) if isinstance(ckpt, dict) else ckpt
            try:
                self._model.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                raise ValueError(
                    f"Checkpoint at {weights!r} does not match the Mask2Former architecture "
                    f"(name={name!r}). Check that config.yaml's models.semantic.name actually "
                    f"matches this checkpoint's architecture -- name and weights must be "
                    f"changed together."
                ) from e
            logger.info("Mask2Former loaded from local checkpoint %s (%d classes)",
                        weights, self._num_classes)
        else:
            self._model = Mask2FormerForUniversalSegmentation.from_pretrained(
                _hf_base, num_labels=self._num_classes, ignore_mismatched_sizes=True,
            )
            if weights:
                logger.warning("Mask2Former: weights %r not found; using ADE20K-pretrained init.", weights)

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
                "Mask2Former: config has %d semantic classes but model outputs %d channels; "
                "classes beyond index %d will never be predicted.",
                len(sem), self._num_classes, self._num_classes - 1,
            )
        self._semantic_classes = sem
        logger.info("Mask2Former warmed up: %d model channels, %d config classes.",
                    self._num_classes, len(sem))

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self._semantic_classes)

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def predict_logits(self, frame_bgr: np.ndarray) -> torch.Tensor:
        if not self._semantic_classes:
            raise RuntimeError("Mask2FormerSemanticModel.predict_logits called before warmup().")

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=[rgb], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device)
        if self._fp16:
            pixel_values = pixel_values.half()

        outputs = self._model(pixel_values=pixel_values)
        class_queries_logits = outputs.class_queries_logits[0].float()  # (Q, C+1)
        masks_queries_logits = outputs.masks_queries_logits[0].float()  # (Q, h, w)

        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False,
        )[0]

        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]  # drop "no object", (Q, C)
        masks_probs = masks_queries_logits.sigmoid()                   # (Q, H, W)
        segmentation = torch.einsum("qc,qhw->chw", masks_classes, masks_probs)  # (C, H, W)
        return segmentation

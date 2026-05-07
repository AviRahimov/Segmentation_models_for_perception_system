"""SegFormer-B2 semantic segmentation wrapper.

The model is closed-vocabulary on ADE20K (150 channels). The configured
``ade20k_indices`` define a merge LUT (shape ``[150, C_user]``) that
combines native predictions into the user's class set.

Critically, we softmax over the 150 native channels **before** applying
the LUT, so the merged tensor is a sum of *posterior probabilities* per
user class, not raw logits. Summing logits would be wrong: ADE20K's
per-channel logit biases differ by several units, so a class with one
high-bias channel (e.g. ``grass`` = #9) can systematically outscore a
class with multiple lower-bias channels (e.g. ``sand_gravel`` =
#46+#91), regardless of what the model actually "sees". Probabilities,
being normalised to a simplex, are comparable across channels, so the
sum gives the model's actual posterior over the user-defined class set.

The merged tensor (still per-pixel scores) is what the EMA smoother
operates on and what the renderer argmaxes downstream.
"""
from __future__ import annotations

import logging
from typing import Sequence

import cv2
import numpy as np
import torch

from ...config.schema import ClassDef
from ..backends.base import InferenceBackend
from .base import SemanticModel

logger = logging.getLogger(__name__)


class SegFormerSemanticModel(SemanticModel):
    def __init__(
        self,
        weights: str = "nvidia/segformer-b2-finetuned-ade-512-512",
        backend: InferenceBackend | None = None,
        device: str = "cuda",
        fp16: bool = True,
    ) -> None:
        # Lazy import for environments without transformers (e.g. unit tests).
        from transformers import (  # type: ignore
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )

        self._device = device
        self._fp16 = bool(fp16) and isinstance(device, str) and device.startswith("cuda")
        self._processor = SegformerImageProcessor.from_pretrained(weights)
        self._model = SegformerForSemanticSegmentation.from_pretrained(weights)
        self._model.eval()

        if backend is not None:
            self._model = backend.prepare(self._model, device=self._device, fp16=self._fp16)
        else:
            self._model = self._model.to(self._device)
            if self._fp16:
                self._model = self._model.half()

        self._semantic_classes: list[ClassDef] = []
        self._lut: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    def warmup(self, classes: Sequence[ClassDef]) -> None:
        sem = [c for c in classes if c.is_semantic]
        self._semantic_classes = sem
        if not sem:
            self._lut = None
            logger.info("SegFormer: no semantic classes configured.")
            return

        n_ade = int(getattr(self._model.config, "num_labels", 150))
        lut = torch.zeros(n_ade, len(sem), dtype=torch.float32)
        for j, c in enumerate(sem):
            for i in c.ade20k_indices:
                if not 0 <= int(i) < n_ade:
                    raise ValueError(
                        f"Class {c.name!r}: ade20k index {i} out of range [0, {n_ade})"
                    )
                lut[int(i), j] = 1.0

        param_dtype = next(self._model.parameters()).dtype
        self._lut = lut.to(self._device).to(param_dtype)
        logger.info(
            "SegFormer warmed up: %d user classes derived from %d ADE20K channels",
            len(sem), int(self._lut.sum().item()),
        )

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self._semantic_classes)

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def predict_logits(self, frame_bgr: np.ndarray) -> torch.Tensor:
        if self._lut is None or not self._semantic_classes:
            raise RuntimeError(
                "SegFormerSemanticModel.predict_logits called before warmup() "
                "(or with no semantic classes). Configure semantic classes "
                "with ade20k_indices in config.yaml."
            )

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=rgb, return_tensors="pt")
        pixel_values: torch.Tensor = inputs["pixel_values"].to(self._device)
        if self._fp16:
            pixel_values = pixel_values.half()

        outputs = self._model(pixel_values=pixel_values)
        logits = outputs.logits  # (1, 150, H/4, W/4)

        # Upsample to native frame resolution before merging so the EMA
        # state stays aligned across frames of the same source.
        logits = torch.nn.functional.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )[0]  # (150, H, W)

        # Softmax across the 150 native ADE20K channels first, *then* sum
        # the relevant native probabilities into each user class via the
        # LUT. See module docstring for why summing raw logits is wrong.
        # Compute softmax in fp32 for numerical stability under fp16 inputs.
        probs = torch.softmax(logits.float(), dim=0).to(logits.dtype)
        merged = torch.einsum("cu,chw->uhw", self._lut, probs)  # (C_user, H, W)
        return merged

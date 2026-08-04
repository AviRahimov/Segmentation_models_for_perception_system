"""DINOv2 (frozen ViT-B/14 backbone) + lightweight head semantic segmentation
wrapper. Fine-tuned mode only. See scripts/segmentation/_dinov2_common.py for
the shared Dinov2SegModel/head architecture used by both training and here.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

from ...config.schema import ClassDef
from ..backends.base import InferenceBackend
from .base import SemanticModel

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_SCRIPTS_SEG_DIR = Path(__file__).resolve().parents[4] / "scripts" / "segmentation"


_HF_BASES: dict[str, str] = {
    "dinov2":       "facebook/dinov2-base",
    "dinov2-base":  "facebook/dinov2-base",
    "dinov2-large": "facebook/dinov2-large",
}


class DINOv2SemanticModel(SemanticModel):
    def __init__(
        self,
        weights: str = "",
        name: str = "dinov2",
        backend: InferenceBackend | None = None,
        device: str = "cuda",
        fp16: bool = True,
        num_classes: int | None = None,
    ) -> None:
        if str(_SCRIPTS_SEG_DIR) not in sys.path:
            sys.path.insert(0, str(_SCRIPTS_SEG_DIR))
        from _dinov2_common import DINOV2_INPUT_SIZE, Dinov2SegModel

        self._device = device
        self._fp16 = bool(fp16) and isinstance(device, str) and device.startswith("cuda")
        self._num_classes = num_classes if num_classes is not None else 3
        self._input_size = DINOV2_INPUT_SIZE
        _hf_base = _HF_BASES.get(name.lower().strip(), _HF_BASES["dinov2"])

        self._model = Dinov2SegModel(num_classes=self._num_classes, backbone_id=_hf_base)

        if weights and Path(weights).is_file():
            ckpt = torch.load(weights, map_location="cpu", weights_only=True)
            state_dict = ckpt.get("net", ckpt) if isinstance(ckpt, dict) else ckpt
            missing, unexpected = self._model.load_state_dict(state_dict, strict=False)
            total = len(self._model.state_dict())
            if len(missing) + len(unexpected) > 0.1 * total:
                raise ValueError(
                    f"Checkpoint at {weights!r} does not look like a DINOv2 ({name!r}) "
                    f"checkpoint ({len(missing)} missing / {len(unexpected)} unexpected of "
                    f"{total} keys). Check that config.yaml's models.semantic.name/weights "
                    f"are paired correctly."
                )
            if missing:
                logger.warning("DINOv2: %d missing keys in checkpoint", len(missing))
            if unexpected:
                logger.warning("DINOv2: %d unexpected keys in checkpoint", len(unexpected))
            logger.info("DINOv2 loaded from %s (%d classes)", weights, self._num_classes)
        elif weights:
            logger.warning("DINOv2: weights path %r not found; head is randomly initialised.", weights)

        self._model.eval()
        self._model = self._model.to(device)
        if self._fp16:
            self._model = self._model.half()

        self._semantic_classes: list[ClassDef] = []

    # ------------------------------------------------------------------ #
    def warmup(self, classes: Sequence[ClassDef]) -> None:
        sem = [c for c in classes if c.is_semantic]
        if sem and len(sem) != self._num_classes:
            logger.warning(
                "DINOv2: config has %d semantic classes but model outputs %d channels; "
                "classes beyond index %d will never be predicted.",
                len(sem), self._num_classes, self._num_classes - 1,
            )
        self._semantic_classes = sem
        logger.info("DINOv2 warmed up: %d model channels, %d config classes.",
                    self._num_classes, len(sem))

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self._semantic_classes)

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def predict_logits(self, frame_bgr: np.ndarray) -> torch.Tensor:
        if not self._semantic_classes:
            raise RuntimeError("DINOv2SemanticModel.predict_logits called before warmup().")

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self._input_size, self._input_size), interpolation=cv2.INTER_LINEAR)
        x = resized.astype(np.float32) / 255.0
        x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
        x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self._device)
        if self._fp16:
            x = x.half()

        logits = self._model(x)  # (1, C, input_size, input_size)
        logits = torch.nn.functional.interpolate(
            logits.float(), size=(h, w), mode="bilinear", align_corners=False,
        )[0]
        return logits

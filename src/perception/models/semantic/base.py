"""Semantic segmentation ABC."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import torch

from ...config.schema import ClassDef


class SemanticModel(ABC):
    """Closed-vocabulary semantic segmentation that returns *raw logits*."""

    @abstractmethod
    def warmup(self, classes: Sequence[ClassDef]) -> None:
        """Configure the model for the given user classes.

        For closed-vocab models like SegFormer this builds the merge LUT
        that maps the model's native logit channels to the user's class set.
        """

    @abstractmethod
    def predict_logits(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """Return raw merged user-class logits ``[C_user, H, W]``.

        Argmax must NOT be taken here - temporal smoothing operates on the
        raw tensor. The returned tensor's spatial size matches ``frame_bgr``.
        """

    @property
    @abstractmethod
    def class_names(self) -> tuple[str, ...]:
        """Tuple of user-class names parallel to dim 0 of returned logits."""

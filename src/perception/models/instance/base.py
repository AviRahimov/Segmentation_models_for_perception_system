"""Instance segmentation ABC."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from ...config.schema import ClassDef
from ...core.types import Detection


class InstanceModel(ABC):
    """Open-vocabulary instance segmentation."""

    @abstractmethod
    def warmup(self, classes: Sequence[ClassDef]) -> None:
        """Configure the model for the given user classes.

        Implementations must cache any text embeddings here so that
        :meth:`predict` never re-invokes a text encoder.
        """

    @abstractmethod
    def predict(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Run inference on a single BGR frame and return :class:`Detection`s.

        Returned detections must have ``class_name`` matching one of the
        :class:`ClassDef.name`s passed to :meth:`warmup`.
        """

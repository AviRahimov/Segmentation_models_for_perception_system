"""No-op instance model — used when ``models.instance.enabled: false``."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ...config.schema import ClassDef
from ...core.types import Detection
from .base import InstanceModel


class NullInstanceModel(InstanceModel):
    def warmup(self, classes: Sequence[ClassDef]) -> None:
        pass

    def predict(self, frame_bgr: np.ndarray) -> list[Detection]:
        return []

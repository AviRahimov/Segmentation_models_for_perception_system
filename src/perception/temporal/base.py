"""Abstract base classes for temporal components."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
import torch

from ..core.types import Detection


class LogitsSmoother(ABC):
    """Causal smoother over a sequence of logit tensors."""

    @abstractmethod
    def update(self, logits: torch.Tensor) -> torch.Tensor:
        """Consume the current frame's raw logits and return the smoothed
        tensor (same shape). Implementations MUST be causal: only past and
        present frames may influence the output.
        """

    @abstractmethod
    def reset(self) -> None:
        """Discard all accumulated state (called on scene cut and seek)."""


class InstanceTracker(ABC):
    """Tracks instance detections across frames, optionally refining masks."""

    @abstractmethod
    def update(
        self,
        frame_bgr: np.ndarray,
        detections: Sequence[Detection],
    ) -> list[Detection]:
        """Return a new list of detections with ``track_id`` populated.
        May modify masks (e.g. SAM2 propagation)."""

    @abstractmethod
    def reset(self) -> None:
        """Discard track memory (called on scene cut and seek)."""


class SceneCutDetector(ABC):
    """Detects abrupt scene changes from past+present frames only."""

    @abstractmethod
    def update(self, frame_bgr: np.ndarray) -> bool:
        """Return ``True`` if the current frame is a scene cut."""

    @abstractmethod
    def reset(self) -> None: ...

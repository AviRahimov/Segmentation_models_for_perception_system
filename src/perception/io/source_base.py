"""Abstract base class for frame sources."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class FrameSource(ABC):
    """A unified interface for video files, webcams, and image directories.

    Implementations must be safe to call from a single producer thread.
    """

    @abstractmethod
    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read the next BGR frame.

        Returns:
            ``(True, frame)`` on success or ``(False, None)`` at EOF.
        """

    @abstractmethod
    def seek(self, frame_idx: int) -> None:
        """Reposition the source. No-op on non-seekable streams."""

    @abstractmethod
    def total_frames(self) -> int:
        """Return total frame count, or ``-1`` if unknown (e.g. webcam)."""

    @abstractmethod
    def fps(self) -> float:
        """Return native FPS, or a sensible default for unknown sources."""

    @abstractmethod
    def release(self) -> None:
        """Release any underlying resources (capture handles, file descriptors)."""

    @property
    @abstractmethod
    def position(self) -> int:
        """0-based index of the *next* frame to be returned by ``read``."""

    @property
    @abstractmethod
    def is_seekable(self) -> bool: ...

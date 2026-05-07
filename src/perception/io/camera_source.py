"""Webcam frame source backed by OpenCV's VideoCapture(index)."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .source_base import FrameSource


class CameraSource(FrameSource):
    def __init__(self, index: int = 0, fps_hint: float = 30.0) -> None:
        self._index = int(index)
        self._cap = cv2.VideoCapture(self._index)
        if not self._cap.isOpened():
            raise RuntimeError(f"OpenCV failed to open camera index {index}")
        reported = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = float(reported) if reported and reported > 0.0 else float(fps_hint)
        self._pos = 0

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, None
        self._pos += 1
        return True, frame

    def seek(self, frame_idx: int) -> None:
        # Webcams don't support seeking; ignore silently.
        return

    def total_frames(self) -> int:
        return -1

    def fps(self) -> float:
        return self._fps

    def release(self) -> None:
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()

    @property
    def position(self) -> int:
        return self._pos

    @property
    def is_seekable(self) -> bool:
        return False

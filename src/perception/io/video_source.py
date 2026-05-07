"""Video-file frame source backed by OpenCV's VideoCapture."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .source_base import FrameSource


class VideoFileSource(FrameSource):
    def __init__(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Video file not found: {p}")
        self._cap = cv2.VideoCapture(str(p))
        if not self._cap.isOpened():
            raise RuntimeError(f"OpenCV failed to open video: {p}")
        self._total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = float(fps) if fps and fps > 0.0 else 30.0
        self._pos = 0

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, None
        self._pos += 1
        return True, frame

    def seek(self, frame_idx: int) -> None:
        idx = max(0, min(int(frame_idx), max(0, self._total - 1)))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        self._pos = idx

    def total_frames(self) -> int:
        return self._total

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
        return self._total > 0

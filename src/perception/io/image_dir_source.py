"""Image-directory frame source.

Files in ``dir_path`` matching ``glob`` are loaded in sorted order and
returned as sequential frames.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .source_base import FrameSource


class ImageDirSource(FrameSource):
    def __init__(
        self,
        dir_path: str | Path,
        glob: str = "*.png",
        fps_hint: float = 30.0,
    ) -> None:
        d = Path(dir_path)
        if not d.is_dir():
            raise NotADirectoryError(f"Not a directory: {d}")
        self._files = sorted(p for p in d.glob(glob) if p.is_file())
        if not self._files:
            raise FileNotFoundError(f"No files matching {glob!r} under {d}")
        self._pos = 0
        self._fps = float(fps_hint)

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if self._pos >= len(self._files):
            return False, None
        path = self._files[self._pos]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        self._pos += 1
        if img is None:
            return False, None
        return True, img

    def seek(self, frame_idx: int) -> None:
        self._pos = max(0, min(int(frame_idx), len(self._files)))

    def total_frames(self) -> int:
        return len(self._files)

    def fps(self) -> float:
        return self._fps

    def release(self) -> None:
        return

    @property
    def position(self) -> int:
        return self._pos

    @property
    def is_seekable(self) -> bool:
        return True

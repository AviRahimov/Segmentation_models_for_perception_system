"""Histogram-based scene-cut detection (causal, single-frame look-back)."""
from __future__ import annotations

import cv2
import numpy as np

from .base import SceneCutDetector


class HistogramSceneCutDetector(SceneCutDetector):
    """Detects scene cuts via Bhattacharyya distance between consecutive
    HSV histograms.

    Bhattacharyya is bounded in ``[0, 1]`` so a single threshold is intuitive
    (typical: 0.4 - 0.55). The detector is causal: only the previous
    frame's histogram is consulted.
    """

    def __init__(
        self,
        threshold: float = 0.45,
        hsv_bins: tuple[int, int, int] = (16, 16, 8),
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self._threshold = float(threshold)
        self._bins = tuple(int(b) for b in hsv_bins)
        self._prev: np.ndarray | None = None

    def update(self, frame_bgr: np.ndarray) -> bool:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None,
            list(self._bins),
            [0, 180, 0, 256, 0, 256],
        )
        cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
        if self._prev is None:
            self._prev = hist
            return False
        d = float(cv2.compareHist(self._prev, hist, cv2.HISTCMP_BHATTACHARYYA))
        self._prev = hist
        return d >= self._threshold

    def reset(self) -> None:
        self._prev = None

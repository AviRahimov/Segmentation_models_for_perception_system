"""Lightweight IoU-based instance tracker across frames.

Greedily matches new detections to the previous frame's by class-conditioned
IoU, propagating ``track_id``s where the match exceeds a threshold and
assigning fresh ids otherwise.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..core.geometry import iou_xyxy
from ..core.types import Detection
from .base import InstanceTracker

logger = logging.getLogger(__name__)


@dataclass
class _PrevDet:
    class_name: str
    bbox: tuple[int, int, int, int]
    track_id: int


class IoUInstanceTracker(InstanceTracker):
    """Greedy class-conditioned IoU tracker with monotonic ids."""

    def __init__(self, iou_threshold: float = 0.3) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold}")
        self._iou_threshold = float(iou_threshold)
        self._next_id = 1
        self._prev: list[_PrevDet] = []

    def update(
        self,
        frame_bgr: np.ndarray,
        detections: Sequence[Detection],
    ) -> list[Detection]:
        out: list[Detection] = []
        used: set[int] = set()
        for det in detections:
            best_idx, best_iou = -1, 0.0
            for i, prev in enumerate(self._prev):
                if i in used or prev.class_name != det.class_name:
                    continue
                iou = iou_xyxy(det.bbox_xyxy, prev.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_idx >= 0 and best_iou >= self._iou_threshold:
                track_id = self._prev[best_idx].track_id
                used.add(best_idx)
            else:
                track_id = self._next_id
                self._next_id += 1
            out.append(
                Detection(
                    class_name=det.class_name,
                    score=det.score,
                    bbox_xyxy=det.bbox_xyxy,
                    mask=det.mask,
                    track_id=track_id,
                )
            )
        self._prev = [
            _PrevDet(class_name=d.class_name, bbox=d.bbox_xyxy, track_id=d.track_id or 0)
            for d in out
        ]
        return out

    def reset(self) -> None:
        # Keep _next_id monotonically increasing across resets so tracks
        # logged before/after a scene cut never collide.
        self._prev = []

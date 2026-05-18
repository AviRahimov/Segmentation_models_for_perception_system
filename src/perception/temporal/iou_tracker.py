"""IoU-based instance tracker with hold and EMA smoothing.

Two stabilisation layers on top of greedy IoU matching:
  Hold  — re-emit a missed track for up to ``max_hold_frames`` frames with
           a decaying score, preventing single-frame blink-outs near threshold.
  EMA   — exponentially blend bbox coordinates and confidence each matched
           frame, eliminating pixel-level jitter from detector noise.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
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
    missed_count: int = 0
    smoothed_bbox: tuple[float, float, float, float] = field(
        default_factory=lambda: (0.0, 0.0, 0.0, 0.0)
    )
    smoothed_score: float = 1.0
    mask: np.ndarray | None = None


def _blend_bbox(
    raw: tuple[int, int, int, int],
    prev: tuple[float, float, float, float],
    alpha: float,
) -> tuple[float, float, float, float]:
    """EMA blend: alpha * raw + (1-alpha) * prev."""
    return tuple(alpha * r + (1.0 - alpha) * p for r, p in zip(raw, prev))  # type: ignore[return-value]


def _bbox_to_int(
    b: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    return (int(round(b[0])), int(round(b[1])), int(round(b[2])), int(round(b[3])))


class IoUInstanceTracker(InstanceTracker):
    """Greedy class-conditioned IoU tracker with hold and EMA smoothing."""

    def __init__(
        self,
        iou_threshold: float = 0.30,
        max_hold_frames: int = 2,
        hold_score_decay: float = 0.85,
        bbox_alpha: float = 0.50,
        score_alpha: float = 0.40,
    ) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold}")
        self._iou_threshold = float(iou_threshold)
        self._max_hold = int(max_hold_frames)
        self._hold_decay = float(hold_score_decay)
        self._bbox_alpha = float(bbox_alpha)
        self._score_alpha = float(score_alpha)
        self._next_id = 1
        self._prev: list[_PrevDet] = []

    def update(
        self,
        frame_bgr: np.ndarray,
        detections: Sequence[Detection],
    ) -> list[Detection]:
        out: list[Detection] = []
        used_prev: set[int] = set()

        # ── Step 1: match each new detection against previous tracks ─────────
        for det in detections:
            best_idx, best_iou = -1, 0.0
            for i, prev in enumerate(self._prev):
                if i in used_prev or prev.class_name != det.class_name:
                    continue
                iou = iou_xyxy(det.bbox_xyxy, prev.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_idx >= 0 and best_iou >= self._iou_threshold:
                # Matched: EMA-blend bbox and score.
                prev = self._prev[best_idx]
                used_prev.add(best_idx)

                sb = _blend_bbox(det.bbox_xyxy, prev.smoothed_bbox, self._bbox_alpha)
                ss = self._score_alpha * det.score + (1.0 - self._score_alpha) * prev.smoothed_score

                emit_bbox = _bbox_to_int(sb)
                out.append(Detection(
                    class_name=det.class_name,
                    score=float(ss),
                    bbox_xyxy=emit_bbox,
                    mask=det.mask,
                    track_id=prev.track_id,
                ))
                self._prev[best_idx] = _PrevDet(
                    class_name=det.class_name,
                    bbox=emit_bbox,
                    track_id=prev.track_id,
                    missed_count=0,
                    smoothed_bbox=sb,
                    smoothed_score=ss,
                    mask=det.mask,
                )
            else:
                # New track: raw values pass through unchanged.
                raw_f = tuple(float(v) for v in det.bbox_xyxy)
                tid = self._next_id
                self._next_id += 1
                out.append(Detection(
                    class_name=det.class_name,
                    score=det.score,
                    bbox_xyxy=det.bbox_xyxy,
                    mask=det.mask,
                    track_id=tid,
                ))
                # Register as a new _prev entry (appended; will survive step 2).
                self._prev.append(_PrevDet(
                    class_name=det.class_name,
                    bbox=det.bbox_xyxy,
                    track_id=tid,
                    missed_count=0,
                    smoothed_bbox=raw_f,  # type: ignore[arg-type]
                    smoothed_score=det.score,
                    mask=det.mask,
                ))
                used_prev.add(len(self._prev) - 1)

        # ── Step 2: hold or expire unmatched previous tracks ─────────────────
        next_prev: list[_PrevDet] = []
        for i, prev in enumerate(self._prev):
            if i in used_prev:
                # Already updated in step 1 (and re-appended inline).
                # Collect only the entries we wrote back.
                next_prev.append(self._prev[i])
                continue

            new_missed = prev.missed_count + 1
            if new_missed <= self._max_hold:
                decayed = prev.smoothed_score * self._hold_decay
                out.append(Detection(
                    class_name=prev.class_name,
                    score=float(decayed),
                    bbox_xyxy=_bbox_to_int(prev.smoothed_bbox),
                    mask=prev.mask,
                    track_id=prev.track_id,
                ))
                next_prev.append(_PrevDet(
                    class_name=prev.class_name,
                    bbox=_bbox_to_int(prev.smoothed_bbox),
                    track_id=prev.track_id,
                    missed_count=new_missed,
                    smoothed_bbox=prev.smoothed_bbox,
                    smoothed_score=decayed,
                    mask=prev.mask,
                ))
            # else: expired — silently drop.

        self._prev = next_prev
        return out

    def reset(self) -> None:
        # Keep _next_id monotonically increasing across resets so tracks
        # logged before/after a scene cut never collide.
        self._prev = []

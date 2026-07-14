"""IoU-based instance tracker with confirmation gating, hold, and EMA smoothing.

Three stabilisation layers on top of (greedy or Hungarian) IoU matching:
  Confirm — a new track must match for ``min_hits`` STRICTLY CONSECUTIVE
            frames before it is ever emitted; any miss during this tentative
            window drops it immediately (no hold grace period — that is a
            privilege earned only after confirmation). Suppresses one-off
            false-positive flicker at its source, before it ever reaches the
            renderer.
  Hold    — re-emit a missed CONFIRMED track for up to ``max_hold_frames``
            frames with a decaying score, preventing brief blink-outs.
  EMA     — exponentially blend bbox coordinates and confidence each matched
            frame, eliminating pixel-level jitter from detector noise.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..core.geometry import iou_xyxy
from ..core.types import Detection
from .base import InstanceTracker

logger = logging.getLogger(__name__)

# Sentinel cost for pairs that must never be matched (different class, or
# IoU below threshold) — large enough that the solver only picks such a pair
# when no valid alternative exists, at which point the iou_threshold cut
# below discards it anyway.
_NO_MATCH_COST = 10.0


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
    hit_count: int = 1        # consecutive successful matches since creation
    confirmed: bool = True    # becomes permanently True once hit_count >= min_hits


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
        use_hungarian_matching: bool = False,
        min_hits: int = 1,
    ) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold}")
        if min_hits < 1:
            raise ValueError(f"min_hits must be >= 1, got {min_hits}")
        self._iou_threshold = float(iou_threshold)
        self._max_hold = int(max_hold_frames)
        self._hold_decay = float(hold_score_decay)
        self._bbox_alpha = float(bbox_alpha)
        self._score_alpha = float(score_alpha)
        self._use_hungarian = bool(use_hungarian_matching)
        self._min_hits = int(min_hits)
        self._next_id = 1
        self._prev: list[_PrevDet] = []

    # ------------------------------------------------------------------ #
    def _match_greedy(
        self, detections: Sequence[Detection],
        candidate_indices: set[int] | None = None,
    ) -> dict[int, int]:
        """detection index -> prev index; best-IoU-first, first come first served.

        ``candidate_indices``: restrict eligible ``self._prev`` entries to this
        set (used by the low-confidence-recovery sub-step); ``None`` = all.
        """
        used_prev: set[int] = set()
        matches: dict[int, int] = {}
        for d_idx, det in enumerate(detections):
            best_idx, best_iou = -1, 0.0
            for i, prev in enumerate(self._prev):
                if i in used_prev or prev.class_name != det.class_name:
                    continue
                if candidate_indices is not None and i not in candidate_indices:
                    continue
                iou = iou_xyxy(det.bbox_xyxy, prev.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_idx >= 0 and best_iou >= self._iou_threshold:
                used_prev.add(best_idx)
                matches[d_idx] = best_idx
        return matches

    def _match_hungarian(
        self, detections: Sequence[Detection],
        candidate_indices: set[int] | None = None,
    ) -> dict[int, int]:
        """detection index -> prev index; globally-optimal one-to-one assignment.

        ``candidate_indices``: restrict eligible ``self._prev`` entries to this
        set (used by the low-confidence-recovery sub-step); ``None`` = all.
        """
        n_det, n_prev = len(detections), len(self._prev)
        if n_det == 0 or n_prev == 0:
            return {}
        cost = np.full((n_det, n_prev), _NO_MATCH_COST, dtype=np.float64)
        for d_idx, det in enumerate(detections):
            for i, prev in enumerate(self._prev):
                if prev.class_name != det.class_name:
                    continue
                if candidate_indices is not None and i not in candidate_indices:
                    continue
                iou = iou_xyxy(det.bbox_xyxy, prev.bbox)
                if iou >= self._iou_threshold:
                    cost[d_idx, i] = 1.0 - iou
        row_ind, col_ind = linear_sum_assignment(cost)
        return {
            int(d_idx): int(i)
            for d_idx, i in zip(row_ind, col_ind)
            if cost[d_idx, i] < _NO_MATCH_COST
        }

    def _match(
        self, detections: Sequence[Detection],
        candidate_indices: set[int] | None = None,
    ) -> dict[int, int]:
        return (self._match_hungarian(detections, candidate_indices) if self._use_hungarian
               else self._match_greedy(detections, candidate_indices))

    def update(
        self,
        frame_bgr: np.ndarray,
        detections: Sequence[Detection],
    ) -> list[Detection]:
        out: list[Detection] = []
        used_prev: set[int] = set()

        # Sub-threshold detections (display_threshold set and score below it)
        # are recovery-only: eligible to extend an already-confirmed track in
        # Step 1.5, never to create or confirm one in Step 1. When recovery
        # is disabled upstream, every detection has display_threshold=None or
        # already satisfies it, so `recovery` is always empty here — Step 1.5
        # never runs and behavior is identical to Stages 1-2.
        confirmable = [d for d in detections
                      if d.display_threshold is None or d.score >= d.display_threshold]
        recovery = [d for d in detections
                   if d.display_threshold is not None and d.score < d.display_threshold]

        # ── Step 1: match each new detection against previous tracks ─────────
        matches = self._match(confirmable)
        for d_idx, det in enumerate(confirmable):
            best_idx = matches.get(d_idx, -1)

            if best_idx >= 0:
                # Matched: EMA-blend bbox and score.
                prev = self._prev[best_idx]
                used_prev.add(best_idx)

                sb = _blend_bbox(det.bbox_xyxy, prev.smoothed_bbox, self._bbox_alpha)
                ss = self._score_alpha * det.score + (1.0 - self._score_alpha) * prev.smoothed_score
                emit_bbox = _bbox_to_int(sb)

                new_hits = prev.hit_count + 1
                now_confirmed = prev.confirmed or new_hits >= self._min_hits

                self._prev[best_idx] = _PrevDet(
                    class_name=det.class_name,
                    bbox=emit_bbox,
                    track_id=prev.track_id,
                    missed_count=0,
                    smoothed_bbox=sb,
                    smoothed_score=ss,
                    mask=det.mask,
                    hit_count=new_hits,
                    confirmed=now_confirmed,
                )
                if now_confirmed:
                    out.append(Detection(
                        class_name=det.class_name,
                        score=float(ss),
                        bbox_xyxy=emit_bbox,
                        mask=det.mask,
                        track_id=prev.track_id,
                    ))
            else:
                # New track: raw values pass through unchanged.
                raw_f = tuple(float(v) for v in det.bbox_xyxy)
                tid = self._next_id
                self._next_id += 1
                is_confirmed = self._min_hits <= 1
                if is_confirmed:
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
                    hit_count=1,
                    confirmed=is_confirmed,
                ))
                used_prev.add(len(self._prev) - 1)

        # ── Step 1.5: low-confidence recovery for already-confirmed tracks ───
        # A last chance, before falling back to hold, for CONFIRMED tracks
        # left unmatched to pick up a sub-threshold detection instead —
        # showing the object's real current position rather than a frozen/
        # decayed guess. Never creates or (re)confirms a track: hit_count is
        # untouched and unconfirmed tracks are not eligible.
        if recovery:
            recoverable = {i for i in range(len(self._prev))
                          if i not in used_prev and self._prev[i].confirmed}
            if recoverable:
                recov_matches = self._match(recovery, recoverable)
                for r_idx, prev_idx in recov_matches.items():
                    det = recovery[r_idx]
                    prev = self._prev[prev_idx]
                    sb = _blend_bbox(det.bbox_xyxy, prev.smoothed_bbox, self._bbox_alpha)
                    ss = self._score_alpha * det.score + (1.0 - self._score_alpha) * prev.smoothed_score
                    emit_bbox = _bbox_to_int(sb)
                    logger.debug(
                        "Track %d recovered via sub-threshold detection score=%.2f "
                        "(display_threshold=%.2f)",
                        prev.track_id, det.score, det.display_threshold,
                    )
                    out.append(Detection(
                        class_name=det.class_name,
                        score=float(ss),
                        bbox_xyxy=emit_bbox,
                        mask=det.mask,
                        track_id=prev.track_id,
                    ))
                    self._prev[prev_idx] = _PrevDet(
                        class_name=det.class_name,
                        bbox=emit_bbox,
                        track_id=prev.track_id,
                        missed_count=0,
                        smoothed_bbox=sb,
                        smoothed_score=ss,
                        mask=det.mask,
                        hit_count=prev.hit_count,
                        confirmed=True,
                    )
                    used_prev.add(prev_idx)

        # ── Step 2: hold or expire unmatched previous tracks ─────────────────
        next_prev: list[_PrevDet] = []
        for i, prev in enumerate(self._prev):
            if i in used_prev:
                # Already updated in step 1 (and re-appended inline).
                # Collect only the entries we wrote back.
                next_prev.append(self._prev[i])
                continue

            if not prev.confirmed:
                # Tentative track missed a frame before ever being displayed.
                # No hold grace period for it — hold is a privilege earned
                # only after confirmation. Drop immediately; must restart a
                # fresh, fully consecutive confirmation window to try again.
                logger.debug(
                    "Track %d expired before confirming (%d/%d consecutive hits)",
                    prev.track_id, prev.hit_count, self._min_hits,
                )
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
                    hit_count=prev.hit_count,
                    confirmed=prev.confirmed,
                ))
            # else: expired — silently drop.

        self._prev = next_prev
        return out

    def reset(self) -> None:
        # Keep _next_id monotonically increasing across resets so tracks
        # logged before/after a scene cut never collide.
        self._prev = []

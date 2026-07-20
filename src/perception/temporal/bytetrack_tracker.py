"""ByteTrack-backed InstanceTracker, wrapping roboflow/trackers' ByteTrackTracker.

Why not use the library directly, unwrapped
--------------------------------------------
Three gaps between the published implementation and this project's needs,
each verified by reading trackers==2.5.0.post0's actual source (not assumed
from docs/blog posts, which describe an older/different API shape):

1. **Class-agnostic matching.** ``ByteTrackTracker.update()`` never reads
   ``detections.class_id`` — it matches purely on box IoU. Feeding it all
   classes in one call would let e.g. a person box hijack a tank's track
   when their boxes happen to overlap. Fixed here by running one tracker
   instance *per class name*, splitting/merging every frame.

2. **Per-instance ID allocation.** ``BaseTracker._allocate_tracker_id()``
   starts each instance's counter at 0 — running one instance per class
   (fix #1) would then produce colliding IDs across classes. Fixed here by
   remapping every (class_name, local_id) pair to a single global counter.

3. **Global vs. per-class confidence threshold.** This project's model
   wrappers already gate detections per-class *before* they reach the
   tracker (``Detection.display_threshold`` — different per class, e.g.
   "mil vehicle" 0.55 vs "person" 0.35), matching the existing
   ``low_conf_recovery`` semantics: a sub-threshold detection may extend an
   already-confirmed track but never create or confirm one.
   ``ByteTrackTracker``'s own high/low confidence split
   (``high_conf_det_threshold`` / ``track_activation_threshold``) is a
   *single global* value, which can't reproduce varying per-class
   thresholds. Rather than re-deriving (and duplicating) the per-class
   decision inside this adapter, the already-gated
   ``display_threshold``/``score`` comparison upstream is reused directly:
   detections that clear their own class's threshold are fed to ByteTrack
   at confidence 1.0 (eligible to match AND spawn new tracks); detections
   below it (recovery-only) are fed at confidence 0.0 (eligible only to
   extend an existing track). ``high_conf_det_threshold`` and
   ``track_activation_threshold`` are both set to 0.5 — the split point
   between those two synthetic values — so ByteTrack's internal tiering
   exactly mirrors the decision this project already made per-class,
   instead of re-deciding it from a single global number.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Sequence

import numpy as np

from ..core.types import Detection
from .base import InstanceTracker

logger = logging.getLogger(__name__)

# Synthetic confidence values fed to ByteTrackTracker in place of the
# detection's real score — see module docstring, point 3.
_CONFIRMABLE_CONF = 1.0
_RECOVERY_CONF = 0.0
_CONF_SPLIT = 0.5


class ByteTrackInstanceTracker(InstanceTracker):
    """One roboflow/trackers ByteTrackTracker per class name, wrapped to
    satisfy the InstanceTracker ABC (list[Detection] in, list[Detection] out)."""

    def __init__(
        self,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        minimum_consecutive_frames: int = 2,
        minimum_iou_threshold: float = 0.1,
        hold_score_decay: float = 0.85,
    ) -> None:
        # +1 to compensate for an off-by-one between the two trackers'
        # expiry semantics, confirmed empirically: ByteTrackTracker expires a
        # track the frame its miss-count REACHES lost_track_buffer (so
        # lost_track_buffer=N holds for only N-1 missed frames), while
        # IoUInstanceTracker's max_hold_frames=N holds for N full missed
        # frames. Without this, passing the "same" configured number (as
        # temporal/factory.py does) makes ByteTrack expire held tracks one
        # frame earlier than IoU for the same nominal setting — this was
        # caught by comparing the same real footage side by side (a held
        # detection visible in the "iou" panel one frame after it had
        # already vanished from "bytetrack").
        self._lost_track_buffer = int(lost_track_buffer) + 1
        self._frame_rate = float(frame_rate)
        self._minimum_consecutive_frames = int(minimum_consecutive_frames)
        self._minimum_iou_threshold = float(minimum_iou_threshold)
        self._hold_score_decay = float(hold_score_decay)

        self._trackers: dict[str, object] = {}  # class_name -> ByteTrackTracker
        self._next_global_id = 1
        self._local_to_global: dict[tuple[str, int], int] = {}
        # ByteTrackTracker.tracked_objects exposes Kalman-predicted boxes for
        # held tracks but no score (confirmed by reading its source: "Kalman-
        # predicted boxes have no associated detection score or class
        # label") — this project's renderer needs one, so the last real
        # score is remembered here and decayed each frame a track is held,
        # mirroring IoUInstanceTracker's hold_score_decay.
        self._last_score: dict[tuple[str, int], float] = {}

    # ------------------------------------------------------------------ #

    def _make_tracker(self):
        from trackers import ByteTrackTracker

        return ByteTrackTracker(
            lost_track_buffer=self._lost_track_buffer,
            frame_rate=self._frame_rate,
            track_activation_threshold=_CONF_SPLIT,
            minimum_consecutive_frames=self._minimum_consecutive_frames,
            minimum_iou_threshold=self._minimum_iou_threshold,
            high_conf_det_threshold=_CONF_SPLIT,
        )

    def _get_tracker(self, class_name: str):
        tracker = self._trackers.get(class_name)
        if tracker is None:
            tracker = self._make_tracker()
            self._trackers[class_name] = tracker
        return tracker

    def _global_id(self, class_name: str, local_id: int) -> int | None:
        """local_id == -1 means "not yet confirmed" (ByteTrack's own
        min-consecutive-frames gate hasn't been met) -> no track_id, matching
        this project's existing convention of only emitting confirmed tracks."""
        if local_id < 0:
            return None
        key = (class_name, local_id)
        gid = self._local_to_global.get(key)
        if gid is None:
            gid = self._next_global_id
            self._next_global_id += 1
            self._local_to_global[key] = gid
        return gid

    # ------------------------------------------------------------------ #

    def update(
        self,
        frame_bgr: np.ndarray,
        detections: Sequence[Detection],
    ) -> list[Detection]:
        import supervision as sv

        by_class: dict[str, list[Detection]] = defaultdict(list)
        for d in detections:
            by_class[d.class_name].append(d)

        # Every class with a live tracker must still be updated this frame
        # (with an empty detection set if none arrived) so its existing
        # tracks get a Kalman prediction and their hold-buffer clock ticks —
        # otherwise a class that's briefly undetected would never expire.
        classes_to_update = set(self._trackers) | set(by_class)

        out: list[Detection] = []
        for class_name in classes_to_update:
            class_dets = by_class.get(class_name, [])
            tracker = self._get_tracker(class_name)

            if class_dets:
                xyxy = np.array([d.bbox_xyxy for d in class_dets], dtype=float)
                synthetic_conf = np.array([
                    _CONFIRMABLE_CONF
                    if d.display_threshold is None or d.score >= d.display_threshold
                    else _RECOVERY_CONF
                    for d in class_dets
                ])
                sv_dets = sv.Detections(
                    xyxy=xyxy,
                    confidence=synthetic_conf,
                    data={"orig_idx": np.arange(len(class_dets))},
                )
            else:
                sv_dets = sv.Detections.empty()
                sv_dets.data["orig_idx"] = np.array([], dtype=int)

            tracked = tracker.update(sv_dets)
            matched_local_ids = set()
            for i in range(len(tracked)):
                local_id = int(tracked.tracker_id[i])
                orig = class_dets[int(tracked.data["orig_idx"][i])]
                self._last_score[(class_name, local_id)] = orig.score
                gid = self._global_id(class_name, local_id)
                if gid is None:
                    continue  # not yet confirmed — never emitted, same as today
                matched_local_ids.add(local_id)
                box = tuple(int(round(v)) for v in tracked.xyxy[i])
                out.append(Detection(
                    class_name=class_name,
                    score=orig.score,
                    bbox_xyxy=box,  # type: ignore[arg-type]
                    mask=orig.mask,
                    track_id=gid,
                ))

            # Confirmed tracks alive but not matched to any detection this
            # frame (held via Kalman prediction) — this project's "hold".
            held = tracker.tracked_objects
            for i in range(len(held)):
                local_id = int(held.tracker_id[i])
                if local_id in matched_local_ids:
                    continue
                gid = self._global_id(class_name, local_id)
                if gid is None:
                    continue
                key = (class_name, local_id)
                decayed = self._last_score.get(key, 1.0) * self._hold_score_decay
                self._last_score[key] = decayed
                box = tuple(int(round(v)) for v in held.xyxy[i])
                out.append(Detection(
                    class_name=class_name,
                    score=float(decayed),
                    bbox_xyxy=box,  # type: ignore[arg-type]
                    mask=None,
                    track_id=gid,
                ))

            # Prune bookkeeping for local ids the underlying tracker has
            # fully dropped (expired past lost_track_buffer) — otherwise
            # _last_score/_local_to_global grow unboundedly over a
            # long-running stream, since ByteTrackTracker prunes its own
            # dead tracklets internally but has no way to tell us to.
            still_alive = matched_local_ids | set(int(v) for v in held.tracker_id)
            for key in [k for k in self._last_score if k[0] == class_name and k[1] not in still_alive]:
                del self._last_score[key]
            for key in [k for k in self._local_to_global if k[0] == class_name and k[1] not in still_alive]:
                del self._local_to_global[key]

        return out

    def reset(self) -> None:
        self._trackers = {}
        self._local_to_global = {}
        self._next_global_id = 1
        self._last_score = {}

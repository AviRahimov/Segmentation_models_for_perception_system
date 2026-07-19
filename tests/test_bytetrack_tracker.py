"""Tests for ByteTrackInstanceTracker — the roboflow/trackers-backed InstanceTracker.

Covers the three adaptation gaps documented in bytetrack_tracker.py's module
docstring (class-agnostic matching, per-instance ID allocation, global vs
per-class confidence threshold), plus the hold/expire/reset lifecycle that
must match this project's existing InstanceTracker semantics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src")))

from perception.core.types import Detection  # noqa: E402
from perception.temporal.bytetrack_tracker import ByteTrackInstanceTracker  # noqa: E402

_FRAME = np.zeros((20, 20, 3), dtype=np.uint8)


def _det(box: tuple[int, int, int, int], score: float = 0.9,
        cls: str = "vehicle", threshold: float | None = None) -> Detection:
    return Detection(class_name=cls, score=score, bbox_xyxy=box, mask=None,
                     display_threshold=threshold)


def _tracker(**kw) -> ByteTrackInstanceTracker:
    # A brand-new track always spawns with tracker_id=-1 on its first frame
    # (confirmed by reading ByteTrackTracker.update()'s source: spawning and
    # matching are different code paths) — so even minimum_consecutive_frames=1
    # needs a *second* matched frame before it's ever emitted. Tests that only
    # care about steady-state identity feed one extra warm-up frame first.
    kw.setdefault("minimum_consecutive_frames", 1)
    return ByteTrackInstanceTracker(**kw)


def _confirm(t: ByteTrackInstanceTracker, box: tuple[int, int, int, int], **kw) -> list[Detection]:
    """Feed two matching frames so the resulting track is confirmed (has a
    real, non-negative track_id) — see _tracker()'s docstring note."""
    t.update(_FRAME, [_det(box, **kw)])
    return t.update(_FRAME, [_det(box, **kw)])


# --------------------------------------------------------------------------- #
# Basic identity persistence                                                   #
# --------------------------------------------------------------------------- #

def test_single_track_persists_id_across_frames():
    t = _tracker(minimum_iou_threshold=0.1)
    t.update(_FRAME, [_det((0, 0, 10, 10))])              # spawn (unconfirmed)
    out2 = t.update(_FRAME, [_det((1, 1, 11, 11))])       # confirmed
    out3 = t.update(_FRAME, [_det((2, 2, 12, 12))])       # small shift, same object
    assert out2 and out3
    assert out2[0].track_id == out3[0].track_id


def test_class_agnostic_library_still_kept_class_conditioned():
    """ByteTrackTracker itself never reads class_id (verified by reading its
    source) — this adapter's own per-class-instance split must still prevent
    a person box from ever adopting a vehicle's track id.

    (out_p's update() call also re-emits the still-alive, held "vehicle"
    track from out_v's tracker in the same frame — dict/set iteration order
    over classes_to_update is unspecified, so out_p[0] is not reliably the
    "person" entry; filter by class_name instead of indexing.)"""
    t = _tracker(minimum_iou_threshold=0.1)
    out_v = _confirm(t, (0, 0, 10, 10), cls="vehicle")
    vehicle_id = out_v[0].track_id
    out_p = _confirm(t, (0, 0, 10, 10), cls="person")
    person_id = next(d.track_id for d in out_p if d.class_name == "person")
    assert vehicle_id != person_id


def test_ids_unique_across_classes_despite_per_instance_allocation():
    """Each class gets its own ByteTrackTracker instance, whose internal id
    counter independently starts at 0 — global ids must still never collide.

    (out_p also carries the still-alive, held "vehicle" track from
    out_v's tracker — default lost_track_buffer=30 keeps it around across
    these few frames — so only the two classes' *own* new ids are compared,
    not every id in the combined output.)"""
    t = _tracker(minimum_iou_threshold=0.1)
    out_v = _confirm(t, (0, 0, 10, 10), cls="vehicle")
    vehicle_id = out_v[0].track_id
    out_p = _confirm(t, (100, 100, 110, 110), cls="person")
    person_id = next(d.track_id for d in out_p if d.class_name == "person")
    assert vehicle_id != person_id


# --------------------------------------------------------------------------- #
# min_hits / confirmation gating                                               #
# --------------------------------------------------------------------------- #

def test_min_hits_gates_first_appearance():
    t = _tracker(minimum_consecutive_frames=2, minimum_iou_threshold=0.1)
    out1 = t.update(_FRAME, [_det((0, 0, 10, 10))])
    assert out1 == []  # not yet confirmed
    out2 = t.update(_FRAME, [_det((1, 1, 11, 11))])
    assert len(out2) == 1  # confirmed on 2nd consecutive match


# --------------------------------------------------------------------------- #
# Per-class threshold / low-confidence recovery semantics                     #
# --------------------------------------------------------------------------- #

def test_sub_threshold_detection_cannot_spawn_a_track():
    t = _tracker(minimum_iou_threshold=0.1)
    # score below its own display_threshold -> recovery-only, must not confirm alone
    out = t.update(_FRAME, [_det((0, 0, 10, 10), score=0.2, threshold=0.5)])
    assert out == []


def test_sub_threshold_detection_can_extend_a_confirmed_track():
    t = _tracker(minimum_iou_threshold=0.1)
    out1 = _confirm(t, (0, 0, 10, 10), score=0.9, threshold=0.5)
    tid = out1[0].track_id
    # Next frame: same object, but only a low-confidence (recovery-only) detection.
    out2 = t.update(_FRAME, [_det((1, 1, 11, 11), score=0.2, threshold=0.5)])
    assert len(out2) == 1
    assert out2[0].track_id == tid


# --------------------------------------------------------------------------- #
# Hold (missed detection) and expiry                                          #
# --------------------------------------------------------------------------- #

def test_hold_decays_score_then_expires():
    # lost_track_buffer=2 (frame_rate=30 default -> maximum_frames_without_update=2):
    # a track survives exactly one missed frame (held) before the second miss
    # prunes it — verified empirically against the installed trackers package,
    # since buffer=1 would expire on the very first miss with no hold at all.
    t = _tracker(minimum_iou_threshold=0.1, lost_track_buffer=2, hold_score_decay=0.5)
    out1 = _confirm(t, (0, 0, 10, 10), score=0.8)
    tid = out1[0].track_id
    out2 = t.update(_FRAME, [])  # missed once -> held via Kalman prediction
    assert len(out2) == 1
    assert out2[0].track_id == tid
    assert out2[0].score < 0.8
    out3 = t.update(_FRAME, [])  # missed twice -> lost_track_buffer exhausted, expired
    assert out3 == []


# --------------------------------------------------------------------------- #
# Empty input / reset                                                          #
# --------------------------------------------------------------------------- #

def test_empty_detections_returns_empty_list():
    t = _tracker()
    assert t.update(_FRAME, []) == []


def test_reset_clears_all_state_and_reassigns_from_scratch():
    t = _tracker(minimum_iou_threshold=0.1)
    out1 = _confirm(t, (0, 0, 10, 10))
    first_id = out1[0].track_id
    t.reset()
    out2 = _confirm(t, (0, 0, 10, 10))
    assert out2[0].track_id == first_id  # counters restart identically post-reset

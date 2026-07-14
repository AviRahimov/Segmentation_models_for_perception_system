"""Tests for IoUInstanceTracker — greedy (default) and Hungarian matching.

Stage 1 of the tracker upgrade: verifies the Hungarian assignment path is
correct and genuinely diverges from greedy in an ambiguous case, and that the
default (use_hungarian_matching=False) reproduces the original greedy
behavior exactly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from perception.core.types import Detection  # noqa: E402
from perception.temporal.iou_tracker import IoUInstanceTracker  # noqa: E402

_FRAME = np.zeros((20, 20, 3), dtype=np.uint8)


def _det(box: tuple[int, int, int, int], score: float = 0.9,
        cls: str = "vehicle") -> Detection:
    return Detection(class_name=cls, score=score, bbox_xyxy=box, mask=None)


# --------------------------------------------------------------------------- #
# Basic regression — default config, single object                            #
# --------------------------------------------------------------------------- #

def test_default_constructor_is_greedy():
    t = IoUInstanceTracker()
    assert t._use_hungarian is False  # explicit default, not accidental


def test_single_track_persists_id_across_frames():
    t = IoUInstanceTracker(iou_threshold=0.3)
    out1 = t.update(_FRAME, [_det((0, 0, 10, 10))])
    out2 = t.update(_FRAME, [_det((1, 1, 11, 11))])  # small shift, same object
    assert out1[0].track_id == out2[0].track_id


def test_class_conditioned_matching_never_crosses_classes():
    t = IoUInstanceTracker(iou_threshold=0.1)
    t.update(_FRAME, [_det((0, 0, 10, 10), cls="vehicle")])
    out = t.update(_FRAME, [_det((0, 0, 10, 10), cls="person")])
    # Identical box, different class -> must NOT match the vehicle track;
    # gets a fresh id instead.
    assert out[0].track_id == 2


def test_hold_then_expire_unchanged_by_refactor():
    t = IoUInstanceTracker(iou_threshold=0.3, max_hold_frames=1, hold_score_decay=0.5)
    out1 = t.update(_FRAME, [_det((0, 0, 10, 10), score=0.8)])
    tid = out1[0].track_id
    # Frame 2: object missing -> held with decayed score.
    out2 = t.update(_FRAME, [])
    assert len(out2) == 1 and out2[0].track_id == tid
    assert out2[0].score < 0.8
    # Frame 3: still missing, max_hold_frames=1 exhausted -> expired.
    out3 = t.update(_FRAME, [])
    assert out3 == []


# --------------------------------------------------------------------------- #
# Greedy vs Hungarian — verified divergence scenario                          #
# --------------------------------------------------------------------------- #
# Geometry (confirmed against the real iou_xyxy, threshold=0.3):
#   A=(0,0,10,10) track#1, B=(3,0,13,10) track#2
#   D1=(0,0,10,10): IoU(D1,A)=1.000 IoU(D1,B)=0.538   (both feasible)
#   D2=(-5,0,5,10): IoU(D2,A)=0.333 IoU(D2,B)=0.111    (only A feasible)
# Greedy (D1 processed first) grabs D1->A (best local choice), leaving D2 with
# only the infeasible B -> D2 becomes a NEW track. Hungarian finds the lower
# total-cost assignment D1->B / D2->A, so BOTH keep their prior identity.

_A, _B = (0, 0, 10, 10), (3, 0, 13, 10)
_D1, _D2 = (0, 0, 10, 10), (-5, 0, 5, 10)


def _seed_two_tracks(tracker: IoUInstanceTracker) -> tuple[int, int]:
    out = tracker.update(_FRAME, [_det(_A), _det(_B)])
    ids = {d.bbox_xyxy: d.track_id for d in out}
    return ids[_A], ids[_B]


def test_greedy_leaves_one_detection_unmatched():
    t = IoUInstanceTracker(iou_threshold=0.3, max_hold_frames=0,
                           use_hungarian_matching=False)
    id_a, id_b = _seed_two_tracks(t)
    out = t.update(_FRAME, [_det(_D1), _det(_D2)])
    assert len(out) == 2
    d1_result, d2_result = out
    assert d1_result.track_id == id_a       # D1 grabbed its best match, A
    assert d2_result.track_id not in (id_a, id_b)  # D2 got a fresh id


def test_hungarian_finds_globally_optimal_swap():
    t = IoUInstanceTracker(iou_threshold=0.3, max_hold_frames=0,
                           use_hungarian_matching=True)
    id_a, id_b = _seed_two_tracks(t)
    out = t.update(_FRAME, [_det(_D1), _det(_D2)])
    assert len(out) == 2
    d1_result, d2_result = out
    # The lower-total-cost assignment: D1<->B, D2<->A. Both keep an identity.
    assert d1_result.track_id == id_b
    assert d2_result.track_id == id_a


def test_hungarian_respects_class_conditioning():
    """A same-geometry different-class pair must never be matched, even
    though Hungarian would otherwise consider it in the global optimum."""
    t = IoUInstanceTracker(iou_threshold=0.3, max_hold_frames=0,
                           use_hungarian_matching=True)
    t.update(_FRAME, [_det((0, 0, 10, 10), cls="vehicle")])
    out = t.update(_FRAME, [_det((0, 0, 10, 10), cls="person")])
    assert out[0].track_id == 2  # fresh id, not the vehicle track's id


def test_hungarian_matches_when_unambiguous():
    """Sanity check: in the common, unambiguous case (each detection has a
    clear unique best partner), Hungarian and greedy must agree."""
    t = IoUInstanceTracker(iou_threshold=0.3, max_hold_frames=0,
                           use_hungarian_matching=True)
    id_a, id_b = _seed_two_tracks(t)
    # Small shifts that keep each detection closest to its own prior track.
    out = t.update(_FRAME, [_det((0, 0, 10, 10)), _det((4, 0, 14, 10))])
    ids = [d.track_id for d in out]
    assert set(ids) == {id_a, id_b}


# --------------------------------------------------------------------------- #
# Stage 2 — min_hits track confirmation                                       #
# --------------------------------------------------------------------------- #

def test_min_hits_default_of_one_emits_immediately():
    t = IoUInstanceTracker(iou_threshold=0.3)
    out = t.update(_FRAME, [_det((0, 0, 10, 10))])
    assert len(out) == 1  # today's behavior: no gating


def test_min_hits_two_suppresses_first_frame():
    t = IoUInstanceTracker(iou_threshold=0.3, min_hits=2)
    out1 = t.update(_FRAME, [_det((0, 0, 10, 10))])
    assert out1 == []  # tentative — not yet confirmed, nothing displayed


def test_min_hits_two_emits_from_second_consecutive_match():
    t = IoUInstanceTracker(iou_threshold=0.3, min_hits=2)
    out1 = t.update(_FRAME, [_det((0, 0, 10, 10))])
    out2 = t.update(_FRAME, [_det((1, 0, 11, 10))])  # same object, small shift
    assert out1 == []
    assert len(out2) == 1
    tid = out2[0].track_id
    # Stays confirmed afterward — a later hit doesn't need to re-earn anything.
    out3 = t.update(_FRAME, [_det((2, 0, 12, 10))])
    assert len(out3) == 1 and out3[0].track_id == tid


def test_min_hits_gap_before_confirmation_resets_and_deletes():
    t = IoUInstanceTracker(iou_threshold=0.3, min_hits=3, max_hold_frames=5)
    out1 = t.update(_FRAME, [_det((0, 0, 10, 10))])   # hit 1/3
    out2 = t.update(_FRAME, [])                        # miss before confirming
    out3 = t.update(_FRAME, [_det((0, 0, 10, 10))])    # reappears
    assert out1 == [] and out2 == []
    # No hold grace period pre-confirmation, even though max_hold_frames=5:
    # the reappearance starts a brand-new tentative track (hit 1/3 again),
    # not a continuation of the deleted one.
    assert out3 == []


def test_min_hits_does_not_affect_hold_of_already_confirmed_tracks():
    """A CONFIRMED track's existing hold/decay behavior must be unaffected
    by min_hits — hold is a privilege only unconfirmed tracks are denied."""
    t = IoUInstanceTracker(iou_threshold=0.3, min_hits=2, max_hold_frames=2,
                           hold_score_decay=0.5)
    t.update(_FRAME, [_det((0, 0, 10, 10), score=0.8)])       # hit 1/2
    out2 = t.update(_FRAME, [_det((0, 0, 10, 10), score=0.8)])  # hit 2/2 -> confirmed
    tid = out2[0].track_id
    out3 = t.update(_FRAME, [])  # miss AFTER confirmation -> held, not deleted
    assert len(out3) == 1
    assert out3[0].track_id == tid
    assert out3[0].score < 0.8


def test_min_hits_one_is_bit_identical_to_pre_stage2_behavior():
    """Regression: default min_hits=1 must never suppress or delay anything,
    matching every pre-existing behavior test above."""
    t = IoUInstanceTracker(iou_threshold=0.3, max_hold_frames=1, hold_score_decay=0.5)
    out1 = t.update(_FRAME, [_det((0, 0, 10, 10), score=0.8)])
    tid = out1[0].track_id
    out2 = t.update(_FRAME, [])
    assert len(out2) == 1 and out2[0].track_id == tid and out2[0].score < 0.8
    out3 = t.update(_FRAME, [])
    assert out3 == []


# --------------------------------------------------------------------------- #
# Stage 3 — low-confidence recovery for already-confirmed tracks             #
# --------------------------------------------------------------------------- #

def _recovery_det(box, score, threshold, cls: str = "vehicle") -> Detection:
    return Detection(class_name=cls, score=score, bbox_xyxy=box, mask=None,
                     display_threshold=threshold)


def test_recovery_disabled_by_default_is_a_pure_regression():
    """Every detection lacking display_threshold (the legacy/disabled case)
    must behave EXACTLY like Stage 1+2 — recovery list is always empty."""
    t = IoUInstanceTracker(iou_threshold=0.3, max_hold_frames=1, hold_score_decay=0.5)
    out1 = t.update(_FRAME, [_det((0, 0, 10, 10), score=0.8)])
    tid = out1[0].track_id
    out2 = t.update(_FRAME, [])  # miss -> held, exactly as before
    assert len(out2) == 1 and out2[0].track_id == tid and out2[0].score < 0.8


def test_confirmed_track_recovers_via_subthreshold_detection():
    t = IoUInstanceTracker(iou_threshold=0.3, max_hold_frames=3, hold_score_decay=0.5)
    # Confirm the track first (min_hits=1 default -> confirmed immediately).
    out1 = t.update(_FRAME, [_det((0, 0, 10, 10), score=0.8)])
    tid = out1[0].track_id
    # A sub-threshold (recovery-only) detection at a NEW real position.
    out2 = t.update(_FRAME, [_recovery_det((5, 0, 15, 10), score=0.2, threshold=0.5)])
    assert len(out2) == 1
    assert out2[0].track_id == tid
    # Recovered using the REAL position (shifted to x=5), not a frozen hold.
    assert out2[0].bbox_xyxy[0] > 0


def test_recovery_uses_real_position_not_frozen_hold_position():
    """The whole point of recovery: the box should track the object's actual
    (shifted) location, unlike hold which freezes the last known position."""
    t = IoUInstanceTracker(iou_threshold=0.2, max_hold_frames=3,
                           bbox_alpha=1.0)  # alpha=1 -> emit exact raw position
    t.update(_FRAME, [_det((0, 0, 10, 10), score=0.8)])
    # Overlaps enough to match (IoU=0.333 >= 0.2) but has clearly moved.
    out_recovered = t.update(_FRAME, [_recovery_det((5, 0, 15, 10), score=0.2, threshold=0.5)])
    assert out_recovered[0].bbox_xyxy[0] == 5  # tracks the shifted real box


def test_unconfirmed_track_never_recovers():
    """A tentative (not-yet-confirmed) track gets no recovery chance — it
    must still be deleted on any miss, per Stage 2 semantics."""
    t = IoUInstanceTracker(iou_threshold=0.3, min_hits=2, max_hold_frames=5)
    t.update(_FRAME, [_det((0, 0, 10, 10))])  # hit 1/2, NOT confirmed
    out = t.update(_FRAME, [_recovery_det((0, 0, 10, 10), score=0.2, threshold=0.5)])
    assert out == []  # recovery does not confirm, nor does it hold a tentative track
    # Confirm this via a subsequent frame: a real detection must start a
    # BRAND NEW tentative track (hit 1/2 again), proving the old one is gone.
    out2 = t.update(_FRAME, [_det((0, 0, 10, 10))])
    assert out2 == []  # still just 1/2, not 2/2 — it's a fresh track


def test_confirmable_detection_preferred_over_recovery_when_both_present():
    """If a real, above-threshold detection is available for a track, the
    confirmable path (Step 1) must claim it before recovery ever runs."""
    t = IoUInstanceTracker(iou_threshold=0.2, max_hold_frames=3)
    t.update(_FRAME, [_det((0, 0, 10, 10), score=0.8)])
    # Both a confirmable and a recovery-only detection this frame; only the
    # confirmable one should be used (single object, single output).
    out = t.update(_FRAME, [
        _det((1, 0, 11, 10), score=0.7),
        _recovery_det((1, 0, 11, 10), score=0.2, threshold=0.5),
    ])
    assert len(out) == 1
    assert out[0].score > 0.5  # came from the confirmable (real) detection


def test_recovery_does_not_bump_hit_count_or_reconfirm():
    """Recovery never touches hit_count — it only extends existing trust."""
    t = IoUInstanceTracker(iou_threshold=0.2, max_hold_frames=3)
    t.update(_FRAME, [_det((0, 0, 10, 10), score=0.8)])
    t.update(_FRAME, [_recovery_det((0, 0, 10, 10), score=0.2, threshold=0.5)])
    # Internal state check: hit_count must be unchanged (still 1) after a
    # recovery-only match, not incremented as a real hit would.
    assert t._prev[0].hit_count == 1
    assert t._prev[0].confirmed is True

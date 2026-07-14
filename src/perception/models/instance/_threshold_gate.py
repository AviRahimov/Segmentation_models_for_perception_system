"""Shared confidence-gating logic for closed-vocab and YOLOE detector wrappers.

A single pure function so both wrappers apply identical low-confidence-
recovery semantics instead of each re-implementing (and potentially
diverging on) the same threshold arithmetic.
"""
from __future__ import annotations


def gate_confidence(
    score: float,
    threshold: float,
    recovery_floor: float | None,
) -> bool:
    """True if a detection at ``score`` should be returned by predict() at all.

    ``recovery_floor is None`` (recovery disabled): legacy behavior — keep
    only detections at or above the class's own ``threshold``.

    ``recovery_floor`` set: keep anything at or above the LOWER of
    ``threshold`` and ``recovery_floor`` — the extra boxes this admits (below
    ``threshold`` but above ``recovery_floor``) are recovery-only candidates;
    the caller attaches ``display_threshold=threshold`` so the tracker can
    tell them apart from fully-confirmable detections.
    """
    floor = threshold if recovery_floor is None else min(threshold, recovery_floor)
    return score >= floor

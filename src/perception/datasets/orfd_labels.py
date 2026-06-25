"""ORFD freespace PNG label semantics (`*_fillcolor.png` grayscale).

Empirically validated on files under ``datasets/orfd/training/gt_image/`` shipped
with the ORFD ZIP: **brightness encodes freespace**.

* ``255`` — **Traversable**. Visually reads as bright / near-white stripes on
  the drivable corridor.
* ``0`` — Non-traversable (obstacle / blocked), dark.
* ``128`` — Intermediate band (typically upper sky / ambiguous void in fisheye
  frames — **ignored** when computing freespace IoU so skies do not falsely
  match model predictions).

This differs from textual summaries that casually list ``128`` as freespace —
check ``docs/sanity_gt`` composites if your mirror swapped encodings.

For an escape hatch, see ``--orfd-trav-gray`` on
``scripts/orfd_semantic_comparison.py``.
"""
from __future__ import annotations

import numpy as np

# Default encoding for fillcolor PNG releases we tested.
_ORFD_NON_TRAVERSABLE: int = 0
_ORFD_AMBIGUOUS_OR_SKY_GRAY: int = 128  # excluded from freespace_binary IoU
_ORFD_TRAVERSABLE_BRIGHT: int = 255


# Public aliases (backward compatible names kept as module-level lookups).
def traversable_gray_value(orfd_trav_gray: int | None = None) -> int:
    """Canonical traversable-encoded gray (default bright path = 255)."""
    return int(orfd_trav_gray) if orfd_trav_gray is not None else _ORFD_TRAVERSABLE_BRIGHT


def orfd_traversable_gt_mask(
    gt_u8: np.ndarray,
    *,
    orfd_trav_gray: int | None = None,
) -> np.ndarray:
    """Boolean mask — pixels labelled traversable freespace."""
    if gt_u8.ndim == 3:
        gt_u8 = gt_u8[..., 0]
    gv = traversable_gray_value(orfd_trav_gray)
    return gt_u8 == gv


def orfd_eval_valid_mask(
    gt_u8: np.ndarray,
    *,
    orfd_trav_gray: int | None = None,
) -> np.ndarray:
    """Valid pixels for freespace IoU — only labelled trav vs anti-trav (exclude sky band).

    Exclude ``gray==128`` (ambiguous / sky corridor) regardless of traversable scalar.
    """
    if gt_u8.ndim == 3:
        gt_u8 = gt_u8[..., 0]
    trav = traversable_gray_value(orfd_trav_gray)
    other = {_ORFD_NON_TRAVERSABLE, trav}
    return np.isin(gt_u8, np.array(tuple(other), dtype=gt_u8.dtype))


def binary_traversable_iou(
    pred_trav: np.ndarray,
    gt_trav: np.ndarray,
    valid: np.ndarray,
) -> float | None:
    """IoU of *traversable* class on ``valid`` pixels only."""
    if pred_trav.shape != gt_trav.shape or pred_trav.shape != valid.shape:
        raise ValueError("pred_trav, gt_trav, valid must share the same shape")

    v = np.asarray(valid, dtype=bool)
    if not v.any():
        return None

    p = np.asarray(pred_trav, dtype=bool) & v
    g = np.asarray(gt_trav, dtype=bool) & v
    inter = np.logical_and(p, g).sum(dtype=np.float64)
    union = np.logical_or(p, g).sum(dtype=np.float64)
    if union <= 0:
        return None
    return float(inter / union)


# Deprecated module-level ints for callers using old naming (still 255 traversable default).
ORFD_NON_TRAVERSABLE_GRAY = _ORFD_NON_TRAVERSABLE
ORFD_SKY_BAND_GRAY = _ORFD_AMBIGUOUS_OR_SKY_GRAY
ORFD_TRAVERSABLE_GRAY = _ORFD_TRAVERSABLE_BRIGHT
ORFD_UNREACHABLE_GRAY = (
    ORFD_SKY_BAND_GRAY  # ambiguous band treated as unreachable for freespace overlay
)

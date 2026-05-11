"""Unit tests ORFD GT masking / freespace helpers (pure numpy, CPU-only)."""

from __future__ import annotations

import numpy as np

from perception.datasets.orfd_labels import (
    ORFD_NON_TRAVERSABLE_GRAY,
    ORFD_SKY_BAND_GRAY,
    ORFD_TRAVERSABLE_GRAY,
    binary_traversable_iou,
    orfd_eval_valid_mask,
    orfd_traversable_gt_mask,
)


def test_orfd_traversable_mask_default():
    gt = np.zeros((5, 5), dtype=np.uint8)
    gt[2, 3] = ORFD_TRAVERSABLE_GRAY
    m = orfd_traversable_gt_mask(gt)
    assert m.dtype == bool
    assert m.sum() == 1
    assert m[2, 3]


def test_orfd_traversable_legacy_128_override():
    gt = np.zeros((3, 3), dtype=np.uint8)
    gt[1, 1] = 128
    m = orfd_traversable_gt_mask(gt, orfd_trav_gray=128)
    assert m[1, 1]


def test_orfd_eval_valid_excludes_sky_band():
    gt = np.zeros((4, 4), dtype=np.uint8)
    gt[:, :2] = ORFD_SKY_BAND_GRAY
    gt[:, 2:] = ORFD_TRAVERSABLE_GRAY
    valid = orfd_eval_valid_mask(gt)
    assert not valid[:, :2].any()
    assert valid[:, 2:].all()


def test_binary_traversable_iou_identical_masks():
    g = np.zeros((6, 6), dtype=np.uint8)
    g[2:6, 0:3] = ORFD_TRAVERSABLE_GRAY
    valid = orfd_eval_valid_mask(g)
    gt_trav = orfd_traversable_gt_mask(g)
    pred = gt_trav.copy()
    iou = binary_traversable_iou(pred, gt_trav, valid)
    assert iou is not None
    assert abs(iou - 1.0) < 1e-9


def test_binary_partial_overlap_with_sky():
    g = np.array(
        [
            [ORFD_TRAVERSABLE_GRAY, ORFD_NON_TRAVERSABLE_GRAY],
            [ORFD_SKY_BAND_GRAY, ORFD_SKY_BAND_GRAY],
        ],
        dtype=np.uint8,
    )
    valid = orfd_eval_valid_mask(g)
    gt_trav = orfd_traversable_gt_mask(g)
    pred = np.zeros_like(gt_trav)
    pred[0, 0] = True
    pred[0, 1] = True
    iou = binary_traversable_iou(pred, gt_trav, valid)
    assert iou == 0.5
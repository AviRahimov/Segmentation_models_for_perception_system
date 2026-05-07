"""Tests for the histogram-based scene cut detector."""
import numpy as np

from perception.temporal.scene_cut import HistogramSceneCutDetector


def _solid(color, h=64, w=64):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


def test_first_frame_is_never_a_cut():
    det = HistogramSceneCutDetector(threshold=0.45)
    assert det.update(_solid((0, 0, 0))) is False


def test_black_to_white_triggers_cut():
    det = HistogramSceneCutDetector(threshold=0.45)
    det.update(_solid((0, 0, 0)))
    assert det.update(_solid((255, 255, 255))) is True


def test_stable_noise_no_cut():
    rng = np.random.default_rng(0)
    det = HistogramSceneCutDetector(threshold=0.45)
    base = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    det.update(base)
    # tiny perturbation
    pert = base.astype(np.int32) + rng.integers(-3, 4, size=base.shape)
    pert = np.clip(pert, 0, 255).astype(np.uint8)
    assert det.update(pert) is False


def test_reset():
    det = HistogramSceneCutDetector(threshold=0.45)
    det.update(_solid((0, 0, 0)))
    det.reset()
    # After reset, the next frame is treated as the new "first" frame.
    assert det.update(_solid((255, 255, 255))) is False

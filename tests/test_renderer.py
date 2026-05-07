"""Tests for the display-mode-aware renderer."""
import numpy as np
import torch

from perception.config.schema import ClassDef, PlayerCfg
from perception.core.types import Detection, FrameResult, SemanticPrediction
from perception.render.renderer import Renderer


def _frame(h=32, w=32):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _player(show_legend=False, show_fps=False):
    return PlayerCfg(
        mask_alpha=1.0,                       # full overwrite for easy assertions
        show_fps=show_fps,
        show_class_legend=show_legend,
        default_speed=1.0,
    )


def test_bbox_only_draws_no_mask_pixels():
    cls = ClassDef("person", "person", "bbox_only", (255, 0, 0), False)
    rdr = Renderer([cls], _player())
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 1
    det = Detection("person", 0.9, (8, 8, 24, 24), mask)
    out = rdr.render(FrameResult(_frame(), [det]))
    # No interior pixel should match the mask color (only the bbox edge).
    interior = out[12:20, 12:20]
    assert (interior == 0).all()


def test_mask_only_draws_no_bbox():
    cls = ClassDef("road", "road", "mask_only", (0, 0, 255), True, ade20k_indices=(6,))
    # Note: instance Detection with mask_only is allowed; no bbox should appear.
    rdr = Renderer([cls], _player())
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 1
    det = Detection("road", 0.9, (8, 8, 24, 24), mask)
    out = rdr.render(FrameResult(_frame(), [det]))
    # Color at center should be the class color in BGR -> (255, 0, 0).
    assert tuple(int(v) for v in out[16, 16]) == (255, 0, 0)
    # No rectangle outline along the bbox edge: top-left corner outside the
    # interior should remain black.
    assert tuple(int(v) for v in out[7, 7]) == (0, 0, 0)


def test_none_renders_nothing():
    cls = ClassDef("hidden", "hidden", "none", (0, 255, 0), False)
    rdr = Renderer([cls], _player())
    mask = np.ones((32, 32), dtype=np.uint8)
    det = Detection("hidden", 0.9, (4, 4, 28, 28), mask)
    out = rdr.render(FrameResult(_frame(), [det]))
    assert (out == 0).all()


def test_both_draws_mask_and_bbox():
    cls = ClassDef("person", "person", "both", (0, 255, 0), False)
    rdr = Renderer([cls], _player())
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[12:20, 12:20] = 1
    det = Detection("person", 0.9, (12, 12, 20, 20), mask)
    out = rdr.render(FrameResult(_frame(), [det]))
    # mask center is class color
    assert tuple(int(v) for v in out[16, 16]) == (0, 255, 0)


def test_semantic_argmax_taken_on_smoothed_logits():
    cls_a = ClassDef("a", "a", "mask_only", (255, 0, 0), True, ade20k_indices=(0,))
    cls_b = ClassDef("b", "b", "mask_only", (0, 255, 0), True, ade20k_indices=(1,))
    rdr = Renderer([cls_a, cls_b], _player())

    # Build logits where the left half wins for class a, right half for b.
    logits = torch.zeros(2, 16, 16)
    logits[0, :, :8] = 5.0
    logits[1, :, 8:] = 5.0
    sem = SemanticPrediction(logits=logits, class_names=("a", "b"))
    fr = FrameResult(_frame(16, 16), detections=[], semantic=sem)
    out = rdr.render(fr)
    # Left should be class A (BGR red = (0, 0, 255)); right class B (green).
    assert tuple(int(v) for v in out[8, 4]) == (0, 0, 255)
    assert tuple(int(v) for v in out[8, 12]) == (0, 255, 0)


def test_legend_drawn_when_enabled():
    cls = ClassDef("person", "person", "both", (255, 0, 0), False)
    rdr = Renderer([cls], _player(show_legend=True))
    out = rdr.render(FrameResult(_frame(64, 200), []))
    # A legend swatch should appear near the top-left.
    assert (out[15, 15] != 0).any()

"""Tests for postprocess.duplicate_filter and its config plumbing."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src")))

from perception.core.types import Detection  # noqa: E402
from perception.postprocess import filter_duplicates  # noqa: E402


def _det(cls: str, box: tuple[int, int, int, int], score: float) -> Detection:
    return Detection(class_name=cls, score=score, bbox_xyxy=box, mask=None)


def test_nested_same_class_close_scores_keeps_tighter_box():
    outer = _det("vehicle", (0, 0, 200, 200), 0.62)
    inner = _det("vehicle", (50, 50, 150, 150), 0.60)  # fully inside, IoM=1.0
    out = filter_duplicates([outer, inner])
    assert out == [inner]


def test_nested_same_class_confident_outer_wins():
    outer = _det("vehicle", (0, 0, 200, 200), 0.90)
    inner = _det("vehicle", (50, 50, 150, 150), 0.40)
    out = filter_duplicates([outer, inner])
    assert out == [outer]


def test_high_iou_pair_drops_lower_score():
    a = _det("person", (100, 100, 200, 300), 0.80)
    b = _det("person", (105, 105, 205, 305), 0.55)  # IoU ~0.9
    out = filter_duplicates([a, b])
    assert out == [a]


def test_different_classes_never_suppressed():
    vehicle = _det("vehicle", (0, 0, 300, 200), 0.9)
    person = _det("person", (100, 50, 150, 180), 0.9)  # person inside vehicle
    out = filter_duplicates([vehicle, person])
    assert out == [vehicle, person]


def test_disjoint_same_class_boxes_both_kept():
    a = _det("vehicle", (0, 0, 100, 100), 0.7)
    b = _det("vehicle", (200, 200, 300, 300), 0.7)
    out = filter_duplicates([a, b])
    assert out == [a, b]


def test_chain_cluster_single_survivor():
    # a~b overlap heavily, b~c overlap heavily, a~c do not — union-find must
    # still collapse all three into one cluster.
    a = _det("vehicle", (0, 0, 100, 100), 0.50)
    b = _det("vehicle", (10, 0, 110, 100), 0.85)
    c = _det("vehicle", (20, 0, 120, 100), 0.60)
    out = filter_duplicates([a, b, c])
    assert out == [b]


def test_moderate_overlap_below_thresholds_kept():
    # IoU ~0.33, no containment — two vehicles side by side, partially overlapping.
    a = _det("vehicle", (0, 0, 100, 100), 0.7)
    b = _det("vehicle", (50, 0, 150, 100), 0.7)
    out = filter_duplicates([a, b])
    assert out == [a, b]


def test_empty_and_single_passthrough():
    assert filter_duplicates([]) == []
    only = [_det("person", (0, 0, 10, 10), 0.5)]
    assert filter_duplicates(only) == only


# --------------------------------------------------------------------------- #
# Config plumbing                                                              #
# --------------------------------------------------------------------------- #

_MINIMAL_YAML = """
models:
  instance: {name: "yoloe26l"}
  semantic: {name: "segformer-b2"}
classes:
  - name: "person"
    text_prompt: "person"
    display_mode: "both"
    color: "green"
    is_semantic: false
  - name: "road_ground"
    text_prompt: "road"
    display_mode: "mask_only"
    color: "blue"
    is_semantic: true
    ade20k_indices: [6, 13]
temporal: {}
hardware: {device: "cpu", fp16: false}
player: {}
source: {type: "video", path: "x.mp4"}
"""


def _load_yaml(tmp_path: Path, text: str):
    from perception.config.loader import load_config
    p = tmp_path / "cfg.yaml"
    p.write_text(text)
    return load_config(str(p))


def test_config_defaults_when_section_absent(tmp_path):
    cfg = _load_yaml(tmp_path, _MINIMAL_YAML)
    df = cfg.postprocess.duplicate_filter
    assert df.enabled is True
    assert df.iou_threshold == pytest.approx(0.55)
    assert df.containment_threshold == pytest.approx(0.85)
    assert df.score_margin == pytest.approx(0.05)


def test_config_section_parsed(tmp_path):
    cfg = _load_yaml(tmp_path, _MINIMAL_YAML + """
postprocess:
  duplicate_filter:
    enabled: false
    iou_threshold: 0.7
    containment_threshold: 0.9
    score_margin: 0.1
""")
    df = cfg.postprocess.duplicate_filter
    assert df.enabled is False
    assert df.iou_threshold == pytest.approx(0.7)
    assert df.containment_threshold == pytest.approx(0.9)
    assert df.score_margin == pytest.approx(0.1)


def test_config_rejects_bad_threshold(tmp_path):
    from perception.config.loader import ConfigError
    with pytest.raises(ConfigError):
        _load_yaml(tmp_path, _MINIMAL_YAML + """
postprocess:
  duplicate_filter: {iou_threshold: 1.5}
""")

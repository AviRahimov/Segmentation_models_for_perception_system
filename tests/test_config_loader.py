"""Tests for the YAML config loader and validator."""
from pathlib import Path

import pytest

from perception.config.loader import ConfigError, load_config, override_source


_VALID_YAML = """
models:
  instance:
    name: "yoloe26l"
    confidence_threshold: 0.4
  semantic:
    name: "segformer-b2"

classes:
  - name: "person"
    text_prompt: "person"
    display_mode: "both"
    color_rgb: [0, 255, 0]
    is_semantic: false
  - name: "road_ground"
    text_prompt: "road"
    display_mode: "mask_only"
    color_rgb: [100, 100, 255]
    is_semantic: true
    ade20k_indices: [6, 13]

temporal:
  semantic_ema:
    alpha: 0.4
  instance_sam2:
    enabled: false

hardware:
  device: "cpu"
  fp16: false

player:
  mask_alpha: 0.5

source:
  type: "video"
  path: "x.mp4"

datasets:
  download_dir: "./d"
"""


def _write(tmp: Path, body: str) -> Path:
    f = tmp / "config.yaml"
    f.write_text(body)
    return f


def test_round_trip(tmp_path):
    cfg = load_config(_write(tmp_path, _VALID_YAML))
    assert cfg.models.instance.name == "yoloe26l"
    assert cfg.models.instance.confidence_threshold == 0.4
    assert len(cfg.classes) == 2
    assert cfg.classes[0].is_semantic is False
    assert cfg.classes[1].ade20k_indices == (6, 13)
    assert cfg.temporal.semantic_ema.alpha == 0.4
    assert cfg.hardware.device == "cpu"
    assert cfg.source.type == "video"
    assert cfg.source.path == "x.mp4"


def test_instance_classes_property(tmp_path):
    cfg = load_config(_write(tmp_path, _VALID_YAML))
    assert [c.name for c in cfg.instance_classes] == ["person"]
    assert [c.name for c in cfg.semantic_classes] == ["road_ground"]


def test_semantic_class_requires_indices(tmp_path):
    bad = _VALID_YAML.replace("ade20k_indices: [6, 13]", "")
    with pytest.raises(ConfigError, match="ade20k_indices"):
        load_config(_write(tmp_path, bad))


def test_instance_class_must_not_have_indices(tmp_path):
    bad = _VALID_YAML.replace(
        '    is_semantic: false',
        '    is_semantic: false\n    ade20k_indices: [1]',
    )
    with pytest.raises(ConfigError, match="must NOT define"):
        load_config(_write(tmp_path, bad))


def test_invalid_display_mode(tmp_path):
    bad = _VALID_YAML.replace('display_mode: "both"', 'display_mode: "magic"')
    with pytest.raises(ConfigError, match="display_mode"):
        load_config(_write(tmp_path, bad))


def test_duplicate_class_name(tmp_path):
    bad = _VALID_YAML.replace('"road_ground"', '"person"')
    with pytest.raises(ConfigError, match="Duplicate class"):
        load_config(_write(tmp_path, bad))


def test_alpha_out_of_range(tmp_path):
    bad = _VALID_YAML.replace("alpha: 0.4", "alpha: 1.5")
    with pytest.raises(ConfigError, match="alpha"):
        load_config(_write(tmp_path, bad))


def test_color_validation(tmp_path):
    bad = _VALID_YAML.replace("color_rgb: [0, 255, 0]", "color_rgb: [0, 999, 0]")
    with pytest.raises(ConfigError, match="color_rgb"):
        load_config(_write(tmp_path, bad))


def test_override_source(tmp_path):
    cfg = load_config(_write(tmp_path, _VALID_YAML))
    new = override_source(cfg, source_type="camera", camera=2)
    assert new.source.type == "camera"
    assert new.source.camera_index == 2
    # Original is unchanged.
    assert cfg.source.type == "video"


def test_missing_classes(tmp_path):
    bad = _VALID_YAML.replace("classes:", "classes_BAD:")
    with pytest.raises(ConfigError, match="classes"):
        load_config(_write(tmp_path, bad))


# --------------------------------------------------------------------------- #
# Per-class confidence threshold                                              #
# --------------------------------------------------------------------------- #


def test_per_class_confidence_default_is_none(tmp_path):
    cfg = load_config(_write(tmp_path, _VALID_YAML))
    # Neither class in _VALID_YAML sets a per-class threshold.
    assert all(c.confidence_threshold is None for c in cfg.classes)


def test_per_class_confidence_parsed(tmp_path):
    body = _VALID_YAML.replace(
        '    is_semantic: false\n  - name: "road_ground"',
        '    is_semantic: false\n    confidence_threshold: 0.2\n  - name: "road_ground"',
    )
    cfg = load_config(_write(tmp_path, body))
    inst = next(c for c in cfg.classes if c.name == "person")
    assert inst.confidence_threshold == 0.2


def test_per_class_confidence_out_of_range(tmp_path):
    body = _VALID_YAML.replace(
        '    is_semantic: false\n  - name: "road_ground"',
        '    is_semantic: false\n    confidence_threshold: 1.5\n  - name: "road_ground"',
    )
    with pytest.raises(ConfigError, match="confidence_threshold"):
        load_config(_write(tmp_path, body))


def test_per_class_confidence_rejected_on_semantic_class(tmp_path):
    body = _VALID_YAML.replace(
        "    ade20k_indices: [6, 13]",
        "    ade20k_indices: [6, 13]\n    confidence_threshold: 0.5",
    )
    with pytest.raises(ConfigError, match="must NOT define confidence_threshold"):
        load_config(_write(tmp_path, body))


def test_per_class_confidence_non_numeric(tmp_path):
    body = _VALID_YAML.replace(
        '    is_semantic: false\n  - name: "road_ground"',
        '    is_semantic: false\n    confidence_threshold: "low"\n  - name: "road_ground"',
    )
    with pytest.raises(ConfigError, match="confidence_threshold"):
        load_config(_write(tmp_path, body))

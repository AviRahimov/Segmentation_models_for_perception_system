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
    color: "green"
    is_semantic: false
  - name: "road_ground"
    text_prompt: "road"
    display_mode: "mask_only"
    color: "blue"
    is_semantic: true
    ade20k_indices: [6, 13]

temporal:
  semantic_ema:
    alpha: 0.4

hardware:
  device: "cpu"
  fp16: false

player:
  mask_alpha: 0.5

source:
  type: "video"
  path: "x.mp4"
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
    with pytest.raises(ConfigError, match="non-empty"):
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
    """An unknown color name produces a clear ConfigError."""
    bad = _VALID_YAML.replace('color: "green"', 'color: "octarine"')
    with pytest.raises(ConfigError, match="unknown color name"):
        load_config(_write(tmp_path, bad))


def test_instance_tracker_backend_defaults_to_iou(tmp_path):
    cfg = load_config(_write(tmp_path, _VALID_YAML))
    assert cfg.temporal.instance_tracker.backend == "iou"
    assert cfg.temporal.instance_tracker.frame_rate == 30.0


def test_instance_tracker_backend_bytetrack(tmp_path):
    yaml_with_trk = _VALID_YAML.replace(
        "temporal:\n  semantic_ema:\n    alpha: 0.4",
        "temporal:\n  semantic_ema:\n    alpha: 0.4\n"
        "  instance_tracker:\n    backend: \"bytetrack\"\n    frame_rate: 24.0",
    )
    cfg = load_config(_write(tmp_path, yaml_with_trk))
    assert cfg.temporal.instance_tracker.backend == "bytetrack"
    assert cfg.temporal.instance_tracker.frame_rate == 24.0


def test_instance_tracker_backend_rejects_unknown(tmp_path):
    yaml_with_trk = _VALID_YAML.replace(
        "temporal:\n  semantic_ema:\n    alpha: 0.4",
        "temporal:\n  semantic_ema:\n    alpha: 0.4\n"
        "  instance_tracker:\n    backend: \"sort\"",
    )
    with pytest.raises(ConfigError, match="backend must be 'iou' or 'bytetrack'"):
        load_config(_write(tmp_path, yaml_with_trk))


def test_instance_tracker_frame_rate_must_be_positive(tmp_path):
    yaml_with_trk = _VALID_YAML.replace(
        "temporal:\n  semantic_ema:\n    alpha: 0.4",
        "temporal:\n  semantic_ema:\n    alpha: 0.4\n"
        "  instance_tracker:\n    frame_rate: 0",
    )
    with pytest.raises(ConfigError, match="frame_rate must be > 0"):
        load_config(_write(tmp_path, yaml_with_trk))


def test_calibration_defaults_disabled(tmp_path):
    cfg = load_config(_write(tmp_path, _VALID_YAML))
    assert cfg.postprocess.calibration.enabled is False
    assert cfg.postprocess.calibration.temperatures_path is None
    assert cfg.postprocess.calibration.default_temperature == 1.0


def test_calibration_enabled_requires_temperatures_path(tmp_path):
    bad = _VALID_YAML + "\npostprocess:\n  calibration:\n    enabled: true\n"
    with pytest.raises(ConfigError, match="temperatures_path"):
        load_config(_write(tmp_path, bad))


def test_calibration_round_trip(tmp_path):
    good = _VALID_YAML + (
        "\npostprocess:\n  calibration:\n    enabled: true\n"
        "    temperatures_path: \"weights/detection/calibration/x.json\"\n"
        "    default_temperature: 1.2\n"
    )
    cfg = load_config(_write(tmp_path, good))
    assert cfg.postprocess.calibration.enabled is True
    assert cfg.postprocess.calibration.temperatures_path == "weights/detection/calibration/x.json"
    assert cfg.postprocess.calibration.default_temperature == 1.2


def test_calibration_default_temperature_must_be_positive(tmp_path):
    bad = _VALID_YAML + (
        "\npostprocess:\n  calibration:\n    enabled: true\n"
        "    temperatures_path: \"x.json\"\n    default_temperature: 0\n"
    )
    with pytest.raises(ConfigError, match="default_temperature must be > 0"):
        load_config(_write(tmp_path, bad))


def test_override_source(tmp_path):
    cfg = load_config(_write(tmp_path, _VALID_YAML))
    new = override_source(cfg, source_type="camera", camera=2)
    assert new.source.type == "camera"
    assert new.source.camera_index == 2
    # Original is unchanged.
    assert cfg.source.type == "video"


def test_discovery_mode_requires_vocab_path(tmp_path):
    disc = _VALID_YAML.replace(
        "    name: \"yoloe26l\"\n    confidence_threshold: 0.4\n",
        "    name: \"yoloe26l\"\n    confidence_threshold: 0.4\n"
        "    prompt_mode: discovery\n",
        1,
    )
    with pytest.raises(ConfigError, match="discovery_vocabulary_path"):
        load_config(_write(tmp_path, disc))


def test_discovery_mode_resolves_vocab_relative_to_yaml(tmp_path):
    (tmp_path / "voc.txt").write_text("# h\nperson\ncar\n", encoding="utf-8")
    disc = _VALID_YAML.replace(
        "    name: \"yoloe26l\"\n    confidence_threshold: 0.4\n",
        "    name: \"yoloe26l\"\n    confidence_threshold: 0.4\n"
        "    prompt_mode: discovery\n"
        '    discovery_vocabulary_path: "voc.txt"\n',
        1,
    )
    cfg = load_config(_write(tmp_path, disc))
    assert cfg.models.instance.prompt_mode == "discovery"
    assert cfg.models.instance.discovery_vocabulary_path.endswith("voc.txt")
    assert cfg.runs_yoloe_instance_inference is True


def test_runs_yoloe_instance_inference_production_follows_classes(tmp_path):
    cfg = load_config(_write(tmp_path, _VALID_YAML))
    assert cfg.runs_yoloe_instance_inference is True


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


# --------------------------------------------------------------------------- #
# native_indices                                                               #
# --------------------------------------------------------------------------- #


def test_legacy_ade20k_indices_shim_still_works(tmp_path):
    """Existing ``ade20k_indices: [...]`` shorthand routes into native_indices
    and the ``ClassDef.ade20k_indices`` property still returns the same tuple."""
    cfg = load_config(_write(tmp_path, _VALID_YAML))
    rg = next(c for c in cfg.classes if c.name == "road_ground")
    assert rg.ade20k_indices == (6, 13)
    assert rg.native_indices == {"ade20k": (6, 13)}


def test_native_indices_dict_form(tmp_path):
    body = _VALID_YAML.replace(
        "    ade20k_indices: [6, 13]",
        "    native_indices:\n      ade20k: [6, 13]\n      goose_12: [5]",
    )
    cfg = load_config(_write(tmp_path, body))
    rg = next(c for c in cfg.classes if c.name == "road_ground")
    assert rg.native_indices == {"ade20k": (6, 13), "goose_12": (5,)}
    # Backward-compat read API still works.
    assert rg.ade20k_indices == (6, 13)


def test_native_indices_unknown_key_rejected(tmp_path):
    body = _VALID_YAML.replace(
        "    ade20k_indices: [6, 13]",
        "    native_indices:\n      ade20k:   [6, 13]\n      goose_64: [3]",
    )
    with pytest.raises(ConfigError, match="unknown native_indices key 'goose_64'"):
        load_config(_write(tmp_path, body))


def test_native_indices_out_of_range_ade20k(tmp_path):
    body = _VALID_YAML.replace("ade20k_indices: [6, 13]", "ade20k_indices: [6, 200]")
    with pytest.raises(ConfigError, match=r"out of range \[0, 150\)"):
        load_config(_write(tmp_path, body))


def test_native_indices_out_of_range_goose_12(tmp_path):
    body = _VALID_YAML.replace(
        "    ade20k_indices: [6, 13]",
        "    native_indices:\n      goose_12: [13]",
    )
    with pytest.raises(ConfigError, match=r"out of range \[0, 12\)"):
        load_config(_write(tmp_path, body))


def test_orfd_semantic_comparison_defaults_and_goose_categories(tmp_path):
    body = (
        _VALID_YAML
        + "\norfd_semantic_comparison:\n  goose:\n    samples: 3\n"
        + "    traversable_categories: [terrain, road]\n"
    )
    cfg = load_config(_write(tmp_path, body))
    o = cfg.orfd_semantic_comparison
    assert o.orfd_trav_gray == 255
    assert o.goose.samples == 3
    assert o.goose.traversable_categories == ("terrain", "road")
    assert o.instance_mask_subtraction.subtract_from_traversable is False


def test_orfd_semantic_comparison_freespace_prob_floor(tmp_path):
    body = (
        _VALID_YAML
        + "\norfd_semantic_comparison:\n  freespace_merged_prob_floor: 0.35\n"
    )
    cfg = load_config(_write(tmp_path, body))
    assert cfg.orfd_semantic_comparison.freespace_merged_prob_floor == pytest.approx(0.35)


def test_orfd_semantic_comparison_unknown_goose_category(tmp_path):
    body = _VALID_YAML + (
        "\norfd_semantic_comparison:\n  goose:\n"
        "    traversable_categories: [terrain, not_a_real_class]\n"
    )
    with pytest.raises(ConfigError, match="unknown GOOSE-12 category"):
        load_config(_write(tmp_path, body))


def test_instance_class_rejects_native_indices(tmp_path):
    body = _VALID_YAML.replace(
        '    is_semantic: false',
        '    is_semantic: false\n    native_indices: {goose_12: [2]}',
        1,
    )
    with pytest.raises(ConfigError, match="must NOT define"):
        load_config(_write(tmp_path, body))


# --------------------------------------------------------------------------- #
# color                                                                        #
# --------------------------------------------------------------------------- #


def test_color_named_parsed(tmp_path):
    cfg = load_config(_write(tmp_path, _VALID_YAML))
    person = next(c for c in cfg.classes if c.name == "person")
    # "green" -> our palette's green: (0, 160, 60).
    assert person.color_rgb == (0, 160, 60)


def test_color_hex_parsed(tmp_path):
    body = _VALID_YAML.replace('color: "green"', 'color: "#112233"')
    cfg = load_config(_write(tmp_path, body))
    person = next(c for c in cfg.classes if c.name == "person")
    assert person.color_rgb == (0x11, 0x22, 0x33)


def test_color_rgb_array_form_rejected(tmp_path):
    body = _VALID_YAML.replace('color: "green"', "color_rgb: [0, 255, 0]")
    with pytest.raises(ConfigError, match="color_rgb is deprecated"):
        load_config(_write(tmp_path, body))


def test_color_4digit_hex_rejected(tmp_path):
    body = _VALID_YAML.replace('color: "green"', 'color: "#0ff0"')
    with pytest.raises(ConfigError, match="6-digit"):
        load_config(_write(tmp_path, body))


# --------------------------------------------------------------------------- #
# coco_classes                                                                 #
# --------------------------------------------------------------------------- #


def test_coco_classes_parsed(tmp_path):
    body = _VALID_YAML.replace(
        '    is_semantic: false\n  - name: "road_ground"',
        '    is_semantic: false\n    coco_classes: [1, 3]\n  - name: "road_ground"',
    )
    cfg = load_config(_write(tmp_path, body))
    person = next(c for c in cfg.classes if c.name == "person")
    assert person.coco_classes == (1, 3)


def test_coco_classes_default_is_empty(tmp_path):
    cfg = load_config(_write(tmp_path, _VALID_YAML))
    person = next(c for c in cfg.classes if c.name == "person")
    assert person.coco_classes == ()


def test_coco_classes_empty_list_allowed(tmp_path):
    body = _VALID_YAML.replace(
        '    is_semantic: false\n  - name: "road_ground"',
        '    is_semantic: false\n    coco_classes: []\n  - name: "road_ground"',
    )
    cfg = load_config(_write(tmp_path, body))
    person = next(c for c in cfg.classes if c.name == "person")
    assert person.coco_classes == ()


def test_coco_classes_rejected_on_semantic_class(tmp_path):
    body = _VALID_YAML.replace(
        "    ade20k_indices: [6, 13]",
        "    ade20k_indices: [6, 13]\n    coco_classes: [1]",
    )
    with pytest.raises(ConfigError, match="must NOT define coco_classes"):
        load_config(_write(tmp_path, body))


def test_coco_classes_non_int_rejected(tmp_path):
    body = _VALID_YAML.replace(
        '    is_semantic: false\n  - name: "road_ground"',
        '    is_semantic: false\n    coco_classes: ["person"]\n  - name: "road_ground"',
    )
    with pytest.raises(ConfigError, match="coco_classes must be a list of ints"):
        load_config(_write(tmp_path, body))


# --------------------------------------------------------------------------- #
# instance_profiles                                                            #
# --------------------------------------------------------------------------- #

_PROFILE_YAML = """
models:
  instance:
    name: "yolo11m"
    profile: "6class"
  semantic:
    name: "segformer-b2"
    num_classes: 3

instance_profiles:
  2class:
    - {name: "vehicle", text_prompt: "vehicle", coco_classes: [1], display_mode: "both", color: "green", is_semantic: false}
    - {name: "person",  text_prompt: "person", coco_classes: [2], display_mode: "both", color: "blue",  is_semantic: false}
  6class:
    - {name: "tank",    text_prompt: "tank", coco_classes: [1], display_mode: "both", color: "green", is_semantic: false, confidence_threshold: 0.5}
    - {name: "soldier", text_prompt: "soldier", coco_classes: [5], display_mode: "both", color: "blue",  is_semantic: false, confidence_threshold: 0.3}

classes:
  - name: "road_ground"
    text_prompt: "road"
    display_mode: "mask_only"
    color: "blue"
    is_semantic: true

temporal: {}
hardware: {device: "cpu", fp16: false}
player: {}
source: {type: "video", path: "x.mp4"}
"""


def _load(tmp_path, text):
    p = tmp_path / "cfg.yaml"
    p.write_text(text)
    return load_config(p)


def test_profile_selects_instance_classes(tmp_path):
    cfg = _load(tmp_path, _PROFILE_YAML)
    inst = [c.name for c in cfg.classes if not c.is_semantic]
    sem = [c.name for c in cfg.classes if c.is_semantic]
    assert inst == ["tank", "soldier"]
    assert sem == ["road_ground"]
    tank = next(c for c in cfg.classes if c.name == "tank")
    assert tank.confidence_threshold == pytest.approx(0.5)
    assert cfg.models.instance.profile == "6class"


def test_profile_switch_changes_classes(tmp_path):
    cfg = _load(tmp_path, _PROFILE_YAML.replace('profile: "6class"', 'profile: "2class"'))
    inst = [c.name for c in cfg.classes if not c.is_semantic]
    assert inst == ["vehicle", "person"]


def test_unknown_profile_rejected(tmp_path):
    with pytest.raises(ConfigError, match="Unknown instance profile"):
        _load(tmp_path, _PROFILE_YAML.replace('profile: "6class"', 'profile: "9class"'))


def test_missing_profile_key_rejected_when_profiles_exist(tmp_path):
    with pytest.raises(ConfigError, match="must select one of"):
        _load(tmp_path, _PROFILE_YAML.replace('profile: "6class"', 'enabled: true'))


def test_instance_class_in_classes_rejected_in_profile_mode(tmp_path):
    bad = _PROFILE_YAML.replace(
        'classes:\n  - name: "road_ground"',
        'classes:\n  - {name: "stray", text_prompt: "stray", coco_classes: [9], display_mode: "both", '
        'color: "red", is_semantic: false}\n  - name: "road_ground"',
    )
    with pytest.raises(ConfigError, match="only.*semantic"):
        _load(tmp_path, bad)


# --------------------------------------------------------------------------- #
# models.instance.low_conf_recovery                                           #
# --------------------------------------------------------------------------- #

_LCR_YAML = """
models:
  instance:
    name: "yoloe26l"
    low_conf_recovery: {{{lcr_body}}}
  semantic:
    name: "segformer-b2"
classes:
  - name: "person"
    text_prompt: "person"
    display_mode: "both"
    color: "green"
    is_semantic: false
temporal: {{}}
hardware: {{device: "cpu", fp16: false}}
player: {{}}
source: {{type: "video", path: "x.mp4"}}
"""


def test_low_conf_recovery_defaults_when_absent(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(_LCR_YAML.format(lcr_body=""))
    cfg = load_config(p)
    lcr = cfg.models.instance.low_conf_recovery
    assert lcr.enabled is False
    assert lcr.recovery_conf_floor == pytest.approx(0.15)


def test_low_conf_recovery_parses_explicit_values(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(_LCR_YAML.format(lcr_body="enabled: true, recovery_conf_floor: 0.1"))
    cfg = load_config(p)
    lcr = cfg.models.instance.low_conf_recovery
    assert lcr.enabled is True
    assert lcr.recovery_conf_floor == pytest.approx(0.1)


def test_low_conf_recovery_floor_out_of_range_rejected(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(_LCR_YAML.format(lcr_body="recovery_conf_floor: 1.5"))
    with pytest.raises(ConfigError, match="recovery_conf_floor"):
        load_config(p)

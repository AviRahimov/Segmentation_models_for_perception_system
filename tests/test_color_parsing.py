"""Tests for the named-color / hex parser used by the YAML config loader."""
import pytest

from perception.config.loader import ConfigError, load_config
from perception.core.colors_named import _NAMED_COLORS, parse_color


def test_named_palette_size_is_30():
    assert len(_NAMED_COLORS) == 30


def test_named_palette_keys_are_lowercase():
    for k in _NAMED_COLORS:
        assert k == k.lower()


def test_parse_named_color():
    assert parse_color("red") == _NAMED_COLORS["red"]
    assert parse_color(" Red ") == _NAMED_COLORS["red"]      # case + whitespace
    assert parse_color("MINT") == _NAMED_COLORS["mint"]


def test_parse_hex_with_hash():
    assert parse_color("#ff8000") == (0xFF, 0x80, 0x00)


def test_parse_hex_without_hash():
    assert parse_color("123abc") == (0x12, 0x3a, 0xbc)


def test_parse_hex_uppercase():
    assert parse_color("#A1B2C3") == (0xA1, 0xB2, 0xC3)


def test_parse_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown color name"):
        parse_color("octarine")


def test_parse_4_digit_hex_rejected():
    with pytest.raises(ValueError, match="6-digit"):
        parse_color("#abcd")


def test_parse_8_digit_hex_rejected():
    with pytest.raises(ValueError, match="6-digit"):
        parse_color("#abcdef12")


def test_parse_3_digit_hex_rejected():
    with pytest.raises(ValueError, match="6-digit"):
        parse_color("#abc")


def test_parse_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        parse_color("   ")


def test_parse_non_string_raises():
    with pytest.raises(ValueError, match="must be a string"):
        parse_color(0xFF8000)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Loader-level: legacy color_rgb is rejected with a deprecation message.       #
# --------------------------------------------------------------------------- #


_LEGACY_YAML = """
models:
  instance: {name: "yoloe26l"}
  semantic: {name: "segformer-b2"}

classes:
  - name: "person"
    text_prompt: "person"
    display_mode: "both"
    color_rgb: [0, 255, 0]
    is_semantic: false

temporal: {}
hardware: {device: "cpu", fp16: false}
player: {}
source: {type: "video", path: "x.mp4"}
datasets: {}
"""


def test_legacy_color_rgb_array_is_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(_LEGACY_YAML)
    with pytest.raises(ConfigError, match="color_rgb is deprecated"):
        load_config(p)

"""Tests for the native class catalogues used by the semantic models."""
from perception.models.semantic._class_catalogues import (
    ADE20K_150_NAMES,
    CATALOGUE_SIZES,
    GOOSE_12_NAMES,
)


def test_ade20k_length():
    assert len(ADE20K_150_NAMES) == 150


def test_goose_12_length():
    assert len(GOOSE_12_NAMES) == 12


def test_goose_12_road_index():
    # Locked by pre-resolved decision #2: GOOSE-12 ordering puts "road"
    # at index 5 (vegetation, terrain, vehicle, object, construction, road, ...).
    assert GOOSE_12_NAMES[5] == "road"


def test_goose_12_sky_index():
    assert GOOSE_12_NAMES[8] == "sky"


def test_catalogue_sizes_consistent():
    assert CATALOGUE_SIZES["ade20k"] == len(ADE20K_150_NAMES)
    assert CATALOGUE_SIZES["goose_12"] == len(GOOSE_12_NAMES)
    # GOOSE-64 intentionally not yet shipped.
    assert "goose_64" not in CATALOGUE_SIZES


def test_ade20k_canonical_anchors():
    # Spot-check a few canonical SceneParse150 indices that the rest of
    # the pipeline (and config.yaml comments) rely on.
    assert ADE20K_150_NAMES[2] == "sky"
    assert ADE20K_150_NAMES[6] == "road"
    assert ADE20K_150_NAMES[9] == "grass"
    assert ADE20K_150_NAMES[46] == "sand"
    assert ADE20K_150_NAMES[91] == "dirt track"

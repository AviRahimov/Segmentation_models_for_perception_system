"""Tests for postprocess.calibration (temperature scaling)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src")))

from perception.core.types import Detection  # noqa: E402
from perception.postprocess.calibration import (  # noqa: E402
    apply_calibration,
    apply_temperature,
    fit_temperature,
    fit_temperatures,
    load_temperatures,
    save_temperatures,
)


def _det(cls: str, score: float) -> Detection:
    return Detection(class_name=cls, score=score, bbox_xyxy=(0, 0, 10, 10), mask=None)


# --------------------------------------------------------------------------- #
# apply_temperature — pure math                                               #
# --------------------------------------------------------------------------- #

def test_apply_temperature_one_is_noop():
    assert apply_temperature(0.73, 1.0) == 0.73


def test_apply_temperature_above_one_pulls_toward_half():
    # T > 1 softens (moves toward 0.5) any score, high or low.
    assert apply_temperature(0.9, 3.0) < 0.9
    assert apply_temperature(0.1, 3.0) > 0.1


def test_apply_temperature_below_one_sharpens():
    assert apply_temperature(0.9, 0.3) > 0.9
    assert apply_temperature(0.1, 0.3) < 0.1


# --------------------------------------------------------------------------- #
# fit_temperature — recovers a known miscalibration                           #
# --------------------------------------------------------------------------- #

def test_fit_temperature_recovers_overconfident_detector():
    # Ground truth: true probability is score/2 (detector systematically
    # overconfident by a factor of 2 in probability space) — simulate labels
    # by sampling a Bernoulli at that true rate for many synthetic scores.
    rng = np.random.default_rng(0)
    scores = rng.uniform(0.05, 0.95, 2000)
    true_p = scores / 2.0
    labels = (rng.uniform(size=len(scores)) < true_p).astype(float)

    t = fit_temperature(scores, labels)
    # An overconfident detector needs T > 1 (softening) to calibrate.
    assert t > 1.0


def test_fit_temperature_empty_returns_noop():
    assert fit_temperature(np.array([]), np.array([])) == 1.0


def test_fit_temperature_single_label_class_returns_noop():
    # All TP (or all FP) carries no information about a threshold — NLL has
    # no informative minimum, so fitting should degrade to a no-op.
    scores = np.array([0.6, 0.7, 0.8])
    labels = np.array([1.0, 1.0, 1.0])
    assert fit_temperature(scores, labels) == 1.0


def test_fit_temperatures_per_class():
    rng = np.random.default_rng(1)
    scores_a = rng.uniform(0.1, 0.9, 200)
    labels_a = (rng.uniform(size=200) < scores_a / 2).astype(float)
    scores_b = rng.uniform(0.1, 0.9, 200)
    labels_b = (rng.uniform(size=200) < scores_b).astype(float)  # already calibrated

    out = fit_temperatures({"vehicle": (scores_a, labels_a), "person": (scores_b, labels_b)})
    assert set(out) == {"vehicle", "person"}
    assert out["vehicle"] > 1.0


# --------------------------------------------------------------------------- #
# apply_calibration — Detection list rescaling                                #
# --------------------------------------------------------------------------- #

def test_apply_calibration_rescales_matching_class_only():
    dets = [_det("vehicle", 0.9), _det("person", 0.9)]
    out = apply_calibration(dets, {"vehicle": 2.0})
    assert out[0].score == pytest.approx(apply_temperature(0.9, 2.0))
    assert out[1].score == 0.9  # "person" absent -> default_temperature=1.0 -> unchanged


def test_apply_calibration_empty_list_returns_as_is():
    assert apply_calibration([], {"vehicle": 2.0}) == []


def test_apply_calibration_default_temperature_applies_to_unknown_class():
    dets = [_det("unknown_cls", 0.8)]
    out = apply_calibration(dets, {}, default_temperature=2.0)
    assert out[0].score == pytest.approx(apply_temperature(0.8, 2.0))


def test_apply_calibration_preserves_other_fields():
    d = _det("vehicle", 0.9)
    out = apply_calibration([d], {"vehicle": 2.0})
    assert out[0].bbox_xyxy == d.bbox_xyxy
    assert out[0].class_name == d.class_name


# --------------------------------------------------------------------------- #
# save/load round trip                                                        #
# --------------------------------------------------------------------------- #

def test_save_and_load_temperatures_round_trip(tmp_path):
    path = tmp_path / "temps.json"
    save_temperatures({"vehicle": 1.7, "person": 1.0}, path)
    loaded = load_temperatures(path)
    assert loaded == {"vehicle": 1.7, "person": 1.0}
    assert json.loads(path.read_text()) == {"vehicle": 1.7, "person": 1.0}

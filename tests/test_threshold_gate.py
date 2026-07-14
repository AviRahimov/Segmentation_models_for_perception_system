"""Tests for the shared confidence-gate helper used by both YOLO wrappers."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from perception.models.instance._threshold_gate import gate_confidence  # noqa: E402


def test_above_threshold_kept_recovery_disabled():
    assert gate_confidence(0.6, threshold=0.5, recovery_floor=None) is True


def test_below_threshold_dropped_recovery_disabled():
    assert gate_confidence(0.4, threshold=0.5, recovery_floor=None) is False


def test_between_floor_and_threshold_kept_when_recovery_enabled():
    assert gate_confidence(0.2, threshold=0.5, recovery_floor=0.15) is True


def test_below_both_floor_and_threshold_dropped():
    assert gate_confidence(0.1, threshold=0.5, recovery_floor=0.15) is False


def test_above_threshold_kept_recovery_enabled_too():
    assert gate_confidence(0.9, threshold=0.5, recovery_floor=0.15) is True


def test_exactly_at_recovery_floor_kept():
    assert gate_confidence(0.15, threshold=0.5, recovery_floor=0.15) is True


def test_exactly_at_threshold_kept():
    assert gate_confidence(0.5, threshold=0.5, recovery_floor=None) is True


def test_recovery_floor_above_threshold_has_no_effect():
    # Misconfiguration guard: a recovery_floor > threshold can't lower the
    # effective floor — it's always min(threshold, recovery_floor).
    assert gate_confidence(0.45, threshold=0.5, recovery_floor=0.8) is False
    assert gate_confidence(0.6, threshold=0.5, recovery_floor=0.8) is True

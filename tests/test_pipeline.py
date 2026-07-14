"""Tests for PerceptionPipeline orchestration logic (stubbed collaborators —
no real models loaded).

Currently covers: the instance_tracker.enabled master bypass (Stage 1 of the
tracker upgrade) — when disabled, detections must pass through untouched and
the tracker's update() must never be called.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from perception.config.loader import load_config  # noqa: E402
from perception.core.types import Detection  # noqa: E402
from perception.models.instance.base import InstanceModel  # noqa: E402
from perception.models.semantic.base import SemanticModel  # noqa: E402
from perception.pipeline.perception import PerceptionPipeline  # noqa: E402
from perception.temporal.base import (  # noqa: E402
    InstanceTracker,
    LogitsSmoother,
    SceneCutDetector,
)

_FRAME = np.zeros((8, 8, 3), dtype=np.uint8)
_CANNED = [Detection(class_name="person", score=0.9, bbox_xyxy=(0, 0, 4, 4), mask=None)]


class _StubInstance(InstanceModel):
    def warmup(self, classes):
        pass

    def predict(self, frame_bgr):
        return list(_CANNED)


class _StubSemantic(SemanticModel):
    def warmup(self, classes):
        pass

    def predict_logits(self, frame_bgr):
        return torch.zeros(1, 4, 4)

    @property
    def class_names(self):
        return ("road_ground",)


class _SpyTracker(InstanceTracker):
    def __init__(self):
        self.calls = 0

    def update(self, frame_bgr, detections: Sequence[Detection]) -> list[Detection]:
        self.calls += 1
        return [Detection(class_name=d.class_name, score=d.score,
                          bbox_xyxy=d.bbox_xyxy, mask=d.mask, track_id=999)
                for d in detections]

    def reset(self):
        pass


class _NoopSmoother(LogitsSmoother):
    def update(self, logits):
        return logits

    def reset(self):
        pass


class _NoopSceneCut(SceneCutDetector):
    def update(self, frame_bgr):
        return False

    def reset(self):
        pass


_YAML = """
models:
  instance: {{name: "yoloe26l"}}
  semantic: {{name: "segformer-b2"}}
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
  instance_tracker: {{enabled: {enabled}}}
hardware: {{device: "cpu", fp16: false}}
player: {{}}
source: {{type: "video", path: "x.mp4"}}
"""


def _build(tmp_path: Path, enabled: bool) -> tuple[PerceptionPipeline, _SpyTracker]:
    p = tmp_path / "cfg.yaml"
    p.write_text(_YAML.format(enabled=str(enabled).lower()))
    cfg = load_config(p)
    tracker = _SpyTracker()
    pipeline = PerceptionPipeline(
        instance_model=_StubInstance(),
        semantic_model=_StubSemantic(),
        instance_tracker=tracker,
        logits_smoother=_NoopSmoother(),
        scene_cut_detector=_NoopSceneCut(),
        config=cfg,
    )
    return pipeline, tracker


def test_tracker_enabled_by_default_runs_update(tmp_path):
    pipeline, tracker = _build(tmp_path, enabled=True)
    result = pipeline.process(_FRAME, 0)
    assert tracker.calls == 1
    assert result.detections[0].track_id == 999  # went through the spy tracker


def test_tracker_disabled_bypasses_update_entirely(tmp_path):
    pipeline, tracker = _build(tmp_path, enabled=False)
    result = pipeline.process(_FRAME, 0)
    assert tracker.calls == 0
    assert result.detections[0].track_id is None  # raw detection, untouched
    assert result.detections[0].bbox_xyxy == _CANNED[0].bbox_xyxy

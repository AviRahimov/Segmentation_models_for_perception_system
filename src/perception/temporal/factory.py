"""Factories for temporal components."""
from __future__ import annotations

from ..config.schema import TemporalCfg
from .base import InstanceTracker, LogitsSmoother, SceneCutDetector
from .ema_logits import LogitsEMA
from .iou_tracker import IoUInstanceTracker
from .scene_cut import HistogramSceneCutDetector


def build_logits_smoother(cfg: TemporalCfg) -> LogitsSmoother:
    return LogitsEMA(alpha=cfg.semantic_ema.alpha)


def build_scene_cut_detector(cfg: TemporalCfg) -> SceneCutDetector:
    return HistogramSceneCutDetector(threshold=cfg.semantic_ema.scene_cut_threshold)


def build_instance_tracker(
    _cfg: TemporalCfg,
    *,
    device: str = "cuda",
) -> InstanceTracker:
    """Associate instance detections across frames via mask/box IoU."""
    del device  # API stability with PerceptionPipeline; IoU tracker ignores device.
    return IoUInstanceTracker()

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
    cfg: TemporalCfg,
    *,
    device: str = "cuda",
) -> InstanceTracker:
    """Associate instance detections across frames via mask/box IoU."""
    del device  # API stability with PerceptionPipeline; IoU tracker ignores device.
    tc = cfg.instance_tracker
    return IoUInstanceTracker(
        iou_threshold=tc.iou_threshold,
        max_hold_frames=tc.max_hold_frames,
        hold_score_decay=tc.hold_score_decay,
        bbox_alpha=tc.bbox_alpha,
        score_alpha=tc.score_alpha,
    )

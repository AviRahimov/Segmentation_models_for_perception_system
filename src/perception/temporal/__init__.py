"""Causal temporal smoothing: logit EMA, scene-cut detection, IoU tracking."""
from .base import InstanceTracker, LogitsSmoother, SceneCutDetector
from .ema_logits import LogitsEMA
from .iou_tracker import IoUInstanceTracker
from .scene_cut import HistogramSceneCutDetector
from .factory import (
    build_instance_tracker,
    build_logits_smoother,
    build_scene_cut_detector,
)

__all__ = [
    "InstanceTracker",
    "LogitsSmoother",
    "SceneCutDetector",
    "LogitsEMA",
    "IoUInstanceTracker",
    "HistogramSceneCutDetector",
    "build_instance_tracker",
    "build_logits_smoother",
    "build_scene_cut_detector",
]

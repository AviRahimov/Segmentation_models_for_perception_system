"""Causal temporal smoothing: logit EMA, scene-cut detection, SAM2 tracking."""
from .base import InstanceTracker, LogitsSmoother, SceneCutDetector
from .ema_logits import LogitsEMA
from .iou_tracker import IoUInstanceTracker
from .sam2_tracker import SAM2InstanceTracker, is_sam2_available
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
    "SAM2InstanceTracker",
    "is_sam2_available",
    "HistogramSceneCutDetector",
    "build_instance_tracker",
    "build_logits_smoother",
    "build_scene_cut_detector",
]

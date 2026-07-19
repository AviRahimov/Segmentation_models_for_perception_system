"""Factories for temporal components."""
from __future__ import annotations

from ..config.schema import TemporalCfg
from .base import InstanceTracker, LogitsSmoother, SceneCutDetector
from .bytetrack_tracker import ByteTrackInstanceTracker
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
    """Associate instance detections across frames — "iou" (this project's
    custom greedy/Hungarian tracker) or "bytetrack" (roboflow/trackers'
    ByteTrackTracker), selected via cfg.instance_tracker.backend."""
    del device  # API stability with PerceptionPipeline; both backends ignore device.
    tc = cfg.instance_tracker
    if tc.backend == "bytetrack":
        return ByteTrackInstanceTracker(
            lost_track_buffer=tc.max_hold_frames,
            frame_rate=tc.frame_rate,
            minimum_consecutive_frames=tc.min_hits,
            minimum_iou_threshold=tc.iou_threshold,
            hold_score_decay=tc.hold_score_decay,
        )
    return IoUInstanceTracker(
        iou_threshold=tc.iou_threshold,
        max_hold_frames=tc.max_hold_frames,
        hold_score_decay=tc.hold_score_decay,
        bbox_alpha=tc.bbox_alpha,
        score_alpha=tc.score_alpha,
        use_hungarian_matching=tc.use_hungarian_matching,
        min_hits=tc.min_hits,
    )

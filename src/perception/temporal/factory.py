"""Factories for temporal components."""
from __future__ import annotations

import logging

from ..config.schema import TemporalCfg
from .base import InstanceTracker, LogitsSmoother, SceneCutDetector
from .ema_logits import LogitsEMA
from .iou_tracker import IoUInstanceTracker
from .sam2_tracker import SAM2InstanceTracker, is_sam2_available
from .scene_cut import HistogramSceneCutDetector

logger = logging.getLogger(__name__)


def build_logits_smoother(cfg: TemporalCfg) -> LogitsSmoother:
    return LogitsEMA(alpha=cfg.semantic_ema.alpha)


def build_scene_cut_detector(cfg: TemporalCfg) -> SceneCutDetector:
    return HistogramSceneCutDetector(threshold=cfg.semantic_ema.scene_cut_threshold)


def build_instance_tracker(
    cfg: TemporalCfg,
    *,
    device: str = "cuda",
) -> InstanceTracker:
    """Return SAM2 tracker if all preconditions are met, otherwise IoU."""
    sam2 = cfg.instance_sam2
    want_sam2 = sam2.enabled and bool(sam2.checkpoint) and bool(sam2.model_config)
    if want_sam2 and is_sam2_available():
        try:
            return SAM2InstanceTracker(
                ckpt=sam2.checkpoint,
                model_cfg=sam2.model_config,
                reprompt_every_n_frames=sam2.reprompt_every_n_frames,
                min_track_score=sam2.min_track_score,
                device=device,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "SAM2 tracker requested but initialisation failed (%s); "
                "falling back to IoU tracker.",
                e,
            )
    elif want_sam2 and not is_sam2_available():
        logger.warning(
            "SAM2 tracker requested but the 'sam2' package is not installed; "
            "falling back to IoU tracker."
        )
    elif sam2.enabled and not (sam2.checkpoint and sam2.model_config):
        logger.info(
            "instance_sam2.enabled=true but checkpoint/model_config not set; "
            "using IoU tracker."
        )
    return IoUInstanceTracker()

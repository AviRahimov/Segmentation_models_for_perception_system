"""High-level perception pipeline.

A pure orchestrator: it owns no models, no Qt widgets, and no IO. All
collaborators are injected through the constructor. Calling code can swap
any single collaborator (model, smoother, tracker, scene-cut detector)
without touching this file.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from ..config.schema import AppConfig
from ..core.types import FrameResult, SemanticPrediction
from ..models.instance.base import InstanceModel
from ..models.semantic.base import SemanticModel
from ..temporal.base import InstanceTracker, LogitsSmoother, SceneCutDetector

logger = logging.getLogger(__name__)


class PerceptionPipeline:
    def __init__(
        self,
        instance_model: InstanceModel,
        semantic_model: SemanticModel,
        instance_tracker: InstanceTracker,
        logits_smoother: LogitsSmoother,
        scene_cut_detector: SceneCutDetector,
        config: AppConfig,
    ) -> None:
        self._inst = instance_model
        self._sem = semantic_model
        self._tracker = instance_tracker
        self._smoother = logits_smoother
        self._scene_cut = scene_cut_detector
        self._cfg = config
        self._has_semantic = bool(config.semantic_classes)
        self._has_instance = bool(config.instance_classes)
        self._reset_on_cut = config.temporal.semantic_ema.reset_on_scene_cut

    # ------------------------------------------------------------------ #
    def warmup(self) -> None:
        """Run one-time setup on both models. Call once at startup."""
        self._inst.warmup(self._cfg.classes)
        self._sem.warmup(self._cfg.classes)
        logger.info(
            "Pipeline warmed: %d instance classes, %d semantic classes",
            len(self._cfg.instance_classes),
            len(self._cfg.semantic_classes),
        )

    # ------------------------------------------------------------------ #
    def process(self, frame_bgr: np.ndarray, frame_idx: int) -> FrameResult:
        t0 = time.perf_counter()

        # 1. Scene cut → reset all temporal state when configured.
        cut = self._scene_cut.update(frame_bgr)
        if cut and self._reset_on_cut:
            logger.debug("Scene cut at frame %d - resetting temporal buffers", frame_idx)
            self._smoother.reset()
            self._tracker.reset()

        # 2. Instance detection + tracking.
        detections = []
        if self._has_instance:
            raw = self._inst.predict(frame_bgr)
            detections = self._tracker.update(frame_bgr, raw)

        # 3. Semantic logits + causal EMA.
        sem_pred: SemanticPrediction | None = None
        if self._has_semantic:
            logits = self._sem.predict_logits(frame_bgr)
            smoothed = self._smoother.update(logits)
            sem_pred = SemanticPrediction(
                logits=smoothed,
                class_names=self._sem.class_names,
            )

        dt_ms = (time.perf_counter() - t0) * 1000.0
        return FrameResult(
            frame_bgr=frame_bgr,
            detections=detections,
            semantic=sem_pred,
            frame_idx=frame_idx,
            inference_ms=dt_ms,
            scene_cut=cut,
        )

    # ------------------------------------------------------------------ #
    def reset_temporal(self) -> None:
        """Drop all temporal buffers. Call after seek or stream restart."""
        self._smoother.reset()
        self._tracker.reset()
        self._scene_cut.reset()


# --------------------------------------------------------------------------- #
# Convenience builder (DI wiring) used by both the GUI and headless scripts. #
# --------------------------------------------------------------------------- #
def build_pipeline(cfg: AppConfig) -> PerceptionPipeline:
    """Wire the whole graph from a single config object.

    This helper exists purely to keep the ``run_player.py`` and
    ``run_headless.py`` entry points short; the pipeline class itself
    remains DI-driven and unaware of these factories.
    """
    from ..models.backends.factory import build_backend
    from ..models.factory import build_instance_model, build_semantic_model
    from ..temporal.factory import (
        build_instance_tracker,
        build_logits_smoother,
        build_scene_cut_detector,
    )

    backend = build_backend(cfg.hardware.use_tensorrt)
    instance_model = build_instance_model(cfg.models.instance, cfg.hardware, backend)
    semantic_model = build_semantic_model(cfg.models.semantic, cfg.hardware, backend)
    smoother = build_logits_smoother(cfg.temporal)
    cut = build_scene_cut_detector(cfg.temporal)
    tracker = build_instance_tracker(cfg.temporal, device=cfg.hardware.device)
    return PerceptionPipeline(
        instance_model=instance_model,
        semantic_model=semantic_model,
        instance_tracker=tracker,
        logits_smoother=smoother,
        scene_cut_detector=cut,
        config=cfg,
    )

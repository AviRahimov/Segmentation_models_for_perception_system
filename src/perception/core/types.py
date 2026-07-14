"""Pure dataclasses shared between modules.

These types form the wire protocol between the inference pipeline, the
temporal smoothers, and the renderer. They deliberately avoid any model-
or UI-specific imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


@dataclass
class Detection:
    """A single instance detection.

    Attributes:
        class_name: Must match a :class:`ClassDef.name` from the config.
        score:      Confidence in [0, 1].
        bbox_xyxy:  Pixel coordinates ``(x1, y1, x2, y2)`` in original frame.
        mask:       Optional binary uint8 mask, shape (H, W), values 0/1.
        track_id:   Integer track identifier assigned by the InstanceTracker;
                    ``None`` if no tracker is in use.
        display_threshold: The per-class confidence threshold that gated this
                    box at the model wrapper. ``None`` when not populated by
                    the wrapper. Used by the tracker's low-confidence-recovery
                    step to tell a normal, confirmable detection
                    (``score >= display_threshold``) apart from a
                    recovery-only one (``score < display_threshold``, which
                    can extend an already-confirmed track but never create
                    or confirm one).
    """

    class_name: str
    score: float
    bbox_xyxy: tuple[int, int, int, int]
    mask: Optional[np.ndarray] = None
    track_id: Optional[int] = None
    display_threshold: Optional[float] = None


@dataclass
class SemanticPrediction:
    """Output of the semantic model after temporal smoothing.

    Attributes:
        logits:      Tensor of shape ``(C_user, H, W)``, smoothed by EMA.
                     Argmax is the renderer's responsibility.
        class_names: Tuple of user class names parallel to dim 0 of ``logits``.
    """

    logits: torch.Tensor
    class_names: tuple[str, ...]


@dataclass
class FrameResult:
    """Per-frame inference output passed from pipeline to renderer/UI."""

    frame_bgr: np.ndarray
    detections: list[Detection] = field(default_factory=list)
    semantic: Optional[SemanticPrediction] = None
    frame_idx: int = 0
    inference_ms: float = 0.0
    scene_cut: bool = False

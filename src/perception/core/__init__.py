"""Pure data contracts shared across all sub-packages.

Modules under :mod:`perception.core` must not import anything from other
``perception`` sub-packages: they sit at the bottom of the dependency graph.
"""
from .types import Detection, FrameResult, SemanticPrediction
from .color import bgr_for, rgb_to_bgr
from .geometry import iou_xyxy

__all__ = [
    "Detection",
    "FrameResult",
    "SemanticPrediction",
    "bgr_for",
    "rgb_to_bgr",
    "iou_xyxy",
]

"""Semantic segmentation models (terrain etc)."""
from .base import SemanticModel
from .segformer import SegFormerSemanticModel

__all__ = ["SemanticModel", "SegFormerSemanticModel"]

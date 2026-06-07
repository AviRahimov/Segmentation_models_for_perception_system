"""Backward-compatibility shim — YOLOEInstanceModel moved to yolo/open.py."""
from .yolo.open import YOLOEInstanceModel  # noqa: F401

__all__ = ["YOLOEInstanceModel"]

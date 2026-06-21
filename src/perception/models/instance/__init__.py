"""Instance segmentation models (open-vocabulary)."""
from .base import InstanceModel
from .yolo.open import YOLOEInstanceModel

__all__ = ["InstanceModel", "YOLOEInstanceModel"]

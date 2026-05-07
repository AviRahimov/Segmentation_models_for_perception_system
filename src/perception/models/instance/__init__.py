"""Instance segmentation models (open-vocabulary)."""
from .base import InstanceModel
from .yoloe import YOLOEInstanceModel

__all__ = ["InstanceModel", "YOLOEInstanceModel"]

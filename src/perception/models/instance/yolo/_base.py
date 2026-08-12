"""Shared Ultralytics helper used by the closed-vocab wrapper."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _load_ultralytics_model(weights: str) -> Any:
    """Return an Ultralytics YOLO model for *weights* (YOLO11/12/26)."""
    from ultralytics import YOLO
    return YOLO(weights)

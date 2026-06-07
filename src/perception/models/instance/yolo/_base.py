"""Shared Ultralytics helpers used by both the open- and closed-vocab wrappers."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _load_ultralytics_model(weights: str) -> Any:
    """Return an Ultralytics model for *weights*, regardless of YOLOE class location.

    Tries three import paths in order to handle API shifts across Ultralytics
    releases.  Works for YOLOE checkpoints and all standard YOLO11/12/26 weights.
    """
    from .._ultralytics_compat import apply_patches

    apply_patches()

    last_exc: Exception | None = None

    try:
        from ultralytics import YOLOE as _Cls  # type: ignore
        logger.debug("Loading model via ultralytics.YOLOE")
        return _Cls(weights)
    except (ImportError, Exception) as e:
        last_exc = e
        logger.debug("ultralytics.YOLOE not available: %s", e)

    try:
        from ultralytics.models.yoloe import YOLOE as _Cls  # type: ignore
        logger.debug("Loading model via ultralytics.models.yoloe.YOLOE")
        return _Cls(weights)
    except (ImportError, Exception) as e:
        last_exc = e
        logger.debug("ultralytics.models.yoloe.YOLOE not available: %s", e)

    try:
        from ultralytics import YOLO as _Cls  # type: ignore
        logger.debug("Loading model via ultralytics.YOLO")
        return _Cls(weights)
    except ImportError as e:
        last_exc = e

    raise RuntimeError(
        "Failed to load any Ultralytics YOLOE-compatible class. "
        "Last error: " + repr(last_exc)
    )


def _apply_patches() -> None:
    """Apply Ultralytics FP16 compat patches (idempotent)."""
    from .._ultralytics_compat import apply_patches
    apply_patches()

"""PyTorch backend (default). Eager execution on the requested device."""
from __future__ import annotations

import logging
from typing import Any

from .base import InferenceBackend

logger = logging.getLogger(__name__)


class PyTorchBackend(InferenceBackend):
    name = "pytorch"

    def is_available(self) -> bool:
        return True

    def prepare(self, model: Any, *, device: str, fp16: bool) -> Any:
        if model is None:
            return None
        if hasattr(model, "to"):
            try:
                model = model.to(device)
            except Exception as e:  # noqa: BLE001
                logger.warning("PyTorchBackend: model.to(%s) failed: %s", device, e)
        if fp16 and isinstance(device, str) and device.startswith("cuda") and hasattr(model, "half"):
            try:
                model = model.half()
            except Exception as e:  # noqa: BLE001
                logger.warning("PyTorchBackend: model.half() failed: %s", e)
        if hasattr(model, "eval"):
            try:
                model.eval()
            except Exception as e:  # noqa: BLE001
                logger.debug("PyTorchBackend: model.eval() not applicable: %s", e)
        return model

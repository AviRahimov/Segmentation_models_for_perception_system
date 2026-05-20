"""Inference backend abstraction.

Concrete subclasses adapt a high-level model object so that the same
user-facing wrapper (YOLOE, SegFormer) works regardless of whether the
underlying execution is PyTorch eager mode or a compiled TensorRT engine.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class InferenceBackend(ABC):
    name: str = "abstract"

    @abstractmethod
    def is_available(self) -> bool:
        """Returns ``True`` if this backend can be used in the current env."""

    @abstractmethod
    def prepare(self, model: Any, *, device: str, fp16: bool, engine_path: str = "") -> Any:
        """Move/compile ``model`` for execution. May return a wrapped object.

        ``engine_path`` is consumed by :class:`TensorRTBackend` (path to a
        pre-built ``.engine`` file) and ignored by all other backends.
        """

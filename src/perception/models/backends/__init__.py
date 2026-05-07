"""Inference-backend abstraction (PyTorch is default; TensorRT is optional)."""
from .base import InferenceBackend
from .pytorch import PyTorchBackend
from .tensorrt import TensorRTBackend
from .factory import build_backend

__all__ = ["InferenceBackend", "PyTorchBackend", "TensorRTBackend", "build_backend"]

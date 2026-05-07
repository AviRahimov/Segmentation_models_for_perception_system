"""Backend factory."""
from __future__ import annotations

from .base import InferenceBackend
from .pytorch import PyTorchBackend
from .tensorrt import TensorRTBackend


def build_backend(use_tensorrt: bool) -> InferenceBackend:
    """Build the inference backend selected by ``hardware.use_tensorrt``.

    Falls back to PyTorch with a clear error if TensorRT is requested but
    unavailable, rather than silently downgrading.
    """
    if use_tensorrt:
        be = TensorRTBackend()
        if not be.is_available():
            raise RuntimeError(
                "hardware.use_tensorrt=true but the 'tensorrt' Python package is not "
                "installed. Install it (Jetson: bundled with JetPack; x86: NVIDIA wheel) "
                "and complete the integration steps documented in "
                "perception.models.backends.tensorrt."
            )
        return be
    return PyTorchBackend()

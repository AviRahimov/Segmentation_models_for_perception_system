"""TensorRT FP16 inference backend for SegFormer.

At inference time this module only *loads* a pre-built engine from disk and
wraps it behind the same interface that ``SegFormerSemanticModel`` expects.
No compilation happens at startup; the one-time build step is handled by
``scripts/export_trt.py``.

YOLOE uses TRT through Ultralytics' own engine loader — just point
``models.instance.weights`` to a ``.engine`` file and it works automatically.
This backend is therefore only used for the semantic model (SegFormer / DDRNet).

Requirements
------------
* JetPack 6.x ships TensorRT 8.6 — the ``set_tensor_address`` /
  ``execute_async_v3`` API used here requires TRT >= 8.5.
* The engine file must be built on the same Jetson (GPU + driver + TRT version
  are baked into the serialised engine). Re-run ``export_trt.py`` after any
  JetPack upgrade.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from .base import InferenceBackend

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helper types that mimic the HuggingFace model interface                      #
# --------------------------------------------------------------------------- #

class _TRTModelConfig:
    """Minimal config object so ``warmup()`` can read ``model.config.num_labels``."""
    def __init__(self, num_labels: int) -> None:
        self.num_labels = num_labels


class _TRTOutput:
    """Mimics HF ``SemanticSegmenterOutput`` so ``predict_logits()`` is unchanged."""
    __slots__ = ("logits",)

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _TRTSegFormerWrapper:
    """Wraps a TRT execution context behind the HuggingFace model call interface.

    After ``TensorRTBackend.prepare()`` stores an instance of this class as
    ``SegFormerSemanticModel._model``, the rest of the model wrapper (warmup,
    predict_logits, LUT merge) runs completely unchanged.
    """

    def __init__(
        self,
        engine: Any,
        num_labels: int,
        device: str,
        fp16: bool,
    ) -> None:
        self.config = _TRTModelConfig(num_labels=num_labels)
        self._context = engine.create_execution_context()
        self._device = device
        self._dtype = torch.float16 if fp16 else torch.float32

        # Pre-allocate a persistent output buffer using the engine's static shape.
        # TRT static engines report the exact output shape from get_tensor_shape.
        out_shape = tuple(self._context.get_tensor_shape("logits"))
        self._out_buf: torch.Tensor = torch.empty(
            out_shape, dtype=self._dtype, device=device
        )

    # ---------------------------------------------------------------------- #
    def __call__(self, *, pixel_values: torch.Tensor) -> _TRTOutput:
        """Run one TRT inference pass.

        ``pixel_values`` must already be on the correct CUDA device and in the
        dtype the engine was compiled for (FP16 or FP32).  We hand its raw
        CUDA pointer directly to TRT — no copies.
        """
        stream = torch.cuda.current_stream().cuda_stream
        self._context.set_tensor_address("pixel_values", pixel_values.data_ptr())
        self._context.set_tensor_address("logits", self._out_buf.data_ptr())
        self._context.execute_async_v3(stream)
        torch.cuda.current_stream().synchronize()
        # Clone so the caller has exclusive ownership of the tensor.
        return _TRTOutput(self._out_buf.clone())


# --------------------------------------------------------------------------- #
# Backend                                                                       #
# --------------------------------------------------------------------------- #

class TensorRTBackend(InferenceBackend):
    name = "tensorrt"

    def __init__(self) -> None:
        self._trt_module = None
        try:
            import tensorrt as trt  # type: ignore
            self._trt_module = trt
        except Exception as e:  # noqa: BLE001
            logger.debug("tensorrt unavailable: %s", e)

    def is_available(self) -> bool:
        return self._trt_module is not None

    def prepare(
        self,
        model: Any,
        *,
        device: str,
        fp16: bool,
        engine_path: str = "",
    ) -> Any:
        """Load a pre-built TRT engine and return a wrapper that mimics the HF model.

        Parameters
        ----------
        model:
            The PyTorch model that would otherwise be used.  Its
            ``config.num_labels`` is read before it is discarded, so the
            downstream warmup() call can still validate the class count.
        engine_path:
            Path to the ``.engine`` file produced by ``scripts/export_trt.py``.
            Raises clearly if absent or empty.
        """
        if not engine_path:
            raise RuntimeError(
                "TensorRT backend requires models.semantic.trt_engine_path in "
                "config.yaml.  Run  python scripts/export_trt.py --config "
                "config/config.yaml  first, then set the printed engine path."
            )
        ep = Path(engine_path)
        if not ep.exists():
            raise FileNotFoundError(
                f"TRT engine not found: {engine_path}\n"
                "Run: python scripts/export_trt.py --config config/config.yaml "
                "--model segformer"
            )

        # Capture num_labels from the PyTorch model before it is replaced.
        num_labels = int(
            getattr(getattr(model, "config", None), "num_labels", 3)
        )

        trt = self._trt_module
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(ep.read_bytes())
        logger.info(
            "TensorRT engine loaded: %s  (%d output classes, device=%s, fp16=%s)",
            engine_path, num_labels, device, fp16,
        )
        return _TRTSegFormerWrapper(
            engine, num_labels=num_labels, device=device, fp16=fp16
        )

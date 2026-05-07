"""TensorRT backend - structured stub.

This module intentionally does *not* depend on ``tensorrt`` at import time
so the rest of the project remains usable on developer laptops without the
TensorRT runtime. To enable TensorRT acceleration on a production target
(e.g. Jetson AGX Orin 64GB), a future engineer must implement the four
steps documented below; nothing else in the codebase needs to change.

Integration steps
-----------------

1. **Export the underlying PyTorch model to ONNX.**

   For Hugging Face SegFormer::

       import torch
       from transformers import SegformerForSemanticSegmentation

       model = SegformerForSemanticSegmentation.from_pretrained(
           "nvidia/segformer-b2-finetuned-ade-512-512"
       ).eval().cuda().half()
       sample = torch.randn(1, 3, 512, 512, device="cuda", dtype=torch.half)
       torch.onnx.export(
           model, sample, "segformer_b2.onnx",
           input_names=["pixel_values"],
           output_names=["logits"],
           dynamic_axes={
               "pixel_values": {0: "B", 2: "H", 3: "W"},
               "logits":       {0: "B", 2: "h", 3: "w"},
           },
           opset_version=17,
       )

   For Ultralytics YOLOE the canonical TensorRT path does NOT go through
   this backend's ``prepare`` hook (Ultralytics has its own engine
   loader). Instead, export and load a ``.engine`` file directly::

       from ultralytics import YOLOE
       model = YOLOE("yoloe-26l-seg.pt")
       model.export(format="engine", imgsz=640, half=True, dynamic=True,
                    workspace=4)
       # Then in config.yaml set:
       #   models.instance.weights: "weights/yoloe-26l-seg.engine"
       # and the YOLOEInstanceModel will pick it up unchanged.

2. **Build a TensorRT engine from the ONNX file.** Implement
   :meth:`TensorRTBackend._build_engine`::

       import tensorrt as trt
       logger = trt.Logger(trt.Logger.WARNING)
       builder = trt.Builder(logger)
       network = builder.create_network(
           1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
       )
       parser = trt.OnnxParser(network, logger)
       with open(onnx_path, "rb") as f:
           parser.parse(f.read())
       config = builder.create_builder_config()
       config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
       config.set_flag(trt.BuilderFlag.FP16)
       engine_bytes = builder.build_serialized_network(network, config)
       Path(engine_path).write_bytes(engine_bytes)

3. **Load the engine and allocate device buffers.** Implement
   :meth:`TensorRTBackend._load_engine`. Allocate cuda buffers for every
   binding (``engine.num_bindings`` x ``engine.get_binding_shape``). Cache
   ``context`` and the binding tensor list on ``self``.

4. **Replace the underlying model's forward.** In :meth:`prepare`, monkey-
   patch ``model.forward`` (PyTorch) or ``model.predict`` (Ultralytics) with
   a thin function that:
       a. copies ``pixel_values`` into the device input buffer,
       b. ``context.execute_async_v3(stream)`` then synchronizes,
       c. wraps the device output buffer as a ``torch.Tensor`` of the same
          shape and dtype the original PyTorch model produced.

   Because the *shape and dtype* match, no downstream code (post-processing,
   smoothing, rendering) needs to be aware of the swap.

The :class:`PerceptionPipeline` selects PyTorch by default; users opt into
TensorRT by setting ``hardware.use_tensorrt: true`` in ``config.yaml``.
"""
from __future__ import annotations

import logging
from typing import Any

from .base import InferenceBackend

logger = logging.getLogger(__name__)


class TensorRTBackend(InferenceBackend):
    name = "tensorrt"

    def __init__(self) -> None:
        self._trt_module = None
        try:
            import tensorrt as trt  # type: ignore  # noqa: F401
            self._trt_module = trt
        except Exception as e:  # noqa: BLE001
            logger.debug("tensorrt unavailable: %s", e)

    def is_available(self) -> bool:
        return self._trt_module is not None

    # --- step 2 hook ------------------------------------------------------ #
    def _build_engine(self, onnx_path: str, engine_path: str) -> None:
        raise NotImplementedError(
            "TensorRTBackend._build_engine is a stub. "
            "Follow step 2 in the module docstring "
            "(see src/perception/models/backends/tensorrt.py)."
        )

    # --- step 3 hook ------------------------------------------------------ #
    def _load_engine(self, engine_path: str) -> Any:
        raise NotImplementedError(
            "TensorRTBackend._load_engine is a stub. "
            "Follow step 3 in the module docstring."
        )

    # --- main entry, called by the model wrappers ------------------------- #
    def prepare(self, model: Any, *, device: str, fp16: bool) -> Any:
        raise NotImplementedError(
            "TensorRTBackend.prepare is a structured stub. To enable TensorRT, "
            "implement the four steps described in "
            "src/perception/models/backends/tensorrt.py — namely _build_engine, "
            "_load_engine, and the forward-replacement logic. The rest of the "
            "perception pipeline does not need to change."
        )

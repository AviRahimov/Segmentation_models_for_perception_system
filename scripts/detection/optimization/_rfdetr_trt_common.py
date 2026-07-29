"""Shared RF-DETR ONNX/TensorRT loading + pre/post-processing.

Standalone by design: only needs ``numpy`` + ``opencv`` + (``torch``+``tensorrt``
for the engine path, or ``onnxruntime`` for the onnx path) — never the
``rfdetr`` training package itself. This is what lets ``benchmark_jetson.py``
run on the Jetson's minimal optimization venv (no full repo checkout, no
``rfdetr`` package) while still being importable from the dev PC's main venv
for ``compare_models.py``.

Pre/post-processing mirrors ``rfdetr``'s own ``predict()`` /
``export._onnx.inference`` (ImageNet-normalized bilinear resize to a fixed
square input, per-query sigmoid + argmax over the real classes, dropping the
model's no-object logit slot, cxcywh -> pixel-space xyxy) — verified against
the real PyTorch reference in ``export_onnx.py``'s validation step (mean
IoU 0.989, mean confidence diff 0.004 over 20 real images). It is not bit-
exact with the production ``RFDETRInstanceModel`` wrapper (that uses
torchvision's tensor-based resize; this uses OpenCV's) — close enough for
benchmarking/comparison, not intended for the live pipeline. Wiring a
validated engine into production is a deliberate follow-up, not this module's
job.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Detection(NamedTuple):
    class_id: int
    score: float
    xyxy: tuple[float, float, float, float]


# --------------------------------------------------------------------------- #
# Pre/post-processing                                                          #
# --------------------------------------------------------------------------- #

def preprocess_bgr(frame_bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """BGR HxWx3 uint8 -> normalized (1,3,H,W) float32, NCHW."""
    import cv2

    h, w = size
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    arr = arr.transpose(2, 0, 1)[np.newaxis]
    return np.ascontiguousarray(arr, dtype=np.float32)


def decode(
    boxes_cxcywh: np.ndarray, logits: np.ndarray, orig_w: int, orig_h: int, threshold: float,
) -> list[Detection]:
    """``boxes_cxcywh``: (Q,4) normalized [0,1]. ``logits``: (Q, num_classes+1) raw.

    Drops the trailing no-object slot, takes per-query argmax over the real
    classes (mirrors ``rfdetr.export._onnx.inference._run_inference``).
    """
    real_logits = logits[:, :-1]
    one = np.asarray(1.0, dtype=real_logits.dtype)
    scores_all = one / (one + np.exp(-real_logits.clip(-88, 88)))
    scores = scores_all.max(axis=-1)
    cls = scores_all.argmax(axis=-1)
    keep = scores > threshold

    cx, cy, bw, bh = boxes_cxcywh[keep].T
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    xyxy *= np.array([orig_w, orig_h, orig_w, orig_h], dtype=np.float32)

    kept_scores = scores[keep]
    kept_cls = cls[keep]
    return [
        Detection(int(kept_cls[i]), float(kept_scores[i]), tuple(float(v) for v in xyxy[i]))
        for i in range(len(kept_scores))
    ]


# --------------------------------------------------------------------------- #
# ONNX Runtime backend                                                         #
# --------------------------------------------------------------------------- #

class RFDETROnnxModel:
    """Runs a ``.onnx`` export through ONNX Runtime."""

    def __init__(self, onnx_path: str | Path, input_size: tuple[int, int] | None = None,
                 providers: list[str] | None = None) -> None:
        import onnxruntime as ort

        providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(onnx_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        out_names = [o.name for o in self._session.get_outputs()]
        self._boxes_idx = out_names.index("dets") if "dets" in out_names else 0
        self._logits_idx = out_names.index("labels") if "labels" in out_names else 1

        # Auto-detect from the ONNX graph's own declared input shape rather than
        # trusting a caller-supplied default — see RFDETRTensorRTEngine's docstring
        # for why a mismatched size silently produces garbage output instead of
        # erroring (a real bug found this way for rfdetr-s, 512x512, vs the 576x576
        # default meant for rfdetr-m).
        onnx_shape = self._session.get_inputs()[0].shape  # e.g. [1, 3, 512, 512] or ['batch', 3, 'h', 'w']
        detected = None
        if len(onnx_shape) == 4 and isinstance(onnx_shape[2], int) and isinstance(onnx_shape[3], int):
            detected = (onnx_shape[2], onnx_shape[3])
        if detected is not None:
            if input_size is not None and tuple(input_size) != detected:
                logger.warning(
                    "Requested input_size=%s does not match %s's actual ONNX input shape %s "
                    "— using the ONNX graph's real shape.", input_size, onnx_path, detected,
                )
            self._input_size = detected
        elif input_size is not None:
            self._input_size = tuple(input_size)
        else:
            raise ValueError(
                f"{onnx_path}: ONNX graph has a dynamic input shape ({onnx_shape}) and no "
                "input_size was provided to disambiguate it."
            )
        logger.info("ONNX Runtime session ready: %s (providers=%s, input_size=%s)",
                    onnx_path, self._session.get_providers(), self._input_size)

    def infer(self, frame_bgr: np.ndarray, threshold: float = 0.35) -> list[Detection]:
        h0, w0 = frame_bgr.shape[:2]
        arr = preprocess_bgr(frame_bgr, self._input_size)
        raw = self._session.run(None, {self._input_name: arr})
        return decode(raw[self._boxes_idx][0], raw[self._logits_idx][0], w0, h0, threshold)


# --------------------------------------------------------------------------- #
# TensorRT backend                                                              #
# --------------------------------------------------------------------------- #

class RFDETRTensorRTEngine:
    """Thin wrapper around a serialized RF-DETR TensorRT engine.

    Deliberately avoids ``pycuda`` (unlike ``rfdetr.export.benchmark.TRTInference``)
    — uses plain ``tensorrt`` + ``torch`` CUDA tensors instead, matching this
    repo's existing ``src/perception/models/backends/tensorrt.py`` convention.

    Two independent, individually-toggleable optimizations over the naive
    per-frame CPU-preprocess + full-stream-sync loop (Phase 1 measured 33.2 FPS
    against trtexec's own 58 FPS on the identical engine — pure harness
    overhead, not an engine problem):

    ``gpu_preprocess`` — resize/normalize on the GPU instead of cv2/numpy on
    the CPU. Uploads raw uint8 BGR via a pinned staging buffer (Jetson's
    unified memory makes *mapped* pinned transfers a real win, not the
    marginal one they'd be on a discrete-GPU desktop), casts to float
    on-device (CUDA bilinear interpolate does not support uint8 — verified
    empirically, not assumed), and folds the BGR->RGB channel swap directly
    into the ImageNet mean/std ordering instead of a separate flip op.

    ``cuda_graph`` — captures the TensorRT ``execute_async_v3`` call once
    (after a throwaway warmup *outside* capture, per the pattern confirmed
    against RF-DETR's own authors' ``roboflow/inference`` and NVIDIA's
    ``jetson-ai-lab`` reference code) and replays it every frame, eliminating
    per-frame kernel-launch overhead. TensorRT/TensorRT#2603 reports
    ``execute_async_v3`` failing *inside* capture on TRT 8.5.2.2 with no
    confirmed fix version — our TRT is 8.6.2.3, unconfirmed either way, so
    capture is attempted and wrapped in try/except: on failure this logs a
    clear warning and falls back to plain (non-graph) async execution rather
    than silently pretending to be optimized.
    """

    def __init__(self, engine_path: str | Path, input_size: tuple[int, int] | None = None,
                 gpu_preprocess: bool = True, cuda_graph: bool = True) -> None:
        import tensorrt as trt
        import torch

        self._torch = torch
        self._gpu_preprocess = gpu_preprocess
        self._graph = None  # set below if capture succeeds

        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self._context = self._engine.create_execution_context()

        self._input_name: str | None = None
        self._output_names: list[str] = []
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._input_name = name
            else:
                self._output_names.append(name)
        if self._input_name is None or not self._output_names:
            raise RuntimeError(f"Engine has no recognizable input/output tensors: {engine_path}")

        # Auto-detect from the engine's own binding shape rather than trusting a
        # caller-supplied default — different RF-DETR variants use different fixed
        # input resolutions (rfdetr-s=512, rfdetr-m=576) and a caller passing the
        # wrong size doesn't error, it silently feeds a misinterpreted buffer into
        # a static-shape engine (wrong strides, no shape-mismatch check anywhere),
        # producing garbage output that never crosses the confidence threshold —
        # exactly how a real rfdetr-s "0 detections in every frame" bug looked
        # before this fix (found via compare_models.py hardcoding 576x576 for
        # every engine: spec regardless of which model it actually was).
        engine_shape = tuple(self._engine.get_tensor_shape(self._input_name))
        if len(engine_shape) == 4 and engine_shape[2] > 0 and engine_shape[3] > 0:
            detected = (engine_shape[2], engine_shape[3])
            if input_size is not None and tuple(input_size) != detected:
                logger.warning(
                    "Requested input_size=%s does not match %s's actual engine "
                    "input shape %s — using the engine's real shape.",
                    input_size, engine_path, detected,
                )
            self._input_size = detected
        elif input_size is not None:
            self._input_size = tuple(input_size)
        else:
            raise ValueError(
                f"{engine_path}: engine has a dynamic input shape ({engine_shape}) and no "
                "input_size was provided to disambiguate it."
            )

        def _torch_dtype(name: str):
            np_dtype = trt.nptype(self._engine.get_tensor_dtype(name))
            return torch.from_numpy(np.zeros(1, dtype=np_dtype)).dtype

        self._in_dtype = _torch_dtype(self._input_name)
        self._in_buf = torch.empty((1, 3, *self._input_size), dtype=self._in_dtype, device="cuda")

        self._out_bufs: dict[str, torch.Tensor] = {}
        for name in self._output_names:
            shape = tuple(self._context.get_tensor_shape(name))
            self._out_bufs[name] = torch.empty(shape, dtype=_torch_dtype(name), device="cuda")

        self._dets_name = next((n for n in self._output_names if "dets" in n), self._output_names[0])
        self._labels_name = next((n for n in self._output_names if "labels" in n), self._output_names[1])
        logger.info("TensorRT engine loaded: %s (input=%s dtype=%s, outputs=%s, gpu_preprocess=%s, cuda_graph=%s)",
                    engine_path, self._input_name, self._in_dtype, self._output_names,
                    gpu_preprocess, cuda_graph)

        # Tensor addresses are fixed for the lifetime of these persistent buffers —
        # bind once here rather than every frame (required anyway for CUDA graph
        # capture, where addresses must be baked in outside the captured region).
        self._context.set_tensor_address(self._input_name, self._in_buf.data_ptr())
        for name, buf in self._out_bufs.items():
            self._context.set_tensor_address(name, buf.data_ptr())

        # Reused every frame — allocating a fresh torch.cuda.Event() per call costs a
        # real cudaEventCreate() each time; that's on the order of the very overhead
        # we're trying to eliminate.
        self._done_event = torch.cuda.Event()

        if gpu_preprocess:
            self._pinned_stage: "torch.Tensor | None" = None
            # BGR-ordered ImageNet mean/std (reversed from the usual RGB order) —
            # normalizing directly in BGR is mathematically identical to an
            # explicit BGR->RGB flip followed by RGB normalize, without the
            # extra materialized copy the flip would cost.
            self._mean_bgr = torch.tensor(_IMAGENET_MEAN[::-1].copy(), device="cuda").view(1, 3, 1, 1)
            self._std_bgr = torch.tensor(_IMAGENET_STD[::-1].copy(), device="cuda").view(1, 3, 1, 1)

        self._warmup_and_maybe_capture(cuda_graph)

    # ------------------------------------------------------------------ #

    def _execute_async(self) -> None:
        stream = self._torch.cuda.current_stream().cuda_stream
        self._context.execute_async_v3(stream)

    def _warmup_and_maybe_capture(self, cuda_graph: bool) -> None:
        torch = self._torch
        self._in_buf.zero_()
        for _ in range(3):  # let TRT's lazy allocation/autotuning settle OUTSIDE capture
            self._execute_async()
        torch.cuda.current_stream().synchronize()

        if not cuda_graph:
            return
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._execute_async()
            torch.cuda.current_stream().synchronize()
            self._graph = g
            logger.info("CUDA graph capture succeeded — steady-state frames will use graph.replay().")
        except Exception as e:  # noqa: BLE001 — deliberately broad: any capture failure must fall back, not crash
            logger.warning(
                "CUDA graph capture failed (%s) — falling back to plain async execution. "
                "Known issue on some TRT versions (NVIDIA/TensorRT#2603); FPS will be lower "
                "than a successful capture would give.", e,
            )
            self._graph = None

    def _preprocess_gpu(self, frame_bgr: np.ndarray) -> "torch.Tensor":
        torch = self._torch
        h, w = frame_bgr.shape[:2]
        if self._pinned_stage is None or tuple(self._pinned_stage.shape[:2]) != (h, w):
            self._pinned_stage = torch.empty((h, w, 3), dtype=torch.uint8, pin_memory=True)
        self._pinned_stage.copy_(torch.from_numpy(frame_bgr))
        gpu_u8 = self._pinned_stage.to("cuda", non_blocking=True)          # H,W,3 uint8 BGR
        gpu_f = gpu_u8.permute(2, 0, 1).unsqueeze(0).float()               # 1,3,H,W BGR — cast BEFORE interpolate
        resized = torch.nn.functional.interpolate(
            gpu_f, size=self._input_size, mode="bilinear", align_corners=False,
        )
        return (resized / 255.0 - self._mean_bgr) / self._std_bgr

    def infer(self, frame_bgr: np.ndarray, threshold: float = 0.35) -> list[Detection]:
        torch = self._torch
        h0, w0 = frame_bgr.shape[:2]

        if self._gpu_preprocess:
            processed = self._preprocess_gpu(frame_bgr)
            self._in_buf.copy_(processed.to(self._in_dtype))
        else:
            arr = preprocess_bgr(frame_bgr, self._input_size)
            self._in_buf.copy_(torch.from_numpy(arr).to(self._in_dtype))

        if self._graph is not None:
            self._graph.replay()
        else:
            self._execute_async()

        self._done_event.record()
        self._done_event.synchronize()

        dets = self._out_bufs[self._dets_name][0].float().cpu().numpy()
        labels = self._out_bufs[self._labels_name][0].float().cpu().numpy()
        return decode(dets, labels, w0, h0, threshold)


# --------------------------------------------------------------------------- #
# PyTorch backend (dev PC only — needs the `rfdetr` package)                  #
# --------------------------------------------------------------------------- #

def _infer_rfdetr_model_name(checkpoint: str | Path) -> str:
    """Guess the RF-DETR variant from a checkpoint path like
    'weights/detection/rfdetr-s/detection_dataset_hardneg/conservative_aug/best.pt'."""
    parts = Path(checkpoint).parts
    for name in ("rfdetr-2xl", "rfdetr-xl", "rfdetr-l", "rfdetr-m", "rfdetr-s", "rfdetr-n"):
        if name in parts:
            return name
    raise ValueError(
        f"Could not infer RF-DETR variant from checkpoint path {checkpoint!r} — "
        "pass model_name explicitly."
    )


class RFDETRPyTorchModel:
    """Wraps a live ``rfdetr`` PyTorch model behind the same ``.infer()`` interface."""

    _VARIANT_CLASSES = {"rfdetr-s": "RFDETRSmall", "rfdetr-m": "RFDETRMedium"}

    def __init__(self, checkpoint: str | Path, model_name: str | None = None) -> None:
        import rfdetr

        model_name = model_name or _infer_rfdetr_model_name(checkpoint)
        cls_name = self._VARIANT_CLASSES.get(model_name)
        if cls_name is None:
            raise ValueError(
                f"Unsupported model_name for pytorch: spec: {model_name!r} "
                f"(supported: {sorted(self._VARIANT_CLASSES)})"
            )
        cls = getattr(rfdetr, cls_name)
        self._model = cls(pretrain_weights=str(checkpoint))

    def infer(self, frame_bgr: np.ndarray, threshold: float = 0.35) -> list[Detection]:
        import cv2
        import PIL.Image

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PIL.Image.fromarray(rgb)
        sv_det = self._model.predict(pil_img, threshold=threshold)
        return [
            Detection(int(sv_det.class_id[i]), float(sv_det.confidence[i]),
                      tuple(float(v) for v in sv_det.xyxy[i]))
            for i in range(len(sv_det))
        ]


# --------------------------------------------------------------------------- #
# Ultralytics backend (YOLO family — .pt/.onnx/.engine all load uniformly      #
# through the same YOLO(path) constructor, unlike RF-DETR's NMS-free output)   #
# --------------------------------------------------------------------------- #

class UltralyticsModel:
    """Wraps an Ultralytics ``YOLO`` object behind the same ``.infer()`` interface
    the RF-DETR wrappers use, so ``compare_models.py``/``_video_bench_common.py``
    can treat any model family uniformly. ``weights`` may be ``.pt``/``.onnx``/
    ``.engine`` — Ultralytics dispatches on the file extension internally."""

    def __init__(self, weights: str | Path, imgsz: int = 640) -> None:
        from ultralytics import YOLO

        self._model = YOLO(str(weights))
        self._imgsz = imgsz

    def infer(self, frame_bgr: np.ndarray, threshold: float = 0.35) -> list[Detection]:
        results = self._model.predict(frame_bgr, conf=threshold, imgsz=self._imgsz, verbose=False)
        r = results[0]
        if r.boxes is None:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy()
        return [
            Detection(int(cls[i]), float(conf[i]), tuple(float(v) for v in xyxy[i]))
            for i in range(len(conf))
        ]


# --------------------------------------------------------------------------- #
# Uniform loader                                                               #
# --------------------------------------------------------------------------- #

def load_model(spec: str, input_size: tuple[int, int] | None = None, **engine_kwargs):
    """Parses a ``pytorch:``/``onnx:``/``engine:``/``ultralytics:`` model spec and loads it.

    Returns an object exposing ``.infer(frame_bgr, threshold) -> list[Detection]``.
    ``engine_kwargs`` (``gpu_preprocess``, ``cuda_graph``) only apply to RF-DETR ``engine:`` specs.
    The first three backends are RF-DETR-specific (NMS-free decode); ``ultralytics:``
    covers YOLO's ``.pt``/``.onnx``/``.engine`` uniformly via Ultralytics' own loader.

    ``input_size`` defaults to ``None`` — both ``onnx:`` and ``engine:`` specs auto-detect
    the real input resolution from the loaded model/engine itself (different RF-DETR
    variants use different fixed sizes: rfdetr-s=512, rfdetr-m=576), so callers mixing
    model sizes in one run (e.g. compare_models.py's N-way comparison) get the correct
    size for each spec automatically without needing to track it themselves.
    """
    if ":" not in spec:
        raise ValueError(
            f"Model spec must be 'pytorch:path', 'onnx:path', 'engine:path', or "
            f"'ultralytics:path'; got: {spec!r}"
        )
    backend, path = spec.split(":", 1)
    if backend == "pytorch":
        return RFDETRPyTorchModel(path)
    if backend == "onnx":
        return RFDETROnnxModel(path, input_size=input_size)
    if backend == "engine":
        return RFDETRTensorRTEngine(path, input_size=input_size, **engine_kwargs)
    if backend == "ultralytics":
        return UltralyticsModel(path)
    raise ValueError(
        f"Unknown model spec backend {backend!r} (want pytorch/onnx/engine/ultralytics): {spec!r}"
    )

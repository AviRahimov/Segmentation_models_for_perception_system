"""Shared SegFormer TensorRT engine loading + pre/post-processing for real
end-to-end (video decode + preprocess + infer) FPS measurement.

Mirrors ``scripts/detection/optimization/_rfdetr_trt_common.py``'s pattern and
lessons: auto-detect the engine's real input shape instead of trusting a
caller-supplied default (the exact bug class that silently broke rfdetr-s
comparisons), and support two independently-toggleable harness optimizations
— GPU-side preprocessing and CUDA-graph capture — so a real naive-vs-optimized
FPS comparison is possible (a synthetic dummy-tensor loop, which is what
``benchmark_jetson.py``'s existing ``_harness_latency`` uses, never exercises
preprocessing at all, so it can't show the improvement these toggles give).

Preprocessing matches training/production exactly: ImageNet-normalized
bilinear resize (see ``src/perception/datasets/orfd_torch.py``'s
``_to_normalized_tensor`` and ``SegformerImageProcessor``'s defaults).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_bgr_cpu(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    """BGR HxWx3 uint8 -> normalized (1,3,size,size) float32, NCHW. CPU/cv2 path."""
    import cv2

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    arr = arr.transpose(2, 0, 1)[np.newaxis]
    return np.ascontiguousarray(arr, dtype=np.float32)


class SegformerTensorRTEngine:
    """Thin wrapper around a serialized SegFormer TensorRT engine with the
    same two optimizations validated on RF-DETR: GPU-side preprocessing and
    CUDA-graph capture around ``execute_async_v3``.

    Returns the predicted class-index map (argmax over channels, upsampled
    back to the input's native size) from ``.infer()`` — not raw logits —
    so callers don't need engine-specific postprocessing knowledge.
    """

    def __init__(self, engine_path: str | Path, input_size: int | None = None,
                 gpu_preprocess: bool = True, cuda_graph: bool = True) -> None:
        import tensorrt as trt
        import torch

        self._torch = torch
        self._gpu_preprocess = gpu_preprocess
        self._graph = None

        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self._context = self._engine.create_execution_context()

        self._input_name = None
        self._output_name = None
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._input_name = name
            else:
                self._output_name = name
        if self._input_name is None or self._output_name is None:
            raise RuntimeError(f"Engine has no recognizable input/output tensors: {engine_path}")

        # Auto-detect from the engine's own binding shape rather than trusting a
        # caller-supplied default — see _rfdetr_trt_common.RFDETRTensorRTEngine's
        # docstring for why a mismatched size silently corrupts output instead
        # of erroring.
        in_shape = tuple(self._engine.get_tensor_shape(self._input_name))
        detected = in_shape[2] if len(in_shape) == 4 and in_shape[2] > 0 else None
        if detected is not None:
            if input_size is not None and input_size != detected:
                logger.warning(
                    "Requested input_size=%s does not match %s's actual engine "
                    "input shape %s — using the engine's real size.",
                    input_size, engine_path, detected,
                )
            self._input_size = detected
        elif input_size is not None:
            self._input_size = input_size
        else:
            raise ValueError(f"{engine_path}: dynamic input shape {in_shape}, need input_size.")

        def _torch_dtype(name: str):
            np_dtype = trt.nptype(self._engine.get_tensor_dtype(name))
            return torch.from_numpy(np.zeros(1, dtype=np_dtype)).dtype

        self._in_dtype = _torch_dtype(self._input_name)
        self._in_buf = torch.empty((1, 3, self._input_size, self._input_size),
                                    dtype=self._in_dtype, device="cuda")
        out_shape = tuple(self._context.get_tensor_shape(self._output_name))
        self._out_buf = torch.empty(out_shape, dtype=_torch_dtype(self._output_name), device="cuda")

        logger.info("SegFormer TensorRT engine loaded: %s (input_size=%d, out_shape=%s, "
                    "gpu_preprocess=%s, cuda_graph=%s)",
                    engine_path, self._input_size, out_shape, gpu_preprocess, cuda_graph)

        self._context.set_tensor_address(self._input_name, self._in_buf.data_ptr())
        self._context.set_tensor_address(self._output_name, self._out_buf.data_ptr())
        self._done_event = torch.cuda.Event()

        if gpu_preprocess:
            self._pinned_stage: "torch.Tensor | None" = None
            self._mean_bgr = torch.tensor(_IMAGENET_MEAN[::-1].copy(), device="cuda").view(1, 3, 1, 1)
            self._std_bgr = torch.tensor(_IMAGENET_STD[::-1].copy(), device="cuda").view(1, 3, 1, 1)

        self._warmup_and_maybe_capture(cuda_graph)

    def _execute_async(self) -> None:
        stream = self._torch.cuda.current_stream().cuda_stream
        self._context.execute_async_v3(stream)

    def _warmup_and_maybe_capture(self, cuda_graph: bool) -> None:
        torch = self._torch
        self._in_buf.zero_()
        for _ in range(3):
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
            logger.info("CUDA graph capture succeeded for SegFormer engine.")
        except Exception as e:  # noqa: BLE001
            logger.warning("CUDA graph capture failed (%s) — falling back to plain async execution.", e)
            self._graph = None

    def _preprocess_gpu(self, frame_bgr: np.ndarray) -> "torch.Tensor":
        torch = self._torch
        h, w = frame_bgr.shape[:2]
        if self._pinned_stage is None or tuple(self._pinned_stage.shape[:2]) != (h, w):
            self._pinned_stage = torch.empty((h, w, 3), dtype=torch.uint8, pin_memory=True)
        self._pinned_stage.copy_(torch.from_numpy(frame_bgr))
        gpu_u8 = self._pinned_stage.to("cuda", non_blocking=True)
        gpu_f = gpu_u8.permute(2, 0, 1).unsqueeze(0).float()
        resized = torch.nn.functional.interpolate(
            gpu_f, size=(self._input_size, self._input_size), mode="bilinear", align_corners=False,
        )
        return (resized / 255.0 - self._mean_bgr) / self._std_bgr

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Returns the predicted class-index map, upsampled to the input's native size."""
        torch = self._torch
        h0, w0 = frame_bgr.shape[:2]

        if self._gpu_preprocess:
            processed = self._preprocess_gpu(frame_bgr)
            self._in_buf.copy_(processed.to(self._in_dtype))
        else:
            arr = preprocess_bgr_cpu(frame_bgr, self._input_size)
            self._in_buf.copy_(torch.from_numpy(arr).to(self._in_dtype))

        raw_logits = self._run()

        logits = torch.nn.functional.interpolate(
            raw_logits.float(), size=(h0, w0), mode="bilinear", align_corners=False,
        )
        return logits.argmax(dim=1)[0].cpu().numpy()

    def infer_tensor(self, pixel_values: "torch.Tensor") -> "torch.Tensor":
        """Low-level path for callers that already have a preprocessed
        (1,3,input_size,input_size) FP32 tensor (e.g. ORFDDataset's own
        transform for mIoU validation) — bypasses gpu_preprocess/CPU
        preprocessing entirely. Returns a clone of the raw engine logits
        (not upsampled, not argmaxed) so existing callers (_engine_miou,
        _soak_test) can keep their own postprocessing unchanged.

        This is the single execution path shared with .infer() — the
        reason this class exists instead of leaving benchmark_jetson.py's
        _load_trt_context/_trt_infer as a second, separate implementation
        of the same engine-bootstrap + execute logic.
        """
        self._in_buf.copy_(pixel_values.to(device="cuda", dtype=self._in_dtype))
        return self._run()

    def _run(self) -> "torch.Tensor":
        if self._graph is not None:
            self._graph.replay()
        else:
            self._execute_async()
        self._done_event.record()
        self._done_event.synchronize()
        return self._out_buf.clone()


def benchmark_videos(engine: SegformerTensorRTEngine, video_paths: list[Path],
                      warmup_frames: int = 20) -> dict:
    """Real decode+preprocess+infer FPS/latency over every frame of every clip —
    not a synthetic dummy-tensor loop, which never exercises preprocessing and
    so can't show the naive-vs-optimized gap these harness toggles are for."""
    import time

    import cv2

    if not video_paths:
        raise SystemExit("No videos found for FPS benchmarking.")

    cap = cv2.VideoCapture(str(video_paths[0]))
    warmed = 0
    while warmed < warmup_frames:
        ok, frame = cap.read()
        if not ok:
            break
        engine.infer(frame)
        warmed += 1
    cap.release()

    per_clip_fps: dict[str, float] = {}
    latencies_ms: list[float] = []
    for vp in video_paths:
        cap = cv2.VideoCapture(str(vp))
        n = 0
        t_clip0 = time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            engine.infer(frame)
            latencies_ms.append((time.perf_counter() - t0) * 1000)
            n += 1
        cap.release()
        elapsed = time.perf_counter() - t_clip0
        fps = n / elapsed if elapsed > 0 else 0.0
        per_clip_fps[vp.name] = fps
        logger.info("  %-70s %4d frames  %.1f FPS", vp.name, n, fps)

    return {
        "per_clip_fps": per_clip_fps,
        "mean_fps": float(np.mean(list(per_clip_fps.values()))),
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
    }

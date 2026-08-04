#!/usr/bin/env python3
"""One-off probe: does putting the detection and segmentation TensorRT graphs
on separate CUDA streams let their kernels overlap on the Jetson's GPU?

Does NOT modify _rfdetr_trt_common.py / _segformer_trt_common.py -- pokes at
their already-captured CUDA graphs from outside, purely to measure whether
overlap is worth building into jetson_combined_survey.py for real.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _rfdetr_trt_common import load_model  # noqa: E402
from _segformer_trt_common import SegformerTensorRTEngine  # noqa: E402

PERCEPTION_OPTIM = Path.home() / "perception_optim"
SEGFORMER_REPO = PERCEPTION_OPTIM / "segformer_repo"

det = load_model(f"engine:{PERCEPTION_OPTIM / 'weights/rfdetr-s/rfdetr-s_fp32.engine'}")
seg = SegformerTensorRTEngine(
    SEGFORMER_REPO / "weights/segmentation/optimization_distilled/baseline_fp16_256x256.engine",
    gpu_preprocess=True, cuda_graph=True,
)

video = PERCEPTION_OPTIM / "data" / "videos" / "tzir-driving.mp4"
cap = cv2.VideoCapture(str(video))
frames = []
for _ in range(60):
    ok, f = cap.read()
    if not ok:
        break
    frames.append(f)
cap.release()
print(f"loaded {len(frames)} frames")

N_WARMUP, N_TIME = 10, 40


def bench_sequential():
    for f in frames[:N_WARMUP]:
        det.infer(f, threshold=0.35)
        seg.infer(f)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(N_TIME):
        f = frames[i % len(frames)]
        det.infer(f, threshold=0.35)
        seg.infer(f)
    torch.cuda.synchronize()
    return N_TIME / (time.perf_counter() - t0)


def bench_streamed():
    """Issue det's graph replay + seg's graph replay on two different streams
    before syncing either, so the GPU scheduler *can* interleave them if
    there's spare SM capacity. Falls back cleanly if either engine's graph
    capture failed (._graph is None)."""
    if det._graph is None or seg._graph is None:
        print("one of the engines has no captured CUDA graph -- skipping streamed test")
        return None

    det_stream = torch.cuda.Stream()
    seg_stream = torch.cuda.Stream()

    def step(frame):
        # Preprocessing (GPU-side resize/normalize) also gets its own stream,
        # not just the graph replay, so H2D copies can overlap too.
        with torch.cuda.stream(det_stream):
            processed = det._preprocess_gpu(frame)
            det._in_buf.copy_(processed.to(det._in_dtype))
            det._graph.replay()
        with torch.cuda.stream(seg_stream):
            if seg._gpu_preprocess:
                sp = seg._preprocess_gpu(frame)
            else:
                import numpy as np
                from _segformer_trt_common import preprocess_bgr_cpu
                sp = torch.from_numpy(preprocess_bgr_cpu(frame, seg._input_size))
            seg._in_buf.copy_(sp.to(seg._in_dtype))
            seg._graph.replay()
        det_stream.synchronize()
        seg_stream.synchronize()
        d = det._out_bufs[det._dets_name][0].float().cpu().numpy()
        l = det._out_bufs[det._labels_name][0].float().cpu().numpy()
        from _rfdetr_trt_common import decode
        h0, w0 = frame.shape[:2]
        dets = decode(d, l, w0, h0, 0.35)
        class_map = torch.nn.functional.interpolate(
            seg._out_buf.float(), size=(h0, w0), mode="bilinear", align_corners=False,
        ).argmax(dim=1)[0].cpu().numpy()
        return dets, class_map

    for f in frames[:N_WARMUP]:
        step(f)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(N_TIME):
        step(frames[i % len(frames)])
    torch.cuda.synchronize()
    return N_TIME / (time.perf_counter() - t0)


seq_fps = bench_sequential()
print(f"sequential (current design): {seq_fps:.1f} FPS")

try:
    streamed_fps = bench_streamed()
    if streamed_fps is not None:
        print(f"stream-overlapped:            {streamed_fps:.1f} FPS")
        print(f"delta: {(streamed_fps / seq_fps - 1) * 100:+.1f}%")
except Exception as e:
    print(f"streamed probe FAILED: {type(e).__name__}: {e}")

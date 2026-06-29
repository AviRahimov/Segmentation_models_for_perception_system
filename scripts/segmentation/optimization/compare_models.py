#!/usr/bin/env python3
"""Stage 6a: Side-by-side visual comparison of two model variants.

Supports two modes:
  --mode images  Sample N random images from the ORFD test set. Produces
                 3-panel PNGs: [original | model-A overlay | model-B overlay]
                 with per-image mIoU shown in each panel title.

  --mode video   Produce a split-screen MP4 from a video file. Left half =
                 Model A; right half = Model B. Each side has an independent
                 FPS overlay measured in real time.

Model spec format (same for --model-a and --model-b)
----------------------------------------------------
    pytorch:path/to/best.pth
    onnx:path/to/model.onnx
    engine:path/to/model.engine

Usage
-----
    # Image comparison (20 random test images):
    python scripts/segmentation/optimization/compare_models.py --mode images \\
        --model-a pytorch:weights/segmentation/orfd/frozen_backbone/segformer-b2/best.pth \\
        --model-b onnx:weights/segmentation/optimization/qat_int8_256x256.onnx \\
        --test-data datasets/Segmentation_Dataset \\
        --n-samples 20

    # Video comparison:
    python scripts/segmentation/optimization/compare_models.py --mode video \\
        --model-a pytorch:weights/segmentation/orfd/frozen_backbone/segformer-b2/best.pth \\
        --model-b onnx:weights/segmentation/optimization/qat_int8_256x256.onnx \\
        --source samples/desert_video.mp4 \\
        --output reports/segmentation/optimization/video_compare_baseline_vs_qat.mp4
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts" / "segmentation" / "training"))

import train_orfd as _t

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("compare_models")

# ORFD class colours (BGR for OpenCV): 0=non_traversable, 1=traversable, 2=sky
_CLASS_COLORS_BGR = [
    (0,   0, 200),   # non_traversable: red
    (0, 180,  50),   # traversable:     green
    (200, 100, 0),   # sky:             blue
]


# --------------------------------------------------------------------------- #
# Model inference abstraction                                                   #
# --------------------------------------------------------------------------- #

class _Inferencer(Protocol):
    def predict_mask(self, bgr: np.ndarray) -> np.ndarray:
        """Return (H, W) uint8 class-index mask at the same resolution as bgr."""
        ...

    def close(self) -> None:
        ...


class _PyTorchInferencer:
    def __init__(self, checkpoint: str, resolution: int, device: str) -> None:
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        ckpt_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state_dict = ckpt_data.get("net", ckpt_data) if isinstance(ckpt_data, dict) else ckpt_data
        from perception.models.semantic.segformer import _remap_segformer_keys
        state_dict = _remap_segformer_keys(state_dict)
        n_labels = state_dict["decode_head.classifier.weight"].shape[0]
        hf_base = "nvidia/segformer-b2-finetuned-ade-512-512"
        self._processor = SegformerImageProcessor.from_pretrained(hf_base)
        self._processor.size = {"height": resolution, "width": resolution}
        self._model = SegformerForSemanticSegmentation.from_pretrained(
            hf_base, num_labels=n_labels, ignore_mismatched_sizes=True
        )
        self._model.load_state_dict(state_dict, strict=True)
        self._model = self._model.eval().to(device)
        self._device = device
        self._resolution = resolution

    @torch.no_grad()
    def predict_mask(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=[rgb], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device)
        outputs = self._model(pixel_values=pixel_values)
        logits = outputs.logits
        logits = torch.nn.functional.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False,
        )
        return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    def close(self) -> None:
        pass


class _ONNXInferencer:
    def __init__(self, onnx_path: str, resolution: int) -> None:
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            self._sess = ort.InferenceSession(onnx_path, providers=providers)
        except Exception:
            self._sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self._resolution = resolution

        from transformers import SegformerImageProcessor
        self._processor = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512"
        )
        self._processor.size = {"height": resolution, "width": resolution}

    def predict_mask(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=[rgb], return_tensors="np")
        pv = inputs["pixel_values"].astype(np.float32)
        logits = self._sess.run(["logits"], {"pixel_values": pv})[0]  # (1, C, H/4, W/4)
        logits_t = torch.from_numpy(logits)
        logits_t = torch.nn.functional.interpolate(
            logits_t, size=(h, w), mode="bilinear", align_corners=False,
        )
        return logits_t.argmax(dim=1).squeeze(0).numpy().astype(np.uint8)

    def close(self) -> None:
        pass


class _TRTInferencer:
    def __init__(self, engine_path: str, resolution: int) -> None:
        import tensorrt as trt  # type: ignore
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        self._context = engine.create_execution_context()
        out_shape = tuple(self._context.get_tensor_shape("logits"))
        trt_dt = engine.get_tensor_dtype("logits")
        out_dtype = torch.float32 if trt_dt == trt.DataType.FLOAT else torch.float16
        self._out_buf = torch.empty(out_shape, dtype=out_dtype, device="cuda")
        self._resolution = resolution

        from transformers import SegformerImageProcessor
        self._processor = SegformerImageProcessor.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512"
        )
        self._processor.size = {"height": resolution, "width": resolution}

    def predict_mask(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=[rgb], return_tensors="pt")
        pv = inputs["pixel_values"].cuda().float()
        stream = torch.cuda.current_stream().cuda_stream
        self._context.set_tensor_address("pixel_values", pv.data_ptr())
        self._context.set_tensor_address("logits", self._out_buf.data_ptr())
        self._context.execute_async_v3(stream)
        torch.cuda.current_stream().synchronize()
        logits = self._out_buf.clone().float()
        logits = torch.nn.functional.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False,
        )
        return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    def close(self) -> None:
        pass


def _variant_name(spec: str) -> str:
    """Extract the model variant name (filename stem) from a spec like 'engine:path/to/model.engine'."""
    path = spec.split(":", 1)[1] if ":" in spec else spec
    return Path(path).stem


def _build_inferencer(spec: str, resolution: int, device: str) -> _Inferencer:
    """Parse 'type:path' spec and return the appropriate inferencer."""
    if ":" not in spec:
        raise ValueError(f"Model spec must be 'type:path', got: {spec!r}")
    kind, path = spec.split(":", 1)
    path = str(_ROOT / path) if not Path(path).is_absolute() else path
    kind = kind.lower().strip()
    if kind == "pytorch":
        return _PyTorchInferencer(path, resolution, device)
    if kind == "onnx":
        return _ONNXInferencer(path, resolution)
    if kind == "engine":
        return _TRTInferencer(path, resolution)
    raise ValueError(f"Unknown model type: {kind!r}. Use pytorch, onnx, or engine.")


# --------------------------------------------------------------------------- #
# Rendering helpers                                                             #
# --------------------------------------------------------------------------- #

def _make_overlay(bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.5,
                  classes: list[int] | None = None) -> np.ndarray:
    """Blend class-coloured mask onto bgr image (BGR).

    classes: if given, only blend those class indices (None = all classes).
    """
    out = bgr.copy()
    for cls_idx, color_bgr in enumerate(_CLASS_COLORS_BGR):
        if classes is not None and cls_idx not in classes:
            continue
        m = (mask == cls_idx)
        if m.any():
            from perception.render.overlay import blend_mask
            out = blend_mask(out, m, color_bgr, alpha=alpha)
    return out


def _draw_text_label(img: np.ndarray, label: str, miou: float | None = None) -> np.ndarray:
    """Draw a title bar at the top of the image panel."""
    bar_h = 28
    out = np.zeros((img.shape[0] + bar_h, img.shape[1], 3), dtype=np.uint8)
    out[bar_h:] = img
    text = label if miou is None else f"{label}  mIoU={miou:.3f}"
    cv2.putText(out, text, (6, bar_h - 7), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _compute_miou_single(mask_pred: np.ndarray, mask_gt: np.ndarray) -> float:
    """mIoU for a single image."""
    preds = torch.from_numpy(mask_pred.astype(np.int64)).unsqueeze(0)
    labels = torch.from_numpy(mask_gt.astype(np.int64)).unsqueeze(0)
    miou, _ = _t.compute_miou(preds, labels)
    return miou


# --------------------------------------------------------------------------- #
# Image comparison mode                                                         #
# --------------------------------------------------------------------------- #

def _mode_images(args) -> int:
    from perception.datasets.orfd_torch import ORFDDataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    resolution = args.resolution

    logger.info("Building inferencers ...")
    inf_a = _build_inferencer(args.model_a, resolution, device)
    inf_b = _build_inferencer(args.model_b, resolution, device)

    test_data = Path(args.test_data)
    if not test_data.is_absolute():
        test_data = _ROOT / test_data

    # Collect raw image/label pairs from test split.
    ds = ORFDDataset(str(test_data), split="testing", augment=False, input_size=resolution)
    pairs = ds.pairs  # list of (image_path, label_path)

    import random as _rnd
    _rnd.seed(42)
    selected = _rnd.sample(pairs, min(args.n_samples, len(pairs)))
    logger.info("Comparing %d test images ...", len(selected))

    # Output dir.
    name_a = _variant_name(args.model_a)
    name_b = _variant_name(args.model_b)
    out_dir = Path(args.output_dir or
                   _ROOT / "reports" / "optimization" / "qualitative" / f"compare_{name_a}_vs_{name_b}")
    out_dir.mkdir(parents=True, exist_ok=True)

    from perception.datasets.orfd_torch import _remap_label

    for i, (img_path, gt_path) in enumerate(selected):
        bgr = cv2.imread(str(img_path))
        gt_raw = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        gt_mask = _remap_label(gt_raw)

        # Run both models.
        mask_a = inf_a.predict_mask(bgr)
        mask_b = inf_b.predict_mask(bgr)

        miou_a = _compute_miou_single(mask_a, gt_mask)
        miou_b = _compute_miou_single(mask_b, gt_mask)

        overlay_a  = _make_overlay(bgr, mask_a)
        overlay_b  = _make_overlay(bgr, mask_b)
        overlay_gt = _make_overlay(bgr, gt_mask)

        # 4-panel: original | GT | model-A | model-B
        panel_orig = _draw_text_label(bgr,        "Original")
        panel_gt   = _draw_text_label(overlay_gt, "Ground Truth")
        panel_a    = _draw_text_label(overlay_a,  _variant_name(args.model_a), miou_a)
        panel_b    = _draw_text_label(overlay_b,  _variant_name(args.model_b), miou_b)

        # Ensure all panels have the same height.
        panels = [panel_orig, panel_gt, panel_a, panel_b]
        max_h  = max(p.shape[0] for p in panels)
        panels = [
            np.vstack([p, np.zeros((max_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)])
            if p.shape[0] < max_h else p
            for p in panels
        ]

        grid = np.hstack(panels)
        out_path = out_dir / f"compare_{i:04d}.png"
        cv2.imwrite(str(out_path), grid)

        logger.info("[%d/%d] mIoU A=%.3f  B=%.3f  saved: %s",
                    i + 1, len(selected), miou_a, miou_b, out_path.name)

    inf_a.close()
    inf_b.close()
    print(f"\n{len(selected)} comparison images saved to: {out_dir}")
    return 0


# --------------------------------------------------------------------------- #
# Video comparison mode                                                         #
# --------------------------------------------------------------------------- #

def _mode_video(args) -> int:
    from perception.io.video_source import VideoFileSource

    device = "cuda" if torch.cuda.is_available() else "cpu"
    resolution = args.resolution

    source_path = Path(args.source)
    if not source_path.is_absolute():
        source_path = _ROOT / source_path
    if not source_path.is_file():
        logger.error("Source video not found: %s", source_path)
        return 1

    logger.info("Building inferencers ...")
    inf_a = _build_inferencer(args.model_a, resolution, device)
    inf_b = _build_inferencer(args.model_b, resolution, device)

    src = VideoFileSource(source_path)
    total = src.total_frames()
    fps_src = src.fps()
    logger.info("Video: %s  frames=%d  fps=%.1f", source_path.name, total, fps_src)

    name_a = _variant_name(args.model_a)
    name_b = _variant_name(args.model_b)

    out_path = Path(args.output or
                    _ROOT / "reports" / "optimization" /
                    f"video_compare_{name_a}_vs_{name_b}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read first frame to get dimensions.
    ok, first = src.read()
    if not ok or first is None:
        logger.error("Cannot read first frame from %s", source_path)
        return 1
    h, w = first.shape[:2]
    src.seek(0)

    # Output video: side-by-side (2*w × h).
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_src, (w * 2, h))

    frame_idx = 0
    fps_a = fps_b = 0.0
    _ALPHA_EMA = 0.1

    while True:
        ok, bgr = src.read()
        if not ok or bgr is None:
            break

        t0 = time.perf_counter()
        mask_a = inf_a.predict_mask(bgr)
        dt_a = time.perf_counter() - t0
        fps_a = _ALPHA_EMA * (1.0 / dt_a) + (1 - _ALPHA_EMA) * fps_a if fps_a > 0 else 1.0 / dt_a

        t0 = time.perf_counter()
        mask_b = inf_b.predict_mask(bgr)
        dt_b = time.perf_counter() - t0
        fps_b = _ALPHA_EMA * (1.0 / dt_b) + (1 - _ALPHA_EMA) * fps_b if fps_b > 0 else 1.0 / dt_b

        left  = _make_overlay(bgr, mask_a, classes=[1])   # traversable only
        right = _make_overlay(bgr, mask_b, classes=[1])

        from perception.render.overlay import draw_fps
        draw_fps(left,  fps_a)
        draw_fps(right, fps_b)

        # Model label at bottom-left.
        for side, label in [(left, f"Model A: {name_a}"), (right, f"Model B: {name_b}")]:
            cv2.putText(side, label, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (200, 200, 200), 1, cv2.LINE_AA)

        combined = np.hstack([left, right])
        writer.write(combined)
        frame_idx += 1

        if frame_idx % 100 == 0:
            logger.info("[%d/%d]  A: %.1f FPS  B: %.1f FPS", frame_idx, total, fps_a, fps_b)

        if args.max_frames and frame_idx >= args.max_frames:
            break

    writer.release()
    src.release()
    inf_a.close()
    inf_b.close()
    logger.info("Video saved: %s  (%d frames)", out_path, frame_idx)
    print(f"\nVideo comparison saved to: {out_path}")
    return 0


# --------------------------------------------------------------------------- #
# Entry point                                                                   #
# --------------------------------------------------------------------------- #

def main() -> int:
    p = argparse.ArgumentParser(description="Stage 6a: side-by-side model comparison")
    p.add_argument("--mode",      choices=["images", "video"], required=True)
    p.add_argument("--model-a",   required=True,
                   help="Spec: pytorch|onnx|engine:path")
    p.add_argument("--model-b",   required=True,
                   help="Spec: pytorch|onnx|engine:path")
    p.add_argument("--resolution", type=int, default=256,
                   help="Input resolution for both models")

    # Image mode
    p.add_argument("--test-data",  default="datasets/Segmentation_Dataset",
                   help="ORFD root (must contain testing/ split)")
    p.add_argument("--n-samples",  type=int, default=20)
    p.add_argument("--output-dir", default=None,
                   help="Where to save comparison PNGs")

    # Video mode
    p.add_argument("--source",     default=None, help="Input video file")
    p.add_argument("--output",     default=None, help="Output video path")
    p.add_argument("--max-frames", type=int, default=None)

    args = p.parse_args()

    if args.mode == "images":
        return _mode_images(args)
    return _mode_video(args)


if __name__ == "__main__":
    raise SystemExit(main())

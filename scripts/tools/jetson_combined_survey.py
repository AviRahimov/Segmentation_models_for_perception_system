#!/usr/bin/env python3
"""Interactive survey: run ONE detection engine + ONE segmentation engine
together on a real video, on the Jetson, and report real combined FPS.

Standalone by design -- the Jetson's on-device checkouts (~/perception_optim/,
~/perception_optim/segformer_repo/) don't have the full src/perception package
installed (no models/pipeline/render), only the proven TensorRT-spec loaders
each optimization phase already built and validated:
  - detection:   _rfdetr_trt_common.py's load_model() ("engine:"/"ultralytics:" specs)
  - segmentation: _segformer_trt_common.py's SegformerTensorRTEngine

This script imports both directly -- deploy it alongside copies of those two
files (see DEPLOY.md-style instructions in the module docstring below), not
inside the main repo checkout.

Every choice (detection model, segmentation model, video, detection
confidence) can be passed as a flag or left to the interactive prompt; the
first-listed / default option in each prompt is this project's own documented
recommendation (see the *_REGISTRY dicts below for the reasoning per entry).
Segmentation has no confidence knob -- it's always argmax (see CLAUDE.md).

Usage
-----
    # fully interactive
    python3 jetson_combined_survey.py

    # explicit, for scripted comparisons
    python3 jetson_combined_survey.py --detection rfdetr-s --segmentation distilled_fp16 \\
        --video tzir-driving.mp4 --det-conf 0.35 --max-frames 300
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from _rfdetr_trt_common import load_model  # noqa: E402
from _segformer_trt_common import SegformerTensorRTEngine  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("jetson_combined_survey")

# --------------------------------------------------------------------------- #
# Registries -- edit these paths to match your on-device layout.
# Order matters: index 0 is this project's own documented recommendation.
# --------------------------------------------------------------------------- #

_PERCEPTION_OPTIM = Path.home() / "perception_optim"
_SEGFORMER_REPO = _PERCEPTION_OPTIM / "segformer_repo"


@dataclass(frozen=True)
class DetChoice:
    spec: str    # "engine:path" or "ultralytics:path"
    note: str


@dataclass(frozen=True)
class SegChoice:
    engine_path: Path
    note: str


DETECTION_REGISTRY: dict[str, DetChoice] = {
    # Recommended: this project's own comparison (CLAUDE.md) found rfdetr-s the
    # strongest overall candidate -- best accuracy, second-best FPS, no FP16
    # precision cliff to work around (still FP32-only below, see next line).
    "rfdetr-s": DetChoice(
        spec=f"engine:{_PERCEPTION_OPTIM / 'weights/rfdetr-s/rfdetr-s_fp32.engine'}",
        note="strongest overall per CLAUDE.md (best accuracy, 2nd-best FPS)",
    ),
    # rfdetr-m/-s FP16 both have a confirmed real precision collapse (recall->0,
    # not a bug) -- deliberately not offered here, FP32 only.
    "rfdetr-m": DetChoice(
        spec=f"engine:{_PERCEPTION_OPTIM / 'weights/rfdetr-m/rfdetr-m_fp32.engine'}",
        note="current production pin in config.yaml",
    ),
    "yolo11m_freeze21_fp16": DetChoice(
        spec=f"ultralytics:{_PERCEPTION_OPTIM / 'weights/yolo11m_freeze21/yolo11m_freeze21_fp16.engine'}",
        note="fastest option, but meaningfully lower real precision/recall than either RF-DETR variant",
    ),
    "yolo11m_freeze21_fp32": DetChoice(
        spec=f"ultralytics:{_PERCEPTION_OPTIM / 'weights/yolo11m_freeze21/yolo11m_freeze21_fp32.engine'}",
        note="same accuracy caveat as fp16, slower",
    ),
}

SEGMENTATION_REGISTRY: dict[str, SegChoice] = {
    # NEW top recommendation (Stage 6, rock/dune over-segmentation fix): joint
    # ORFD+Gaza+hard-negative training with asymmetric Tversky loss, warm-started
    # from gaza_only_from_frozen_backbone_tuned_gazaval ("balanced_current" below).
    # Beats balanced_current on BOTH Gaza-val (0.9476 vs 0.9050) and ORFD-val
    # (0.8536 vs 0.8411) with no tradeoff, and real Jetson engine mIoU (0.8622)
    # is essentially back to original production's own historic best (0.8624) --
    # at the SAME optimized-harness FPS as balanced_current (114.7-116.7 vs
    # 115.2-115.7). A free upgrade on this device, not just on paper.
    "gaza_joint_hardneg_tversky_b2_fp16": SegChoice(
        engine_path=_SEGFORMER_REPO / "weights/segmentation/optimization_gaza_joint_hardneg_tversky_b2/baseline_fp16_256x256.engine",
        note="Stage 6 fix -- beats balanced_current on both domains, same FPS, engine mIoU 0.8622",
    ),
    "gaza_joint_hardneg_tversky_b2_fp32": SegChoice(
        engine_path=_SEGFORMER_REPO / "weights/segmentation/optimization_gaza_joint_hardneg_tversky_b2/baseline_fp32_256x256.engine",
        note="same as _fp16, no measured accuracy difference",
    ),
    # SegFormer-B3 variant of the same Stage 6 recipe -- highest Gaza-val
    # (0.9727) of any checkpoint tried, at a real FPS cost vs the B2 variants
    # (109.0-110.2 vs 114.7-116.7 optimized) and a small ORFD-val give-up
    # (engine mIoU 0.8362 vs B2's 0.8622).
    "gaza_joint_hardneg_tversky_b3_fp16": SegChoice(
        engine_path=_SEGFORMER_REPO / "weights/segmentation/optimization_segformer-b3/baseline_fp16_256x256.engine",
        note="highest Gaza-val (0.9727) but slower + lower ORFD-val than the B2 Stage 6 variant",
    ),
    "gaza_joint_hardneg_tversky_b3_fp32": SegChoice(
        engine_path=_SEGFORMER_REPO / "weights/segmentation/optimization_segformer-b3/baseline_fp32_256x256.engine",
        note="same as _fp16, no measured accuracy difference",
    ),
    # The pre-Stage-6 "balanced" checkpoint (production candidate before this
    # session's rock/hillside over-segmentation fix) -- kept for direct
    # before/after comparison, now superseded by gaza_joint_hardneg_tversky_b2 above.
    "balanced_current_fp16": SegChoice(
        engine_path=_SEGFORMER_REPO / "weights/segmentation/optimization_balanced_current/baseline_fp16_256x256.engine",
        note="pre-Stage-6 candidate -- superseded by gaza_joint_hardneg_tversky_b2 (same FPS, worse accuracy)",
    ),
    "balanced_current_fp32": SegChoice(
        engine_path=_SEGFORMER_REPO / "weights/segmentation/optimization_balanced_current/baseline_fp32_256x256.engine",
        note="same as _fp16, no measured accuracy difference",
    ),
    # Older recommendation, prior to the Stage 6 rock/dune over-segmentation fix
    # above -- kept for reference. this session's Mask2Former-Large distillation
    # result -- beats both production and its own teacher on the full 3-class
    # metric, same architecture/speed as production so it's a drop-in swap.
    # FP16 confirmed here with zero mIoU loss vs FP32 (unlike RF-DETR, SegFormer
    # has no FP16 cliff).
    "distilled_fp16": SegChoice(
        engine_path=_SEGFORMER_REPO / "weights/segmentation/optimization_distilled/baseline_fp16_256x256.engine",
        note="NEW distillation result -- beats production AND its Mask2Former-Large teacher",
    ),
    "distilled_fp32": SegChoice(
        engine_path=_SEGFORMER_REPO / "weights/segmentation/optimization_distilled/baseline_fp32_256x256.engine",
        note="same as distilled_fp16, no measured accuracy difference",
    ),
    "production_fp16": SegChoice(
        engine_path=_SEGFORMER_REPO / "weights/segmentation/optimization/baseline_fp16_256x256.engine",
        note="current production pin in config.yaml (frozen_backbone/segformer-b2)",
    ),
    "production_fp32": SegChoice(
        engine_path=_SEGFORMER_REPO / "weights/segmentation/optimization/baseline_fp32_256x256.engine",
        note="same as production_fp16, no measured accuracy difference",
    ),
    "qat_int8": SegChoice(
        engine_path=_SEGFORMER_REPO / "weights/segmentation/optimization/qat_int8_256x256.engine",
        note="slower AND less accurate than fp16/fp32 per CLAUDE.md -- not recommended, kept for reference",
    ),
}

_CLASS_NAMES = {0: "mil_vehicle", 1: "person"}
_CLASS_COLORS_BGR = {0: (0, 200, 255), 1: (0, 255, 128)}
_SEG_PALETTE_BGR = np.array([(40, 40, 220), (140, 255, 40), (255, 160, 80)], dtype=np.uint8)  # non_trav, trav, sky


def _videos_dir() -> Path:
    return _PERCEPTION_OPTIM / "data" / "videos"


def _ask_choice(question: str, options: list[tuple[str, str]]) -> str:
    print(f"\n{question}")
    for i, (key, note) in enumerate(options):
        marker = " (default)" if i == 0 else ""
        print(f"  [{i}] {key}  -- {note}{marker}")
    try:
        raw = input(f"Choice [0-{len(options) - 1}], Enter=0: ").strip()
    except (EOFError, OSError):
        raw = ""
    if not raw:
        return options[0][0]
    try:
        idx = int(raw)
        if 0 <= idx < len(options):
            return options[idx][0]
    except ValueError:
        pass
    print(f"Unrecognised choice {raw!r}; using default.")
    return options[0][0]


def _ask_float(question: str, default: float) -> float:
    try:
        raw = input(f"{question} [Enter={default}]: ").strip()
    except (EOFError, OSError):
        raw = ""
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"Unrecognised value {raw!r}; using default.")
        return default


def _draw_detections(frame: np.ndarray, dets: list) -> np.ndarray:
    out = frame
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.xyxy)
        color = _CLASS_COLORS_BGR.get(d.class_id, (200, 200, 200))
        name = _CLASS_NAMES.get(d.class_id, f"class_{d.class_id}")
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{name} {d.score:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def _overlay_segmentation(frame: np.ndarray, class_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = _SEG_PALETTE_BGR[np.clip(class_map, 0, 2)]
    return cv2.addWeighted(frame, 1 - alpha, color, alpha, 0.0)


def run(det_key: str, seg_key: str, video_path: Path, det_conf: float,
        output_path: Path, max_frames: int) -> dict:
    det_choice = DETECTION_REGISTRY[det_key]
    seg_choice = SEGMENTATION_REGISTRY[seg_key]

    logger.info("Loading detection model: %s (%s)", det_key, det_choice.spec)
    det_model = load_model(det_choice.spec)
    logger.info("Loading segmentation model: %s (%s)", seg_key, seg_choice.engine_path)
    seg_engine = SegformerTensorRTEngine(seg_choice.engine_path, gpu_preprocess=True, cuda_graph=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0

    writer = None
    n = 0
    det_times: list[float] = []
    seg_times: list[float] = []
    combined_times: list[float] = []
    n_warmup = 10

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames and n >= max_frames + n_warmup:
            break

        t0 = time.perf_counter()
        dets = det_model.infer(frame, threshold=det_conf)
        t1 = time.perf_counter()
        class_map = seg_engine.infer(frame)
        t2 = time.perf_counter()

        if n >= n_warmup:
            det_times.append(t1 - t0)
            seg_times.append(t2 - t1)
            combined_times.append(t2 - t0)

        rendered = _overlay_segmentation(frame, class_map)
        rendered = _draw_detections(rendered, dets)
        if writer is None:
            h, w = rendered.shape[:2]
            writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
        writer.write(rendered)
        n += 1

    cap.release()
    if writer is not None:
        writer.release()

    def _stats(times: list[float]) -> dict:
        if not times:
            return {"fps": float("nan"), "p50_ms": float("nan")}
        arr = np.array(times)
        return {"fps": 1.0 / arr.mean(), "p50_ms": float(np.median(arr) * 1000)}

    result = {
        "detection": det_key, "segmentation": seg_key, "video": video_path.name,
        "det_conf": det_conf, "n_frames": len(combined_times),
        "detection_only": _stats(det_times),
        "segmentation_only": _stats(seg_times),
        "combined": _stats(combined_times),
    }
    logger.info(
        "detection=%s segmentation=%s video=%s -> combined %.1f FPS "
        "(detection-only %.1f FPS, segmentation-only %.1f FPS)",
        det_key, seg_key, video_path.name, result["combined"]["fps"],
        result["detection_only"]["fps"], result["segmentation_only"]["fps"],
    )
    logger.info("Wrote %s", output_path)
    return result


def _check_environment() -> None:
    """Fail fast, with a clear message, before the interactive survey --
    otherwise a missing `source ~/perception_optim/env.sh` only surfaces as a
    raw ImportError (libcusparseLt.so.0 missing from LD_LIBRARY_PATH) at
    model-load time, after the user has already answered every prompt."""
    try:
        import torch  # noqa: F401
        import tensorrt  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            f"Environment not set up ({e}).\n"
            f"Run this first, in the same shell:\n"
            f"    source ~/perception_optim/env.sh\n"
            f"then re-run this script."
        )


def main() -> int:
    _check_environment()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--detection", choices=list(DETECTION_REGISTRY), default=None)
    p.add_argument("--segmentation", choices=list(SEGMENTATION_REGISTRY), default=None)
    p.add_argument("--video", default=None, help="Filename under ~/perception_optim/data/videos/")
    p.add_argument("--det-conf", type=float, default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--max-frames", type=int, default=300, help="0 = full clip")
    args = p.parse_args()

    det_key = args.detection or _ask_choice(
        "Which detection model?", [(k, v.note) for k, v in DETECTION_REGISTRY.items()],
    )
    seg_key = args.segmentation or _ask_choice(
        "Which segmentation model?", [(k, v.note) for k, v in SEGMENTATION_REGISTRY.items()],
    )

    videos = sorted(_videos_dir().glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No .mp4 files under {_videos_dir()}")
    if args.video:
        video_path = _videos_dir() / args.video
    else:
        # Recommendation: tzir-driving.mp4 is this project's own canonical
        # single-clip demo (see JETSON.md's run_player.py examples).
        names = [v.name for v in videos]
        ordered = (["tzir-driving.mp4"] if "tzir-driving.mp4" in names else []) + \
                  [n for n in names if n != "tzir-driving.mp4"]
        video_name = _ask_choice("Which video?", [(n, "canonical demo clip" if n == "tzir-driving.mp4" else "")
                                                    for n in ordered])
        video_path = _videos_dir() / video_name
    if not video_path.is_file():
        raise SystemExit(f"Video not found: {video_path}")

    det_conf = args.det_conf if args.det_conf is not None else _ask_float(
        "Detection confidence threshold (production-default FPS-pass value is 0.35; "
        "0.50 is the best-F1 accuracy operating point)", 0.35,
    )

    output_path = Path(args.output) if args.output else (
        Path.home() / "perception_optim" / "results" / f"combined_{det_key}_{seg_key}_{video_path.stem}.mp4"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run(det_key, seg_key, video_path, det_conf, output_path, args.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

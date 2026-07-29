"""Shared real-video FPS benchmarking + coarse accuracy check.

Framework-agnostic: works with any object exposing
``.infer(frame_bgr: np.ndarray, threshold: float) -> list[Detection-like]``,
where each detection has ``.class_id``, ``.score``, ``.xyxy`` — both
``_rfdetr_trt_common.Detection`` and the small wrapper ``benchmark_yolo_jetson.py``
builds around Ultralytics' ``Results`` satisfy this. Used by both
``benchmark_jetson.py`` (RF-DETR) and ``benchmark_yolo_jetson.py`` (YOLO) so the
FPS/accuracy methodology is identical across model families — an honest
comparison, not two different measurement rulers.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Real-video FPS/latency                                                       #
# --------------------------------------------------------------------------- #

def benchmark_videos(model, video_paths: list[Path], threshold: float, warmup_frames: int = 20) -> dict:
    """Decode+preprocess+infer over every frame of every clip. Real usage, not a
    synthetic dummy-tensor loop — this is what actually answers 'what FPS do I get'.
    """
    if not video_paths:
        raise SystemExit("No videos found for FPS benchmarking.")

    cap = cv2.VideoCapture(str(video_paths[0]))
    warmed = 0
    while warmed < warmup_frames:
        ok, frame = cap.read()
        if not ok:
            break
        model.infer(frame, threshold=threshold)
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
            model.infer(frame, threshold=threshold)
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


# --------------------------------------------------------------------------- #
# Coarse accuracy sanity check                                                  #
# --------------------------------------------------------------------------- #

def load_yolo_seg_polygon_gt(label_path: Path, img_w: int, img_h: int) -> list[tuple[int, tuple]]:
    """YOLO-seg polygon labels ('cls x1 y1 x2 y2 ...' normalized) -> axis-aligned bboxes."""
    if not label_path.is_file():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        coords = [float(v) for v in parts[1:]]
        xs = [coords[i] * img_w for i in range(0, len(coords), 2)]
        ys = [coords[i] * img_h for i in range(1, len(coords), 2)]
        boxes.append((cls, (min(xs), min(ys), max(xs), max(ys))))
    return boxes


def iou_xyxy(a: tuple, b: tuple) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate_accuracy(model, images_dir: Path, labels_dir: Path,
                       threshold: float = 0.4, iou_thr: float = 0.5) -> dict:
    """Fixed operating-point precision/recall/FP-per-image, matching this repo's own
    leaderboard.py convention (conf=0.40) rather than a full AP curve — a regression
    guard against a badly broken export, not a publishable accuracy number.
    """
    images = sorted(images_dir.glob("*.jpg"))
    if not images:
        return {"precision": None, "recall": None, "fp_per_image": None, "n_gt": 0, "n_images": 0}

    tp, fp, n_gt = 0, 0, 0
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = load_yolo_seg_polygon_gt(labels_dir / (img_path.stem + ".txt"), w, h)
        n_gt += len(gts)
        preds = model.infer(img, threshold=threshold)

        matched_gt = set()
        for det in sorted(preds, key=lambda d: -d.score):
            best_iou, best_j = 0.0, None
            for j, (gt_cls, gt_box) in enumerate(gts):
                if j in matched_gt or gt_cls != det.class_id:
                    continue
                iou = iou_xyxy(det.xyxy, gt_box)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j is not None and best_iou >= iou_thr:
                matched_gt.add(best_j)
                tp += 1
            else:
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / n_gt if n_gt > 0 else None
    fp_per_image = fp / len(images) if images else None
    logger.info("Accuracy @ conf=%.2f: precision=%s recall=%s FP/img=%s (n_gt=%d, n_images=%d)",
                threshold, precision, recall, fp_per_image, n_gt, len(images))
    return {"precision": precision, "recall": recall, "fp_per_image": fp_per_image,
            "n_gt": n_gt, "n_images": len(images)}

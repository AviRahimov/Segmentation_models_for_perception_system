"""Paper-style comparison of multiple trained detection models.

Three modes:

  --mode table
      Run model.val() on the validation set for each model and print a metrics
      table (mAP50, mAP50-95, per-class AP50).  Saves results to
      reports/detection/comparison_*.csv.

  --mode images
      For each sampled image, draw a grid panel: [GT] [Model-A] [Model-B] ...
      GT boxes are drawn in green; predicted boxes use per-class colours.
      Accepts labelled val/test images or unlabelled test images.
      Saves PNGs to reports/detection/qualitative/compare_*/img_*.png.

  --mode video
      For each frame, run all models and produce a side-by-side MP4:
      [Model-A | Model-B | ...].  Saves to
      reports/detection/video_compare_*.mp4.

Model spec format
-----------------
    pytorch:weights/detection/yolo26m/round1/best.pt
    pytorch:weights/detection/yoloe-26m/round1/best.pt

Usage
-----
    # Metrics table (all 6 trained models):
    python scripts/detection/evaluation/compare_detection_models.py --mode table \\
        --models pytorch:weights/detection/yolo26s/round1/best.pt \\
                 pytorch:weights/detection/yolo26m/round1/best.pt \\
                 pytorch:weights/detection/yolo26l/round1/best.pt \\
                 pytorch:weights/detection/yoloe-26s/round1/best.pt \\
                 pytorch:weights/detection/yoloe-26m/round1/best.pt \\
                 pytorch:weights/detection/yoloe-26l/round1/best.pt \\
        --data datasets/Detection_Dataset/data.yaml

    # Side-by-side on val images:
    python scripts/detection/evaluation/compare_detection_models.py --mode images \\
        --models pytorch:weights/detection/yolo26m/round1/best.pt \\
                 pytorch:weights/detection/yoloe-26m/round1/best.pt \\
        --test-data datasets/Detection_Dataset/valid/images --n-samples 20

    # Side-by-side on custom test images (no labels):
    python scripts/detection/evaluation/compare_detection_models.py --mode images \\
        --models pytorch:weights/detection/yolo26m/round1/best.pt \\
                 pytorch:weights/detection/yoloe-26m/round1/best.pt \\
        --test-data /path/to/test_images/

    # Video comparison:
    python scripts/detection/evaluation/compare_detection_models.py --mode video \\
        --models pytorch:weights/detection/yolo26m/round1/best.pt \\
                 pytorch:weights/detection/yoloe-26m/round1/best.pt \\
        --source samples/clip.mp4
"""
from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _ap_utils import (  # noqa: E402
    infer_rfdetr_profile,
    is_rfdetr_checkpoint,
    load_rfdetr_for_eval,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("compare_detection")

# Per-class box colours (BGR): index 0 = first class, 1 = second, etc.
_CLASS_COLORS_BGR = [
    (0,   200,  50),   # class 0 — Military Vehicle: green
    (50,  150, 255),   # class 1 — person: orange
    (255,  50,  50),   # class 2+: blue (future classes)
    (255,  50, 255),
    (0,   255, 255),
]
_GT_COLOR_BGR = (0, 255, 0)   # ground-truth: bright green
_PANEL_WIDTH  = 420            # pixels per panel column
_FONT         = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class _ModelSpec:
    label: str          # e.g. "yolo26m"
    weights: Path


def _parse_spec(spec: str) -> _ModelSpec:
    """Parse 'pytorch:path/to/best.pt' → _ModelSpec."""
    if ":" in spec:
        backend, path_str = spec.split(":", 1)
        if backend != "pytorch":
            raise ValueError(f"Only 'pytorch:' specs are supported in this script; got: {backend!r}")
    else:
        path_str = spec
    weights = Path(path_str)
    if not weights.is_absolute():
        weights = _ROOT / weights
    # Derive a unique label from the path parts after "detection":
    #   .../detection/yolo26m/round1/best.pt            → "yolo26m/round1"
    #   .../detection/yolo11m/exp/freeze10_aug_clean/best.pt
    #                                                    → "yolo11m/exp/freeze10_aug_clean"
    # (parent.parent.name alone collapses all exp variants to the label "exp",
    #  which breaks table rows and per-model threshold lookups)
    parts = weights.parts
    if "detection" in parts:
        label = "/".join(parts[parts.index("detection") + 1 : -1])
    else:
        label = weights.parent.parent.name
    return _ModelSpec(label=label, weights=weights)


def _reject_rfdetr(specs: list[_ModelSpec], mode: str) -> None:
    """--mode table/images aren't extended for RF-DETR (only --mode video is,
    per scope) — fail clearly here instead of a confusing AttributeError deep
    in Ultralytics-specific code (model.val()/predict(conf=...) kwargs)."""
    bad = [s.label for s in specs if is_rfdetr_checkpoint(s.weights)]
    if bad:
        raise NotImplementedError(
            f"--mode {mode} does not support RF-DETR checkpoints yet (only "
            f"--mode video does): {bad}"
        )


def _load_model(spec: _ModelSpec, conf: float = 0.25):
    """conf only matters for RF-DETR: its confidence threshold is fixed at
    construction time (no per-call override like Ultralytics' predict(conf=))."""
    if not spec.weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {spec.weights}")
    if is_rfdetr_checkpoint(spec.weights):
        return load_rfdetr_for_eval(spec.weights, confidence_floor=conf)
    from ultralytics import YOLO
    return YOLO(str(spec.weights))


def _draw_boxes(img_bgr: np.ndarray, boxes_xyxy: list[list[float]],
                scores: list[float], class_ids: list[int],
                class_names: dict[int, str], color_override: tuple | None = None) -> np.ndarray:
    """Draw bounding boxes on a copy of img_bgr."""
    out = img_bgr.copy()
    for box, score, cid in zip(boxes_xyxy, scores, class_ids):
        x1, y1, x2, y2 = (int(v) for v in box)
        color = color_override or _CLASS_COLORS_BGR[cid % len(_CLASS_COLORS_BGR)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names.get(cid, str(cid))} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.45, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(out, label, (x1 + 1, y1 - 3), _FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> tuple[list, list]:
    """Read YOLO-format label file; return (boxes_xyxy, class_ids)."""
    if not label_path.exists():
        return [], []
    boxes, classes = [], []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        # YOLO polygon format: class cx cy x1 y1 x2 y2 ... or bbox: class cx cy w h
        # If > 5 values: polygon → compute tight bbox
        coords = [float(v) for v in parts[1:]]
        if len(coords) == 4:
            cx, cy, bw, bh = coords
            x1 = (cx - bw / 2) * img_w
            y1 = (cy - bh / 2) * img_h
            x2 = (cx + bw / 2) * img_w
            y2 = (cy + bh / 2) * img_h
        else:
            # Polygon: alternating x y pairs; compute bounding box
            xs = [c * img_w for c in coords[0::2]]
            ys = [c * img_h for c in coords[1::2]]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        boxes.append([x1, y1, x2, y2])
        classes.append(cid)
    return boxes, classes


def _make_label_panel(img_bgr: np.ndarray, title: str, panel_w: int) -> np.ndarray:
    """Resize img to panel_w × scaled_h and add a title bar at the top."""
    h, w = img_bgr.shape[:2]
    scale = panel_w / w
    new_h = int(h * scale)
    resized = cv2.resize(img_bgr, (panel_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Title bar
    bar_h = 30
    bar = np.zeros((bar_h, panel_w, 3), dtype=np.uint8)
    cv2.putText(bar, title, (6, 20), _FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, resized])


# ------------------------------------------------------------------ #
# Mode: table                                                          #
# ------------------------------------------------------------------ #

def _run_table(specs: list[_ModelSpec], data_path: Path, split: str,
               imgsz: int, batch: int, device: str, half: bool,
               conf: float = 0.25, iou: float = 0.7,
               per_model_thresholds: dict | None = None) -> None:
    _reject_rfdetr(specs, "table")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = _ROOT / "reports" / "detection"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / f"comparison_{ts}.csv"

    rows: list[dict] = []
    header_printed = False

    for spec in specs:
        logger.info("Evaluating %s ...", spec.label)
        model = _load_model(spec)
        m_conf = (per_model_thresholds or {}).get(spec.label, {}).get("conf", conf)
        m_iou  = (per_model_thresholds or {}).get(spec.label, {}).get("iou",  iou)
        metrics = model.val(
            data=str(data_path),
            split=split,
            imgsz=imgsz,
            batch=batch,
            device=device,
            half=half,
            conf=m_conf,
            iou=m_iou,
            verbose=False,
        )
        box = metrics.box
        names = model.names
        nc = len(names)

        row: dict[str, Any] = {
            "model":    spec.label,
            "mAP50":    round(float(box.map50), 4),
            "mAP50-95": round(float(box.map),   4),
        }
        for i in range(nc):
            cname = names.get(i, str(i))
            row[f"AP50_{cname}"] = round(float(box.ap50[i]) if i < len(box.ap50) else float("nan"), 4)

        # Param count from model
        try:
            params_m = round(sum(p.numel() for p in model.model.parameters()) / 1e6, 1)
            row["params_M"] = params_m
        except Exception:
            row["params_M"] = None

        rows.append(row)
        del model

        if not header_printed:
            header_printed = True

    # Print table
    col_names = list(rows[0].keys()) if rows else []
    col_w = {k: max(len(k), max(len(str(r.get(k, ""))) for r in rows)) for k in col_names}
    sep = "  ".join("-" * col_w[k] for k in col_names)
    hdr = "  ".join(k.ljust(col_w[k]) for k in col_names)
    logger.info("")
    logger.info("Comparison results (split=%s):", split)
    logger.info(hdr)
    logger.info(sep)
    for row in rows:
        logger.info("  ".join(str(row.get(k, "")).ljust(col_w[k]) for k in col_names))
    logger.info("")

    # Save CSV
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=col_names)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Comparison table saved → %s", csv_path)


# ------------------------------------------------------------------ #
# Mode: images                                                         #
# ------------------------------------------------------------------ #

def _run_images(specs: list[_ModelSpec], test_data_dir: Path,
                n_samples: int, imgsz: int, conf: float, device: str, half: bool,
                iou: float = 0.7, per_model_thresholds: dict | None = None) -> None:
    _reject_rfdetr(specs, "images")
    # Collect images
    img_paths = sorted([
        p for p in test_data_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    ])
    if not img_paths:
        raise FileNotFoundError(f"No images found in: {test_data_dir}")

    if n_samples < len(img_paths):
        random.shuffle(img_paths)
        img_paths = img_paths[:n_samples]
    logger.info("Comparing %d models on %d images ...", len(specs), len(img_paths))

    # Load all models
    models = [(spec, _load_model(spec)) for spec in specs]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _ROOT / "reports" / "detection" / "qualitative" / f"compare_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, img_path in enumerate(img_paths):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            logger.warning("Could not read image: %s", img_path)
            continue
        h, w = bgr.shape[:2]

        panels: list[np.ndarray] = []

        # GT panel (if label file exists)
        label_path = img_path.parent.parent / "labels" / img_path.with_suffix(".txt").name
        gt_boxes, gt_classes = _load_gt_boxes(label_path, w, h)
        gt_names = models[0][1].names if models else {}
        gt_img = _draw_boxes(bgr, gt_boxes, [1.0] * len(gt_boxes), gt_classes, gt_names,
                              color_override=_GT_COLOR_BGR)
        panels.append(_make_label_panel(gt_img, "Ground Truth", _PANEL_WIDTH))

        # Per-model prediction panels
        for spec, model in models:
            m_conf = (per_model_thresholds or {}).get(spec.label, {}).get("conf", conf)
            m_iou  = (per_model_thresholds or {}).get(spec.label, {}).get("iou",  iou)
            results = model.predict(bgr, imgsz=imgsz, conf=m_conf, iou=m_iou,
                                    device=device, half=half, verbose=False)
            r = results[0]
            if r.boxes is not None and len(r.boxes):
                boxes  = r.boxes.xyxy.cpu().numpy().tolist()
                scores = r.boxes.conf.cpu().numpy().tolist()
                cids   = r.boxes.cls.cpu().numpy().astype(int).tolist()
            else:
                boxes, scores, cids = [], [], []

            pred_img = _draw_boxes(bgr, boxes, scores, cids, model.names)
            n_det = len(boxes)
            panels.append(_make_label_panel(pred_img, f"{spec.label} ({n_det})", _PANEL_WIDTH))

        # Pad panels to same height
        max_h = max(p.shape[0] for p in panels)
        padded = []
        for p in panels:
            ph, pw = p.shape[:2]
            if ph < max_h:
                pad = np.zeros((max_h - ph, pw, 3), dtype=np.uint8)
                p = np.vstack([p, pad])
            padded.append(p)

        grid = np.hstack(padded)
        out_path = out_dir / f"img_{idx:04d}_{img_path.stem}.png"
        cv2.imwrite(str(out_path), grid)

    # Cleanup
    for _, model in models:
        del model

    logger.info("Image panels saved → %s", out_dir)
    logger.info("  %d images, %d panels each (GT + %d models)", len(img_paths), len(specs) + 1, len(specs))


# ------------------------------------------------------------------ #
# Mode: video                                                          #
# ------------------------------------------------------------------ #

def _run_video(specs: list[_ModelSpec], source: Path, out_path: Path,
               imgsz: int, conf: float, device: str, half: bool, max_frames: int,
               iou: float = 0.7, per_model_thresholds: dict | None = None) -> None:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    fps_src   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_panels  = len(specs)
    out_w     = _PANEL_WIDTH * n_panels
    scale     = _PANEL_WIDTH / src_w
    out_h     = int(src_h * scale) + 30  # +30 for title bar

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_src, (out_w, out_h))

    models = [
        (spec, _load_model(
            spec, conf=(per_model_thresholds or {}).get(spec.label, {}).get("conf", conf),
        ))
        for spec in specs
    ]
    # Stable class-name -> id map per RF-DETR spec, built once (not per-frame:
    # per-frame would reassign ids/colors depending on which classes happen to
    # appear in that one frame, making box colors flicker across the video).
    rfdetr_class_maps: dict[str, dict[str, int]] = {
        spec.label: {c.name: i for i, c in enumerate(infer_rfdetr_profile(spec.weights))}
        for spec in specs if is_rfdetr_checkpoint(spec.weights)
    }
    frame_idx = 0
    logger.info("Processing video: %s  →  %s", source.name, out_path.name)

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if max_frames > 0 and frame_idx >= max_frames:
            break

        t0 = time.perf_counter()
        panels = []
        for spec, model in models:
            if spec.label in rfdetr_class_maps:
                name_to_id = rfdetr_class_maps[spec.label]
                dets = model.predict(bgr)
                boxes  = [list(d.bbox_xyxy) for d in dets]
                scores = [d.score for d in dets]
                cids   = [name_to_id.get(d.class_name, 0) for d in dets]
                names  = {i: n for n, i in name_to_id.items()}
            else:
                m_conf = (per_model_thresholds or {}).get(spec.label, {}).get("conf", conf)
                m_iou  = (per_model_thresholds or {}).get(spec.label, {}).get("iou",  iou)
                results = model.predict(bgr, imgsz=imgsz, conf=m_conf, iou=m_iou,
                                        device=device, half=half, verbose=False)
                r = results[0]
                if r.boxes is not None and len(r.boxes):
                    boxes  = r.boxes.xyxy.cpu().numpy().tolist()
                    scores = r.boxes.conf.cpu().numpy().tolist()
                    cids   = r.boxes.cls.cpu().numpy().astype(int).tolist()
                else:
                    boxes, scores, cids = [], [], []
                names = model.names
            pred_img = _draw_boxes(bgr, boxes, scores, cids, names)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            fps_label  = f"{spec.label}  {elapsed_ms:.0f}ms"
            panels.append(_make_label_panel(pred_img, fps_label, _PANEL_WIDTH))
            t0 = time.perf_counter()

        # Pad to same height and concat
        max_h = max(p.shape[0] for p in panels)
        padded = []
        for p in panels:
            ph, pw = p.shape[:2]
            if ph < max_h:
                pad = np.zeros((max_h - ph, pw, 3), dtype=np.uint8)
                p = np.vstack([p, pad])
            padded.append(p)
        frame_out = np.hstack(padded)

        # Ensure exact output dimensions
        if frame_out.shape[:2] != (out_h, out_w):
            frame_out = cv2.resize(frame_out, (out_w, out_h))

        writer.write(frame_out)
        frame_idx += 1
        if frame_idx % 50 == 0:
            logger.info("  Processed %d frames ...", frame_idx)

    cap.release()
    writer.release()
    for _, model in models:
        del model

    logger.info("Video saved → %s  (%d frames)", out_path, frame_idx)


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare detection models — table / images / video")
    p.add_argument("--mode",    required=True, choices=["table", "images", "video"],
                   help="Comparison mode")
    p.add_argument("--models",  nargs="+", required=True, metavar="SPEC",
                   help="One or more model specs: pytorch:path/to/best.pt")
    p.add_argument("--data",    default="datasets/Detection_Dataset/data.yaml",
                   help="YOLO data.yaml (used by --mode table)")
    p.add_argument("--split",   default="val", choices=["val", "test"],
                   help="Dataset split (--mode table)")
    p.add_argument("--test-data", default=None, dest="test_data",
                   help="Path to images dir (--mode images). Defaults to val set from data.yaml")
    p.add_argument("--source",  default=None,
                   help="Video file path (--mode video)")
    p.add_argument("--output",  default=None,
                   help="Output MP4 path (--mode video, optional)")
    p.add_argument("--n-samples", type=int, default=20, dest="n_samples",
                   help="Number of images to compare (--mode images)")
    p.add_argument("--max-frames", type=int, default=0, dest="max_frames",
                   help="Max frames to process (--mode video, 0=all)")
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--conf",    type=float, default=0.25,
                   help="Default confidence threshold (overridden per-model by --thresholds-file)")
    p.add_argument("--iou",     type=float, default=0.7,
                   help="NMS IoU threshold (overridden per-model by --thresholds-file)")
    p.add_argument("--thresholds-file", default=None, dest="thresholds_file",
                   help="JSON from tune_thresholds.py — sets per-model conf+iou automatically")
    p.add_argument("--batch",   type=int, default=16)
    p.add_argument("--device",  default="0")
    p.add_argument("--no-half", dest="half", action="store_false", default=True)
    return p.parse_args()


def _load_thresholds(path: str | None) -> dict:
    """Load per-model thresholds from a JSON file produced by tune_thresholds.py."""
    if not path:
        return {}
    import json
    p = Path(path)
    if not p.is_absolute():
        p = _ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Thresholds file not found: {p}")
    data = json.loads(p.read_text())
    logger.info("Loaded per-model thresholds from %s", p)
    for label, t in data.items():
        logger.info("  %s: conf=%.2f  iou=%.2f  mAP50=%.4f",
                    label, t.get("conf", 0), t.get("iou", 0), t.get("map50", 0))
    return data


def main() -> None:
    args = parse_args()

    specs = [_parse_spec(s) for s in args.models]
    logger.info("Models: %s", [s.label for s in specs])

    per_model_thresholds = _load_thresholds(args.thresholds_file)

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = _ROOT / data_path

    if args.mode == "table":
        _run_table(specs, data_path, args.split, args.imgsz,
                   args.batch, args.device, args.half,
                   conf=args.conf, iou=args.iou,
                   per_model_thresholds=per_model_thresholds)

    elif args.mode == "images":
        if args.test_data:
            test_dir = Path(args.test_data)
            if not test_dir.is_absolute():
                test_dir = _ROOT / test_dir
        else:
            # Default: val images directory from data.yaml.
            # Roboflow-generated yamls use ../valid/images relative to the yaml's
            # parent directory, which resolves one level too high given our layout
            # (datasets/Detection_Dataset/valid/images).  Try the yaml-relative path
            # first; if it doesn't exist, strip leading ../ components and retry.
            import re, yaml
            with data_path.open() as f:
                d = yaml.safe_load(f)
            val_rel = d.get("val", "valid/images")
            test_dir = (data_path.parent / val_rel).resolve()
            if not test_dir.exists():
                val_rel_clean = re.sub(r"^(\.\./)+", "", val_rel)
                test_dir = (data_path.parent / val_rel_clean).resolve()

        _run_images(specs, test_dir, args.n_samples, args.imgsz,
                    args.conf, args.device, args.half,
                    iou=args.iou, per_model_thresholds=per_model_thresholds)

    elif args.mode == "video":
        if not args.source:
            raise ValueError("--source is required for --mode video")
        source = Path(args.source)
        if not source.is_absolute():
            source = _ROOT / source
        if not source.exists():
            raise FileNotFoundError(f"Video not found: {source}")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        labels = "_vs_".join(s.label for s in specs)
        if args.output:
            out_path = Path(args.output)
            if not out_path.is_absolute():
                out_path = _ROOT / out_path
        else:
            out_dir = _ROOT / "reports" / "detection"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"video_compare_{labels}_{ts}.mp4"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        _run_video(specs, source, out_path, args.imgsz, args.conf,
                   args.device, args.half, args.max_frames,
                   iou=args.iou, per_model_thresholds=per_model_thresholds)


if __name__ == "__main__":
    main()

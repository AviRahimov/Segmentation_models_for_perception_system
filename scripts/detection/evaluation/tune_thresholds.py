"""Sweep confidence × NMS-IoU thresholds on the val set to find the best pair per model.

Runs model.val() at every (conf, iou) combination, picks the best by mAP50, and writes
the results to JSON (for use with compare_detection_models.py --thresholds-file) and CSV.

Usage
-----
    python scripts/detection/evaluation/tune_thresholds.py \\
        --models pytorch:weights/detection/yolo26m/round1/best.pt \\
                 pytorch:weights/detection/yoloe-26m/round1/best.pt \\
                 pytorch:weights/detection/yolo11m/round1/best.pt \\
        --data datasets/Detection_Dataset/data.yaml

    # Custom sweep range:
    python scripts/detection/evaluation/tune_thresholds.py \\
        --models pytorch:weights/detection/yolo26m/round1/best.pt \\
        --data datasets/Detection_Dataset/data.yaml \\
        --conf-range 0.20 0.50 0.05 \\
        --iou-range  0.40 0.71 0.10 \\
        --out reports/detection/best_thresholds.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tune_thresholds")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep conf × iou thresholds on val set; save best per model to JSON"
    )
    p.add_argument("--models", nargs="+", required=True, metavar="SPEC",
                   help="Model specs: pytorch:path/to/best.pt")
    p.add_argument("--data",   default="datasets/Detection_Dataset/data.yaml",
                   help="YOLO data.yaml")
    p.add_argument("--conf-range", nargs=3, type=float, default=[0.10, 0.56, 0.05],
                   metavar=("START", "STOP", "STEP"),
                   help="Confidence sweep: start stop step (default 0.10 0.56 0.05)")
    p.add_argument("--iou-range",  nargs=3, type=float, default=[0.30, 0.71, 0.10],
                   metavar=("START", "STOP", "STEP"),
                   help="NMS IoU sweep: start stop step (default 0.30 0.71 0.10)")
    p.add_argument("--split",   default="val", choices=["val", "test"])
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--batch",   type=int, default=16)
    p.add_argument("--device",  default="0")
    p.add_argument("--no-half", dest="half", action="store_false", default=True)
    p.add_argument("--out",     default=None,
                   help="Output JSON path (default: reports/detection/best_thresholds.json)")
    return p.parse_args()


def _parse_spec(spec: str) -> tuple[str, Path]:
    """Return (label, weights_path) from 'pytorch:path/to/best.pt'."""
    if ":" in spec:
        _, path_str = spec.split(":", 1)
    else:
        path_str = spec
    weights = Path(path_str)
    if not weights.is_absolute():
        weights = _ROOT / weights
    label = weights.parent.parent.name   # .../yolo26m/round1/best.pt → "yolo26m"
    return label, weights


def _arange(start: float, stop: float, step: float) -> list[float]:
    """np.arange wrapper that rounds to 4 dp to avoid float drift."""
    return [round(float(v), 4) for v in np.arange(start, stop, step)]


def _sweep_model(label: str, weights: Path, data_path: Path,
                 conf_values: list[float], iou_values: list[float],
                 split: str, imgsz: int, batch: int, device: str, half: bool,
                 ) -> list[dict]:
    from ultralytics import YOLO

    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")

    model = YOLO(str(weights))
    total = len(conf_values) * len(iou_values)
    logger.info("  %s: sweeping %d conf × %d iou = %d combos",
                label, len(conf_values), len(iou_values), total)

    rows: list[dict] = []
    done = 0
    for conf in conf_values:
        for iou in iou_values:
            metrics = model.val(
                data=str(data_path),
                split=split,
                imgsz=imgsz,
                batch=batch,
                device=device,
                half=half,
                conf=conf,
                iou=iou,
                verbose=False,
            )
            box = metrics.box
            map50   = float(box.map50)
            map5095 = float(box.map)
            prec    = float(box.mp)   # mean precision
            rec     = float(box.mr)   # mean recall
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            rows.append({
                "conf":     conf,
                "iou":      iou,
                "map50":    round(map50,   4),
                "map50_95": round(map5095, 4),
                "precision":round(prec,    4),
                "recall":   round(rec,     4),
                "f1":       round(f1,      4),
            })
            done += 1
            if done % 10 == 0:
                logger.info("    %d/%d  [last conf=%.2f iou=%.2f → mAP50=%.4f]",
                            done, total, conf, iou, map50)

    del model
    return rows


def main() -> None:
    args = parse_args()

    conf_values = _arange(*args.conf_range)
    iou_values  = _arange(*args.iou_range)
    logger.info("Conf values (%d): %s", len(conf_values), conf_values)
    logger.info("IoU  values (%d): %s", len(iou_values),  iou_values)

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = _ROOT / data_path

    out_path = Path(args.out) if args.out else _ROOT / "reports" / "detection" / "best_thresholds.json"
    if not out_path.is_absolute():
        out_path = _ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_per_model: dict[str, dict] = {}

    for spec_str in args.models:
        label, weights = _parse_spec(spec_str)
        logger.info("=== %s ===", label)

        rows = _sweep_model(
            label, weights, data_path,
            conf_values, iou_values,
            args.split, args.imgsz, args.batch, args.device, args.half,
        )

        # Save per-model CSV alongside the JSON
        csv_path = out_path.parent / f"threshold_sweep_{label}.csv"
        fieldnames = ["conf", "iou", "map50", "map50_95", "precision", "recall", "f1"]
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("  Full sweep saved → %s", csv_path)

        # Pick best by mAP50 (tie-break: higher mAP50-95)
        best = max(rows, key=lambda r: (r["map50"], r["map50_95"]))
        best_per_model[label] = {
            "conf":    best["conf"],
            "iou":     best["iou"],
            "map50":   best["map50"],
            "map50_95": best["map50_95"],
            "f1":      best["f1"],
        }

        # Print top-5
        top5 = sorted(rows, key=lambda r: (r["map50"], r["map50_95"]), reverse=True)[:5]
        logger.info("  Top-5 by mAP50:")
        logger.info("    %s", "  ".join(f"{'conf':>6} {'iou':>5} {'mAP50':>7} {'F1':>6}".split()))
        logger.info("    %s", "-" * 34)
        for r in top5:
            logger.info("    conf=%.2f  iou=%.2f  mAP50=%.4f  F1=%.4f",
                        r["conf"], r["iou"], r["map50"], r["f1"])
        logger.info("  >> Best: conf=%.2f  iou=%.2f  mAP50=%.4f  F1=%.4f",
                    best["conf"], best["iou"], best["map50"], best["f1"])

    # Write final JSON
    out_path.write_text(json.dumps(best_per_model, indent=2))
    logger.info("")
    logger.info("Best thresholds saved → %s", out_path)
    logger.info("Use with: --thresholds-file %s", out_path.relative_to(_ROOT))
    logger.info("")
    logger.info("Summary:")
    for label, t in best_per_model.items():
        logger.info("  %-20s conf=%.2f  iou=%.2f  mAP50=%.4f",
                    label, t["conf"], t["iou"], t["map50"])


if __name__ == "__main__":
    main()

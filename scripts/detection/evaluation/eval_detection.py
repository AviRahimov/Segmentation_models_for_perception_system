"""Evaluate a trained detection model on the val or test split.

Computes mAP50, mAP50-95, and per-class AP.  Saves results as JSON under
reports/detection/.

Usage
-----
    python scripts/detection/evaluation/eval_detection.py \\
        --weights weights/detection/yolo26m/round1/best.pt

    python scripts/detection/evaluation/eval_detection.py \\
        --weights weights/detection/yoloe-26m/round1/best.pt \\
        --split test \\
        --data datasets/Detection_Dataset/data.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_detection")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a detection model (mAP50 / mAP50-95)")
    p.add_argument("--weights", required=True,
                   help="Path to trained best.pt checkpoint")
    p.add_argument("--data",    default="datasets/Detection_Dataset/data.yaml",
                   help="Path to YOLO data.yaml")
    p.add_argument("--split",   default="val", choices=["val", "test"],
                   help="Dataset split to evaluate on")
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--batch",   type=int, default=16)
    p.add_argument("--device",  default="0")
    p.add_argument("--conf",    type=float, default=0.25,
                   help="Confidence threshold for NMS (default 0.25)")
    p.add_argument("--iou",     type=float, default=0.7,
                   help="NMS IoU threshold (default 0.7)")
    p.add_argument("--no-half", dest="half", action="store_false", default=True,
                   help="Disable FP16 inference")
    p.add_argument("--out",     default=None,
                   help="Output JSON path (default: reports/detection/eval_*.json)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = _ROOT / weights_path
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = _ROOT / data_path

    model_name = weights_path.parent.parent.name  # e.g. "yolo26m" from .../yolo26m/round1/best.pt

    logger.info("Model:    %s", weights_path)
    logger.info("Data:     %s  (split=%s)", data_path, args.split)

    # YOLO works for both YOLO26 and post-YOLOEPETrainer YOLOE checkpoints
    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        half=args.half,
        conf=args.conf,
        iou=args.iou,
        verbose=False,
    )

    # Extract metrics
    box = metrics.box
    names = model.names  # {0: 'Military Vehicle', 1: 'person'}
    nc = len(names)

    map50    = float(box.map50)
    map5095  = float(box.map)
    ap50_per_class    = [float(v) for v in box.ap50]
    ap5095_per_class  = [float(v) for v in box.ap]

    # Print table
    col_w = max(len(n) for n in names.values()) + 2
    header = f"{'Class':<{col_w}}  {'AP50':>8}  {'AP50-95':>10}"
    logger.info("=" * len(header))
    logger.info("Evaluation results — %s  (split=%s)", model_name, args.split)
    logger.info(header)
    logger.info("-" * len(header))
    for i in range(nc):
        cls_name = names.get(i, str(i))
        logger.info("  %-{col_w}s  %8.4f  %10.4f".replace("{col_w}", str(col_w - 2)),
                    cls_name, ap50_per_class[i] if i < len(ap50_per_class) else float("nan"),
                    ap5095_per_class[i] if i < len(ap5095_per_class) else float("nan"))
    logger.info("-" * len(header))
    logger.info("  %-{col_w}s  %8.4f  %10.4f".replace("{col_w}", str(col_w - 2)),
                "mAP (all)", map50, map5095)
    logger.info("=" * len(header))

    # Save JSON
    if args.out:
        out_path = Path(args.out)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        reports_dir = _ROOT / "reports" / "detection"
        reports_dir.mkdir(parents=True, exist_ok=True)
        out_path = reports_dir / f"eval_{model_name}_{args.split}_{ts}.json"

    result = {
        "model": model_name,
        "weights": str(weights_path),
        "split": args.split,
        "mAP50": map50,
        "mAP50-95": map5095,
        "per_class": {
            names.get(i, str(i)): {
                "AP50":    ap50_per_class[i] if i < len(ap50_per_class) else None,
                "AP50-95": ap5095_per_class[i] if i < len(ap5095_per_class) else None,
            }
            for i in range(nc)
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("Results saved → %s", out_path)


if __name__ == "__main__":
    main()

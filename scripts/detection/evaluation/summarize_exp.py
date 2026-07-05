"""Summarize hyperparameter sweep results vs round1 baseline.

Evaluates every exp/{variant}/best.pt found under each requested model, plus the
round1/best.pt baseline, and produces a sorted comparison table.

Usage
-----
    python scripts/detection/evaluation/summarize_exp.py \\
        --models yolo26m yolo11m \\
        --data datasets/Detection_Dataset/data.yaml

    # Single model, custom output dir:
    python scripts/detection/evaluation/summarize_exp.py \\
        --models yolo26m \\
        --out reports/detection/sweep_yolo26m.md

Output
------
    reports/detection/exp_summary.json   — raw metrics (mAP50, mAP50-95, P, R, F1)
    reports/detection/exp_summary.md     — human-readable sorted markdown table
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("summarize_exp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize exp sweep results vs round1 baseline")
    p.add_argument("--models", nargs="+", default=["yolo26m", "yolo11m"],
                   metavar="MODEL",
                   help="Model names to include (e.g. yolo26m yolo11m)")
    p.add_argument("--data", default="datasets/Detection_Dataset/data.yaml",
                   help="Path to YOLO data.yaml")
    p.add_argument("--split", default="val", choices=["val", "test"],
                   help="Dataset split to evaluate on (default: val)")
    p.add_argument("--conf",   type=float, default=0.25)
    p.add_argument("--iou",    type=float, default=0.70)
    p.add_argument("--imgsz",  type=int,   default=640)
    p.add_argument("--batch",  type=int,   default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--out",    default=None,
                   help="Override output path for .md report")
    p.add_argument("--json",   default=None,
                   help="Override output path for .json metrics")
    return p.parse_args()


def find_checkpoints(model_name: str) -> list[tuple[str, Path]]:
    """Return list of (label, checkpoint_path) for a given model.

    Includes round1/best.pt as baseline plus all exp/{variant}/best.pt files.
    """
    base = _ROOT / "weights" / "detection" / model_name
    entries: list[tuple[str, Path]] = []

    # round1 baseline
    r1 = base / "round1" / "best.pt"
    if r1.exists():
        entries.append((f"{model_name}/round1", r1))
    else:
        logger.warning("round1 checkpoint not found: %s", r1)

    # exp variants — sorted for deterministic order
    exp_root = base / "exp"
    if exp_root.is_dir():
        for variant_dir in sorted(exp_root.iterdir()):
            ckpt = variant_dir / "best.pt"
            if variant_dir.is_dir() and ckpt.exists():
                entries.append((f"{model_name}/exp/{variant_dir.name}", ckpt))

    if not entries:
        logger.warning("No checkpoints found under %s", base)

    return entries


def evaluate_checkpoint(ckpt: Path, data: str, args: argparse.Namespace) -> dict[str, Any]:
    """Run YOLO.val() and return a metrics dict."""
    from ultralytics import YOLO

    logger.info("Evaluating %s ...", ckpt)
    model = YOLO(str(ckpt))
    metrics = model.val(
        data=data,
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        verbose=False,
        plots=False,
    )

    rd = metrics.results_dict if hasattr(metrics, "results_dict") else {}
    map50   = rd.get("metrics/mAP50(B)",    float("nan"))
    map5095 = rd.get("metrics/mAP50-95(B)", float("nan"))
    prec    = rd.get("metrics/precision(B)", float("nan"))
    recall  = rd.get("metrics/recall(B)",    float("nan"))
    f1      = (2 * prec * recall / (prec + recall)) if (prec + recall) > 0 else float("nan")

    return {
        "mAP50":    round(map50, 5),
        "mAP50_95": round(map5095, 5),
        "precision": round(prec, 5),
        "recall":    round(recall, 5),
        "F1":        round(f1, 5),
        "checkpoint": str(ckpt),
    }


def render_markdown(rows: list[dict[str, Any]]) -> str:
    """Render a sorted markdown table from list of metric rows."""
    sorted_rows = sorted(rows, key=lambda r: -r["mAP50"])

    header = (
        "| Model/Variant | mAP50 | mAP50-95 | Precision | Recall | F1 |\n"
        "|---|---|---|---|---|---|\n"
    )
    lines = [header]
    for r in sorted_rows:
        def fmt(v: float) -> str:
            return f"{v:.4f}" if v == v else "—"  # nan check

        line = (
            f"| `{r['label']}` "
            f"| {fmt(r['mAP50'])} "
            f"| {fmt(r['mAP50_95'])} "
            f"| {fmt(r['precision'])} "
            f"| {fmt(r['recall'])} "
            f"| {fmt(r['F1'])} |\n"
        )
        lines.append(line)
    return "".join(lines)


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = _ROOT / data_path
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")

    # Collect all checkpoints to evaluate
    all_checkpoints: list[tuple[str, Path]] = []
    for model_name in args.models:
        all_checkpoints.extend(find_checkpoints(model_name))

    if not all_checkpoints:
        logger.error("No checkpoints found. Train with train_exp.py first.")
        sys.exit(1)

    logger.info("Found %d checkpoint(s) to evaluate.", len(all_checkpoints))

    # Evaluate
    rows: list[dict[str, Any]] = []
    for label, ckpt in all_checkpoints:
        try:
            result = evaluate_checkpoint(ckpt, str(data_path), args)
            result["label"] = label
            rows.append(result)
            logger.info("  %s — mAP50=%.4f  mAP50-95=%.4f  P=%.4f  R=%.4f  F1=%.4f",
                        label, result["mAP50"], result["mAP50_95"],
                        result["precision"], result["recall"], result["F1"])
        except Exception as exc:
            logger.error("Failed to evaluate %s: %s", ckpt, exc)

    if not rows:
        logger.error("All evaluations failed.")
        sys.exit(1)

    # Output paths
    out_dir = _ROOT / "reports" / "detection"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = Path(args.json) if args.json else out_dir / "exp_summary.json"
    md_path   = Path(args.out)  if args.out  else out_dir / "exp_summary.md"

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    logger.info("Metrics JSON → %s", json_path)

    md_content = (
        "# Hyperparameter Sweep — Detection Model Comparison\n\n"
        f"> Split: `{args.split}` | conf={args.conf} | iou={args.iou} | imgsz={args.imgsz}\n\n"
        "Sorted by mAP50 descending. `round1` entries are the baseline.\n\n"
    )
    md_content += render_markdown(rows)
    md_path.write_text(md_content, encoding="utf-8")
    logger.info("Markdown report → %s", md_path)

    # Print table to stdout too
    logger.info("")
    logger.info("%-40s  %8s  %10s  %9s  %8s  %6s",
                "Model/Variant", "mAP50", "mAP50-95", "Precision", "Recall", "F1")
    logger.info("-" * 90)
    for r in sorted(rows, key=lambda x: -x["mAP50"]):
        def _f(v: float) -> str:
            return f"{v:.4f}" if v == v else "   —  "
        logger.info("%-40s  %8s  %10s  %9s  %8s  %6s",
                    r["label"], _f(r["mAP50"]), _f(r["mAP50_95"]),
                    _f(r["precision"]), _f(r["recall"]), _f(r["F1"]))


if __name__ == "__main__":
    main()

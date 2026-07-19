#!/usr/bin/env python3
"""Fit per-class confidence-calibration temperatures for one checkpoint.

Evaluates a single checkpoint on a benchmark split using the same collapsed
2-class scheme (``Military Vehicle`` / ``person``) as leaderboard.py, then
fits a temperature per class via ``postprocess.calibration.fit_temperature``
(scalar NLL minimization over every prediction's TP/FP label at IoU 0.5 —
not just the counts at one operating point). Output is a small JSON file
consumed at runtime by ``postprocess.calibration.load_temperatures`` when
``postprocess.calibration.enabled: true`` in config.yaml.

Class-name note
----------------
The eval pipeline's collapsed benchmark scheme names the vehicle class
"Military Vehicle"; the production config.yaml profile (`2class` /
`rfdetr_2class`) names the same class "mil vehicle" (`Detection.class_name`
at actual pipeline runtime). Since ``postprocess.calibration`` looks up a
detection's calibration temperature by its runtime ``class_name``, the
output JSON's keys are remapped from the eval names to the production names
via ``--class-names`` (positional-order-matched to the fixed benchmark order
[vehicle, person]) before saving — default matches the currently-active
2-class profile. If you retarget calibration at a different class scheme,
pass matching names explicitly.

Usage
-----
    python scripts/detection/evaluation/fit_calibration.py \\
        --weights weights/detection/rfdetr-2xl/detection_dataset/coco/best.pt
    # interactive checkpoint + benchmark picker if --weights is omitted
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / "scripts" / "detection" / "training"))

from _ap_utils import (  # noqa: E402
    collect_predictions,
    collect_predictions_rfdetr,
    infer_rfdetr_profile,
    is_rfdetr_checkpoint,
    load_rfdetr_for_eval,
    load_yolo_gts,
    scores_and_labels,
)
from leaderboard import _COLLAPSE, _discover_benchmark_dirs, _discover_checkpoints, _label  # noqa: E402
from _survey_common import _ask  # noqa: E402

from perception.postprocess.calibration import fit_temperatures, save_temperatures  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("fit_calibration")

_BENCHMARK_CLASSES = ["Military Vehicle", "person"]  # eval-collapsed order, fixed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=str, default=None,
                   help="Checkpoint to calibrate (interactive picker if omitted)")
    p.add_argument("--benchmark", type=str, default=None,
                   help="datasets/... split with images/+labels/ (interactive picker if omitted)")
    p.add_argument("--class-names", type=str, nargs=2, default=["mil vehicle", "person"],
                   metavar=("VEHICLE_NAME", "PERSON_NAME"),
                   help="Production class names (in [vehicle, person] order) the output "
                        "JSON should be keyed by — must match config.yaml's active profile")
    p.add_argument("--iou-thr", type=float, default=0.5)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.05,
                   help="Prediction confidence floor fed to the model — must stay low "
                        "so low-score predictions are still available to fit against")
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--out", type=str, default=None,
                   help="Output JSON path (default: weights/detection/calibration/<label>.json)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.weights is None:
        ckpts = _discover_checkpoints()
        if not ckpts:
            raise SystemExit("No checkpoints found under weights/detection/.")
        options = [(_label(c), str(c.relative_to(_ROOT))) for c in ckpts]
        pick = _ask("Which checkpoint to calibrate?", options, default_idx=0)[0]
        ckpt = ckpts[pick]
    else:
        ckpt = Path(args.weights)
        if not ckpt.is_absolute():
            ckpt = _ROOT / ckpt
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    if args.benchmark is None:
        candidates = _discover_benchmark_dirs()
        if not candidates:
            raise SystemExit("No images/+labels/ benchmark dirs found under datasets/.")
        options = [(label, f"{n} images") for label, _, _, n in candidates]
        default_idx = next((i for i, (label, *_ ) in enumerate(candidates)
                            if label == "Detection_Dataset/valid"), 0)
        pick = _ask("Which data to fit calibration on?", options, default_idx=default_idx)[0]
        _, img_dir, lbl_dir, _ = candidates[pick]
    else:
        bench_dir = Path(args.benchmark)
        if not bench_dir.is_absolute():
            bench_dir = _ROOT / bench_dir
        img_dir, lbl_dir = bench_dir / "images", bench_dir / "labels"
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            raise SystemExit(f"Benchmark needs images/ and labels/ under {bench_dir}")

    pairs = load_yolo_gts(img_dir, lbl_dir, _BENCHMARK_CLASSES)
    logger.info("Benchmark: %s  (%d images)", img_dir.parent, len(pairs))

    label = _label(ckpt)
    is_rfdetr = is_rfdetr_checkpoint(ckpt)
    logger.info("Evaluating %s ...", label)
    if is_rfdetr:
        model = load_rfdetr_for_eval(ckpt, confidence_floor=args.conf)
        names = [c.name for c in infer_rfdetr_profile(ckpt)]
        preds, gts = collect_predictions_rfdetr(model, pairs, _COLLAPSE)
    else:
        from ultralytics import YOLO
        model = YOLO(str(ckpt))
        names = list((model.names or {}).values())
        preds, gts = collect_predictions(
            model, pairs, _COLLAPSE, imgsz=args.imgsz, conf=args.conf, device=args.device,
        )
    unknown = [n for n in names if n not in _COLLAPSE and n != "civilian_vehicle"]
    if unknown:
        logger.warning("classes not in collapse map (dropped from fitting): %s", unknown)

    per_class = {}
    for cls in _BENCHMARK_CLASSES:
        scores, tp = scores_and_labels(preds, gts, cls, iou_thr=args.iou_thr)
        per_class[cls] = (scores, tp)
        n_tp, n_fp = int(tp.sum()), int(len(tp) - tp.sum())
        logger.info("  %-16s %5d preds (%d TP / %d FP)", cls, len(scores), n_tp, n_fp)

    temperatures = fit_temperatures(per_class)
    for cls, t in temperatures.items():
        logger.info("  %-16s T = %.3f", cls, t)

    # Remap eval-scheme names -> production config.yaml class names before saving,
    # so postprocess.calibration.apply_calibration's runtime lookup (keyed by
    # Detection.class_name) actually finds these entries — see module docstring.
    vehicle_name, person_name = args.class_names
    name_map = {"Military Vehicle": vehicle_name, "person": person_name}
    prod_temperatures = {name_map.get(k, k): v for k, v in temperatures.items()}

    out_path = Path(args.out) if args.out else (
        _ROOT / "weights" / "detection" / "calibration" / f"{label.replace('/', '_')}.json"
    )
    if not out_path.is_absolute():
        out_path = _ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_temperatures(prod_temperatures, out_path)
    logger.info("Saved -> %s", out_path.relative_to(_ROOT))
    logger.info("Set postprocess.calibration.enabled: true and "
                "temperatures_path: %s in config.yaml to use it.",
                out_path.relative_to(_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

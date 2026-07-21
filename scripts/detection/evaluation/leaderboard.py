"""Experiment leaderboard — one ranked table over every trained checkpoint.

Evaluates ALL checkpoints under weights/detection/ on the fixed real-image
benchmark (Detection_Dataset val) using a single scheme-agnostic evaluation
path: predictions are collapsed to the 2-class benchmark scheme (identity for
2-class models; tank/soldier/... maps for fine-grained models), then scored
with the tested all-point AP50 implementation from _ap_utils.py. This makes
2-class, 6-class, and foreign (e.g. verrckter 8-class) models directly
comparable on one table.

Results are cached per checkpoint (path+mtime+size+eval-params) in
reports/detection/leaderboard_cache.json — re-runs only evaluate new models.

Usage
-----
    python scripts/detection/evaluation/leaderboard.py               # full board
    python scripts/detection/evaluation/leaderboard.py --tta        # + TTA rows (yolo11-family)
    python scripts/detection/evaluation/leaderboard.py \\
        --only weights/detection/yolo11m/merged_2class/noaug/best.pt ...   # subset
    # Held-out final check (when a test set exists):
    python scripts/detection/evaluation/leaderboard.py --benchmark path/to/test \\
        --only <top-checkpoints...>

Output: reports/detection/leaderboard.md (+ console table)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / "scripts" / "detection" / "training"))

import numpy as np  # noqa: E402

from _ap_utils import (  # noqa: E402
    ap_per_class,
    collect_predictions,
    collect_predictions_rfdetr,
    false_positives,
    infer_rfdetr_profile,
    is_rfdetr_checkpoint,
    load_rfdetr_for_eval,
    load_yolo_gts,
    operating_point,
    size_bucketed_recall,
    threshold_sweep,
)
from _survey_common import _ask  # noqa: E402

_DEPLOY_CONF = 0.40  # fixed operating point for P / R / FP-per-image columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("leaderboard")

_BENCHMARK_CLASSES = ["Military Vehicle", "person"]

# Known class-name → benchmark-class mappings. Identity entries included so a
# single lookup covers every scheme; unknown names are dropped with a warning.
_COLLAPSE: dict[str, str] = {
    # benchmark scheme (identity)
    "Military Vehicle": "Military Vehicle",
    "person": "person",
    # merged_6class scheme
    "tank": "Military Vehicle",
    "truck": "Military Vehicle",
    "armored_vehicle": "Military Vehicle",
    "soldier": "person",
    "civilian": "person",
    # civilian_vehicle intentionally absent → dropped (not a benchmark class)
    # verrckter 8-class scheme
    "Tanks": "Military Vehicle",
    "Trucks": "Military Vehicle",
    "APC": "Military Vehicle",
    "IFV": "Military Vehicle",
    "IMV": "Military Vehicle",
    "ENG": "Military Vehicle",
    "ART": "Military Vehicle",
    "MRL": "Military Vehicle",
    # kaggle 12-class scheme (raw dataset)
    "military_tank": "Military Vehicle",
    "military_truck": "Military Vehicle",
    "military_vehicle": "Military Vehicle",
    "camouflage_soldier": "person",
    # 'soldier'/'civilian' handled by the 6-class entries above
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ranked leaderboard over all trained detectors")
    p.add_argument("--benchmark", default=None,
                   help="Benchmark split dir containing images/ and labels/. "
                        "Omit to be asked interactively (Enter = Detection_Dataset/valid).")
    p.add_argument("--only", nargs="*", default=None,
                   help="Evaluate only these checkpoints (paths); default: all found")
    p.add_argument("--tta", action="store_true",
                   help="Add augment=True rows for models that support it")
    p.add_argument("--imgsz", type=int, default=1280,
                   help="Inference size for the benchmark (default 1280)")
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--device", default="0")
    p.add_argument("--out", default=None,
                   help="Output markdown path. Default: reports/detection/leaderboard.md "
                        "for the valid split, leaderboard_{split}.md for any other "
                        "(so a test-set run never clobbers the val results).")
    p.add_argument("--thresholds", action="store_true",
                   help="Write per-model best-F1 threshold recommendations")
    p.add_argument("--fp-gallery", action="store_true", dest="fp_gallery",
                   help="Save annotated false-positive crops per model")
    return p.parse_args()


def _discover_benchmark_dirs() -> list[tuple[str, Path, Path, int]]:
    """Every images/+labels/ pair under datasets/ -> [(label, img_dir, lbl_dir, n_images)].

    Not hardcoded to "valid"/"test": scans every subdirectory of
    Detection_Dataset/ (covers valid, test, and anything else dropped there
    later) plus every top-level datasets/* dir that itself directly holds
    images/+labels/ (a wholly separate benchmark dataset dropped elsewhere) —
    same discovery spirit as _survey_common._scan_datasets().

    Excludes anything named "train": it also has images/+labels/, but
    evaluating a checkpoint on its own training data gives a misleadingly
    optimistic score, not a meaningful benchmark choice — confirmed as a real
    footgun while testing this (picked it by accident, got 0.98 mAP50).
    """
    datasets_root = _ROOT / "datasets"
    out: list[tuple[str, Path, Path, int]] = []
    if not datasets_root.is_dir():
        return out

    def _add(label: str, d: Path) -> None:
        if d.name.lower() == "train":
            return
        img_dir, lbl_dir = d / "images", d / "labels"
        if img_dir.is_dir() and lbl_dir.is_dir():
            n = sum(1 for p in img_dir.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
            out.append((label, img_dir, lbl_dir, n))

    det_ds = datasets_root / "Detection_Dataset"
    if det_ds.is_dir():
        for sub in sorted(det_ds.iterdir()):
            if sub.is_dir():
                _add(f"Detection_Dataset/{sub.name}", sub)

    for d in sorted(datasets_root.iterdir()):
        if d.is_dir() and d != det_ds:
            _add(d.name, d)

    return out


def _discover_checkpoints() -> list[Path]:
    det = _ROOT / "weights" / "detection"
    return [p for p in sorted(det.glob("**/best.pt")) if p.parent.name != "weights"]


def _label(ckpt: Path) -> str:
    det = _ROOT / "weights" / "detection"
    try:
        return "/".join(ckpt.relative_to(det).parts[:-1])
    except ValueError:
        # Foreign checkpoint (e.g. datasets/verrckter_military_vehicle/best.pt)
        try:
            return str(ckpt.relative_to(_ROOT))
        except ValueError:
            return str(ckpt)


_CACHE_SCHEMA = "v3"  # bump when row fields change → old entries recompute


def _cache_key(ckpt: Path, args, tta: bool) -> str:
    st = ckpt.stat()
    raw = (f"{_CACHE_SCHEMA}|{ckpt}|{st.st_mtime_ns}|{st.st_size}|"
           f"{args.benchmark}|{args.imgsz}|{args.conf}|tta={tta}")
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _save_fp_gallery(fps, label: str, cap: int = 30) -> Path:
    """Annotated context crops of false positives → reports/detection/fp_gallery/."""
    import cv2

    out_dir = _ROOT / "reports" / "detection" / "fp_gallery" / label.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("fp_*.png"):
        old.unlink()
    for i, p in enumerate(fps[:cap]):
        img = cv2.imread(p.image_id)
        if img is None:
            continue
        h, w = img.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in p.box)
        # context margin: 40% of box size on each side
        mx, my = int((x2 - x1) * 0.4) + 20, int((y2 - y1) * 0.4) + 20
        cx1, cy1 = max(0, x1 - mx), max(0, y1 - my)
        cx2, cy2 = min(w, x2 + mx), min(h, y2 + my)
        crop = img[cy1:cy2, cx1:cx2].copy()
        cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), (0, 0, 255), 2)
        cv2.putText(crop, f"{p.class_name} {p.score:.2f}", (x1 - cx1, max(12, y1 - cy1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        stem = Path(p.image_id).stem[:40]
        cv2.imwrite(str(out_dir / f"fp_{p.score:.2f}_{p.class_name.replace(' ', '')}_{i:02d}_{stem}.png"), crop)
    return out_dir


def _load_experiments() -> dict[str, dict]:
    """run_dir (relative) → last experiment record for that dir."""
    path = _ROOT / "reports" / "detection" / "experiments.jsonl"
    out: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
                out[rec.get("run_dir", "")] = rec
            except json.JSONDecodeError:
                continue
    return out


def main() -> int:
    args = parse_args()

    if args.benchmark is None:
        # Not passed on the CLI -> ask interactively (EOF/non-interactive
        # contexts fall back to the recommended default via _ask itself).
        candidates = _discover_benchmark_dirs()
        if not candidates:
            raise SystemExit("No images/+labels/ benchmark dirs found under datasets/.")
        options = [(label, f"{n} images") for label, _, _, n in candidates]
        default_idx = next((i for i, (label, *_ ) in enumerate(candidates)
                            if label == "Detection_Dataset/valid"), 0)
        pick = _ask("Which data to evaluate on?", options, default_idx=default_idx)[0]
        split_label, img_dir, lbl_dir, _ = candidates[pick]
        bench_dir = img_dir.parent
        # _cache_key() hashes args.benchmark directly — must be the resolved
        # path, not None, or val/test picks would collide on the same cache
        # entry and silently reuse the wrong split's cached results.
        args.benchmark = str(bench_dir.relative_to(_ROOT))
    else:
        bench_dir = Path(args.benchmark)
        if not bench_dir.is_absolute():
            bench_dir = _ROOT / bench_dir
        img_dir, lbl_dir = bench_dir / "images", bench_dir / "labels"
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            raise SystemExit(f"Benchmark needs images/ and labels/ under {bench_dir}")
        split_label = None

    if args.out is None:
        if split_label in (None, "Detection_Dataset/valid"):
            args.out = "reports/detection/leaderboard.md"
        else:
            args.out = f"reports/detection/leaderboard_{bench_dir.name}.md"

    pairs = load_yolo_gts(img_dir, lbl_dir, _BENCHMARK_CLASSES)
    logger.info("Benchmark: %s  (%d images)", bench_dir, len(pairs))

    if args.only:
        ckpts = []
        for c in args.only:
            p = Path(c)
            ckpts.append(p if p.is_absolute() else _ROOT / p)
    else:
        ckpts = _discover_checkpoints()
    logger.info("Checkpoints to evaluate: %d", len(ckpts))

    cache_path = _ROOT / "reports" / "detection" / "leaderboard_cache.json"
    cache: dict = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    experiments = _load_experiments()

    rows: list[dict] = []
    threshold_recs: dict[str, dict] = {}
    for ckpt in ckpts:
        if not ckpt.exists():
            logger.warning("missing checkpoint: %s", ckpt)
            continue
        label = _label(ckpt)
        is_rfdetr = is_rfdetr_checkpoint(ckpt)
        # RF-DETR has no augment=True inference path (unlike Ultralytics TTA).
        variants = [False] if is_rfdetr else [False] + ([True] if args.tta else [])
        for tta in variants:
            key = _cache_key(ckpt, args, tta)
            # Cache short-circuits only when no fresh predictions are needed.
            if key in cache and not (args.thresholds or args.fp_gallery):
                rows.append(cache[key])
                continue
            if is_rfdetr:
                model = load_rfdetr_for_eval(ckpt, confidence_floor=args.conf)
                names = [c.name for c in infer_rfdetr_profile(ckpt)]
            else:
                from ultralytics import YOLO
                model = YOLO(str(ckpt))
                names = list((model.names or {}).values())
            unknown = [n for n in names if n not in _COLLAPSE and n != "civilian_vehicle"]
            if unknown:
                logger.warning("%s: classes not in collapse map (dropped): %s",
                               label, unknown)
            if tta and "yolo11" not in label and "yolo11" not in str(names):
                continue  # TTA only meaningful where supported
            logger.info("Evaluating %s%s ...", label, " +TTA" if tta else "")
            try:
                if is_rfdetr:
                    preds, gts = collect_predictions_rfdetr(model, pairs, _COLLAPSE)
                else:
                    preds, gts = collect_predictions(
                        model, pairs, _COLLAPSE,
                        imgsz=args.imgsz, conf=args.conf, device=args.device,
                        augment=tta,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.error("  failed: %s", exc)
                continue

            ap = ap_per_class(preds, gts, _BENCHMARK_CLASSES)
            valid = [v for v in ap.values() if v == v]
            map50 = float(np.mean(valid)) if valid else float("nan")
            op = operating_point(preds, gts, _BENCHMARK_CLASSES, len(pairs),
                                 conf_thr=_DEPLOY_CONF)
            sizes = size_bucketed_recall(preds, gts, _BENCHMARK_CLASSES,
                                         conf_thr=_DEPLOY_CONF)

            def _macro(field: str) -> float:
                vals = [op[c][field] for c in _BENCHMARK_CLASSES
                        if op[c][field] == op[c][field]]
                return float(np.mean(vals)) if vals else float("nan")

            def _szstr(cls: str) -> str:
                s = sizes.get(cls)
                if not s:
                    return ""
                r = s["recall"]
                fmt = lambda v: f"{v:.2f}" if v == v else "—"
                return f"{fmt(r['small'])}/{fmt(r['medium'])}/{fmt(r['large'])}"

            exp = experiments.get(label and f"weights/detection/{label}", {})
            row = {
                "label": label + (" +TTA" if tta else ""),
                "scheme": f"{len(names)}c",
                "mAP50": round(map50, 4),
                "vehicle_AP50": round(ap.get("Military Vehicle", float("nan")), 4),
                "person_AP50": round(ap.get("person", float("nan")), 4),
                "P40": round(_macro("precision"), 4),
                "R40": round(_macro("recall"), 4),
                "fp_img": round(float(sum(op[c]["fp"] for c in _BENCHMARK_CLASSES))
                                / len(pairs), 3),
                "fn_img": round(float(sum(op[c]["fn"] for c in _BENCHMARK_CLASSES))
                                / len(pairs), 3),
                "veh_szrec": _szstr("Military Vehicle"),
                "per_szrec": _szstr("person"),
                "size_bounds": {c: sizes[c]["boundaries_px"] for c in sizes},
                "dataset": exp.get("dataset", ""),
                "imgsz_train": exp.get("imgsz", ""),
                "checkpoint": str(ckpt.relative_to(_ROOT)),
            }
            rows.append(row)
            cache[key] = row
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=1))

            if args.thresholds:
                rec = {cls: threshold_sweep(preds, gts, cls, len(pairs))["best"]
                       for cls in _BENCHMARK_CLASSES}
                threshold_recs[row["label"]] = rec
            if args.fp_gallery:
                fps = false_positives(preds, gts, _BENCHMARK_CLASSES,
                                      conf_thr=_DEPLOY_CONF)
                gal = _save_fp_gallery(fps, row["label"])
                logger.info("  %d FPs @%.2f → %s", len(fps), _DEPLOY_CONF,
                            gal.relative_to(_ROOT))
            del model

    rows.sort(key=lambda r: -(r["mAP50"] if r["mAP50"] == r["mAP50"] else -1))

    # Console table
    logger.info("")
    logger.info("%-62s %-4s %7s %7s %7s %6s %6s %7s %7s",
                "Model", "cls", "mAP50", "vehAP", "perAP", "P@.4", "R@.4", "FP/img", "FN/img")
    logger.info("-" * 112)
    for r in rows:
        logger.info("%-62s %-4s %7.4f %7.4f %7.4f %6.3f %6.3f %7.3f %7.3f",
                    r["label"], r["scheme"], r["mAP50"],
                    r["vehicle_AP50"], r["person_AP50"],
                    r.get("P40", float("nan")), r.get("R40", float("nan")),
                    r.get("fp_img", float("nan")), r.get("fn_img", float("nan")))

    # Markdown
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    size_note = ""
    if rows and rows[0].get("size_bounds"):
        sb = rows[0]["size_bounds"]
        size_note = (" | size buckets (data-driven terciles): "
                     + ", ".join(f"{c} ≤{b[0]}px/≤{b[1]}px/&gt;" for c, b in sb.items()))
    lines = [
        "# Detection Leaderboard\n\n",
        f"> Benchmark: `{bench_dir.relative_to(_ROOT)}` ({len(pairs)} images) | "
        f"collapsed AP50, all schemes comparable | imgsz={args.imgsz} conf={args.conf} | "
        f"P/R/FP at conf={_DEPLOY_CONF}{size_note}\n\n",
        "| # | Model | Cls | mAP50 | Veh AP | Per AP | P@.4 | R@.4 | FP/img | FN/img "
        "| Veh R s/m/l | Per R s/m/l | Trained on |\n",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n",
    ]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | `{r['label']}` | {r['scheme']} | {r['mAP50']:.4f} "
            f"| {r['vehicle_AP50']:.4f} | {r['person_AP50']:.4f} "
            f"| {r.get('P40', float('nan')):.3f} | {r.get('R40', float('nan')):.3f} "
            f"| {r.get('fp_img', float('nan')):.3f} | {r.get('fn_img', float('nan')):.3f} "
            f"| {r.get('veh_szrec', '')} | {r.get('per_szrec', '')} "
            f"| {r['dataset']} |\n")
    out_path.write_text("".join(lines))
    logger.info("")
    logger.info("Leaderboard → %s", out_path)

    # Threshold recommendations
    if args.thresholds and threshold_recs:
        rec_path = _ROOT / "reports" / "detection" / "threshold_recommendations.md"
        rl = ["# Per-class threshold recommendations (best F1 on the benchmark)\n\n",
              "| Model | Class | Best conf | Precision | Recall | F1 | FP/img |\n",
              "|---|---|---|---|---|---|---|\n"]
        for label, rec in threshold_recs.items():
            for cls, best in rec.items():
                if best is None:
                    continue
                rl.append(f"| `{label}` | {cls} | **{best['conf']:.2f}** "
                          f"| {best['precision']:.3f} | {best['recall']:.3f} "
                          f"| {best['f1']:.3f} | {best['fp_per_image']:.3f} |\n")
        rec_path.write_text("".join(rl))
        logger.info("Threshold recommendations → %s", rec_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare N semantic segmentation models on a labeled or qualitative-only dataset.

Three dataset adapters are built in:

  orfd    ORFD's fillcolor freespace GT (the project's primary benchmark).
          255=traversable, 0=non-traversable, 128=sky -- same 3-class encoding
          used everywhere else in training/eval (see orfd_torch.py).
  zikim   off_road_zikim's Cityscapes-style color GT, converted to the same
          3-class scheme (road->traversable, ground->non_traversable,
          sky->sky, unlabeled/other->ignored).
  fcdd    Floripa Coast Driving Dataset images -- QUALITATIVE ONLY. FCDD ships
          its own 17-class GT, but no 17->3 collapse mapping has been decided
          yet, so this adapter deliberately ignores its GT and only renders
          model predictions side by side (no metrics).

Omit --dataset and/or --models to pick interactively from what's actually on
disk under datasets/segmentation/ and weights/segmentation/.

For labeled datasets (orfd, zikim), reports both:
  * the binary traversable-only freespace metric (mean/median IoU + micro
    precision/recall/F1 on valid pixels) -- the narrower "is the drivable
    corridor right" check, and
  * the 3-class mIoU (non_traversable/traversable/sky) via the same
    compute_miou() used by every training/eval script in this project.

Writes (default):
  <output-dir>/strips/<name>.png   -- one horizontal comparison strip per frame
  <output-dir>/README.txt
  <output-dir>/performance_summary.{json,md}   (labeled datasets only)

Optional ``--single-mosaic`` stacks all strips into one PNG.
Optional ``--stitch-video`` (auto-on for zikim, since its frames are a
continuous recording) encodes the strips into an .mp4 instead.

Usage
-----
    # interactive: pick a dataset and models from what's on disk
    PYTHONPATH=src python scripts/segmentation/evaluation/compare_semantic_models.py

    PYTHONPATH=src python scripts/segmentation/evaluation/compare_semantic_models.py \\
        --dataset orfd --models segformer-b2 mask2former-large --samples 20

    # a specific checkpoint recipe, not just the architecture default
    PYTHONPATH=src python scripts/segmentation/evaluation/compare_semantic_models.py \\
        --dataset orfd --models "segformer-b2:weights/segmentation/orfd/lora/segformer-b2/best.pth"

    # params + latency only, any --models, no dataset needed
    PYTHONPATH=src python scripts/segmentation/evaluation/compare_semantic_models.py \\
        --latency-only --models segformer-b2 mask2former-large auriganet
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts" / "segmentation" / "training"))

from _orfd_common import compute_miou  # noqa: E402
from _semantic_eval_common import (  # noqa: E402
    BASELINE_CHOICES,
    ask_choice,
    ask_multi_choice,
    measure_forward_latency_ms,
    parse_model_spec,
    render_panel,
    resolve_weights,
    scan_segmentation_datasets,
    scan_semantic_checkpoints,
)
from perception.config.loader import load_config  # noqa: E402
from perception.config.schema import ClassDef, SemanticModelCfg  # noqa: E402
from perception.datasets.orfd_torch import _remap_label  # noqa: E402
from perception.models.backends.pytorch import PyTorchBackend  # noqa: E402
from perception.models.factory import build_semantic_model  # noqa: E402

logger = logging.getLogger("compare_semantic_models")

IGNORE_INDEX = 255  # matches _orfd_common.IGNORE_INDEX / orfd_torch.py's convention
NUM_CLASSES = 3      # non_traversable, traversable, sky

_TRAV_CLASSES: tuple[ClassDef, ...] = (
    ClassDef(name="non_traversable", text_prompt="-", display_mode="mask_only",
             color_rgb=(220, 40, 40), is_semantic=True, native_indices={}),
    ClassDef(name="traversable", text_prompt="-", display_mode="mask_only",
             color_rgb=(40, 255, 140), is_semantic=True, native_indices={}),
    ClassDef(name="sky", text_prompt="-", display_mode="mask_only",
             color_rgb=(80, 160, 255), is_semantic=True, native_indices={}),
)

# zikim label name -> ORFD class index (255 = ignore, matches IGNORE_INDEX)
_ZIKIM_TO_ORFD: dict[str, int] = {"road": 1, "sky": 2, "ground": 0}


@dataclass(frozen=True)
class EvalFrame:
    strip_name: str
    img_path: Path
    img_bgr: np.ndarray
    gt_user: np.ndarray | None  # (H, W) uint8 in {0, 1, 2, 255}; None = qualitative-only


# --------------------------------------------------------------------------- #
# Dataset adapters
# --------------------------------------------------------------------------- #


def gather_orfd_pairs(training_root: Path) -> list[tuple[Path, Path]]:
    """Align ``image_data/*.png`` with ``gt_image/<id>_fillcolor.png``."""
    img_dir = training_root / "image_data"
    gt_dir = training_root / "gt_image"
    if not img_dir.is_dir() or not gt_dir.is_dir():
        logger.error("Expected ORFD layout under %s: image_data/ and gt_image/ missing.", training_root)
        return []
    imgs = {p.stem: p for p in img_dir.glob("*.png")}
    stems = [s for s in sorted(imgs.keys()) if (gt_dir / f"{s}_fillcolor.png").is_file()]
    pairs = [(imgs[s], gt_dir / f"{s}_fillcolor.png") for s in stems]
    logger.info("Found %d image/GT pairs under %s", len(pairs), training_root)
    return pairs


def _orfd_frames(dataset_root: Path, *, split: str, samples: int, seed: int) -> list[EvalFrame]:
    training_root = dataset_root / split
    pairs = gather_orfd_pairs(training_root)
    if not pairs:
        return []
    if samples and samples > 0:
        rng = random.Random(seed)
        picks = rng.sample(pairs, k=min(samples, len(pairs)))
    else:
        picks = pairs

    frames: list[EvalFrame] = []
    for img_path, gt_path in picks:
        img_bgr = cv2.imread(str(img_path))
        gt_raw = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
        if img_bgr is None or gt_raw is None:
            logger.warning("Read failed %s / %s; skip.", img_path, gt_path)
            continue
        gt_gray = gt_raw[..., 0] if gt_raw.ndim == 3 else gt_raw
        gt_gray = gt_gray.astype(np.uint8, copy=False)
        if gt_gray.shape[:2] != img_bgr.shape[:2]:
            gt_gray = cv2.resize(gt_gray, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        gt_user = _remap_label(gt_gray)
        frames.append(EvalFrame(strip_name=_name_key("orfd", img_path.stem),
                                 img_path=img_path, img_bgr=img_bgr, gt_user=gt_user))
    return frames


def _load_zikim_color_to_orfd(config_path: Path) -> dict[tuple[int, int, int], int]:
    labels = json.loads(config_path.read_text())["labels"]
    out: dict[tuple[int, int, int], int] = {}
    for name, spec in labels.items():
        rgb = tuple(int(v) for v in spec["color"])
        out[rgb] = _ZIKIM_TO_ORFD.get(name, IGNORE_INDEX)
    return out


def _zikim_color_mask_to_orfd(color_mask_rgb: np.ndarray, color_to_orfd: dict[tuple[int, int, int], int]) -> np.ndarray:
    """(H,W,3) RGB color mask -> (H,W) uint8 {0,1,2,255}."""
    h, w = color_mask_rgb.shape[:2]
    flat = color_mask_rgb.reshape(-1, 3)
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    class_per_uniq = np.full(len(uniq), IGNORE_INDEX, dtype=np.uint8)
    for i, rgb in enumerate(uniq):
        class_per_uniq[i] = color_to_orfd.get(tuple(int(v) for v in rgb), IGNORE_INDEX)
    return class_per_uniq[inv].reshape(h, w)


def _gather_zikim_frames(val_dir: Path) -> list[tuple[Path, Path, int]]:
    """Pair base images with their _color_mask.png, sorted by trailing frame index."""
    pairs = []
    for img_path in val_dir.glob("*.png"):
        stem = img_path.stem
        if stem.endswith(("_mask", "_color_mask", "_watershed_mask")):
            continue
        color_mask = val_dir / f"{stem}_color_mask.png"
        if not color_mask.is_file():
            continue
        m = re.search(r"(\d+)$", stem)
        idx = int(m.group(1)) if m else 0
        pairs.append((img_path, color_mask, idx))
    pairs.sort(key=lambda t: t[2])
    logger.info("Found %d image/color_mask pairs in %s", len(pairs), val_dir)
    return pairs


def _pick_zikim_val_subdir(dataset_root: Path, preferred: str) -> str | None:
    val_root = dataset_root / "val"
    if (val_root / preferred).is_dir():
        return preferred
    subdirs = sorted(p.name for p in val_root.iterdir()) if val_root.is_dir() else []
    return subdirs[0] if subdirs else None


def _zikim_frames(dataset_root: Path, *, val_subdir: str, samples: int, seed: int) -> list[EvalFrame]:
    resolved = _pick_zikim_val_subdir(dataset_root, val_subdir)
    if resolved is None:
        logger.error("No val subdirectories found under %s/val", dataset_root)
        return []
    if resolved != val_subdir:
        logger.info("val subdir %r not found; using %r instead.", val_subdir, resolved)
    val_dir = dataset_root / "val" / resolved
    color_to_orfd = _load_zikim_color_to_orfd(dataset_root / "config_zikim.json")
    triples = _gather_zikim_frames(val_dir)
    if not triples:
        return []
    if samples and samples > 0:
        rng = random.Random(seed)
        triples = sorted(rng.sample(triples, k=min(samples, len(triples))), key=lambda t: t[2])

    frames: list[EvalFrame] = []
    for img_path, color_mask_path, idx in triples:
        img_bgr = cv2.imread(str(img_path))
        color_mask = cv2.imread(str(color_mask_path))
        if img_bgr is None or color_mask is None:
            continue
        color_mask_rgb = cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB)
        gt_user = _zikim_color_mask_to_orfd(color_mask_rgb, color_to_orfd)
        frames.append(EvalFrame(strip_name=_name_key("zikim", f"{idx:06d}_{img_path.stem}"),
                                 img_path=img_path, img_bgr=img_bgr, gt_user=gt_user))
    return frames


def _fcdd_frames(dataset_root: Path, *, split: str, samples: int, seed: int) -> list[EvalFrame]:
    """Qualitative only -- FCDD's own 17-class GT is deliberately ignored (no
    verified collapse mapping to non_traversable/traversable/sky yet)."""
    im_dir = dataset_root / split / "im"
    if not im_dir.is_dir():
        logger.error("Expected FCDD layout under %s: %s/im missing.", dataset_root, split)
        return []
    imgs = sorted(p for p in im_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if samples and samples > 0:
        rng = random.Random(seed)
        imgs = rng.sample(imgs, k=min(samples, len(imgs)))
    frames: list[EvalFrame] = []
    for img_path in imgs:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        frames.append(EvalFrame(strip_name=_name_key("fcdd", img_path.stem),
                                 img_path=img_path, img_bgr=img_bgr, gt_user=None))
    return frames


def _name_key(prefix: str, stem: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", f"{prefix}_{stem}")
    return safe[:200] if len(safe) > 200 else safe


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def _gt_trav_valid(gt_user: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Binary traversable-only view: sky and ignore are both excluded."""
    return gt_user == 1, np.isin(gt_user, (0, 1))


def _binary_traversable_iou(pred_trav: np.ndarray, gt_trav: np.ndarray, valid: np.ndarray) -> float | None:
    v = valid
    if not v.any():
        return None
    p = pred_trav & v
    g = gt_trav & v
    union = np.logical_or(p, g).sum(dtype=np.float64)
    if union <= 0:
        return None
    return float(np.logical_and(p, g).sum(dtype=np.float64) / union)


def _micro_precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
    denom_p, denom_r = tp + fp, tp + fn
    p = float(tp) / denom_p if denom_p > 0 else None
    r = float(tp) / denom_r if denom_r > 0 else None
    if p is None or r is None:
        f1: float | None = None
    elif p + r == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * p * r / (p + r)
    return p, r, f1


def _write_performance_artifacts(
    out_dir: Path,
    *,
    dataset_key: str,
    models: list[str],
    iou_by: defaultdict[str, list[float]],
    micro_by: defaultdict[str, dict[str, int]],
    n_frames: defaultdict[str, int],
    miou_3class_by: dict[str, tuple[float, list[float]]],
) -> None:
    def row_metrics(model: str) -> dict[str, object]:
        ious = iou_by.get(model, [])
        n_tot = int(n_frames.get(model, 0))
        mic = micro_by.get(model, {"tp": 0, "fp": 0, "fn": 0})
        p, r, f1 = _micro_precision_recall_f1(int(mic["tp"]), int(mic["fp"]), int(mic["fn"]))
        m3, per_class = miou_3class_by.get(model, (float("nan"), [float("nan")] * NUM_CLASSES))
        return {
            "n_frames": n_tot,
            "n_frames_iou_defined": len(ious),
            "mean_iou_traversable_binary": float(np.mean(ious)) if ious else None,
            "median_iou_traversable_binary": float(np.median(ious)) if ious else None,
            "micro_precision": p, "micro_recall": r, "micro_f1": f1,
            "micro_tp": int(mic["tp"]), "micro_fp": int(mic["fp"]), "micro_fn": int(mic["fn"]),
            "mean_iou_3class": None if m3 != m3 else m3,  # NaN check
            "per_class_iou_3class": {n: (None if v != v else v) for n, v in zip(("non_traversable", "traversable", "sky"), per_class)},
        }

    payload = {dataset_key: {m: row_metrics(m) for m in models}}
    jp = out_dir / "performance_summary.json"
    jp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %s", jp)

    def _cell(x: float | None, nd: int = 3) -> str:
        return "—" if x is None else f"{x:.{nd}f}"

    md_lines = [
        f"# {dataset_key} performance summary", "",
        "Binary traversable IoU: freespace-only, sky+ignore excluded from valid pixels. "
        "3-class mIoU: non_traversable/traversable/sky, same metric used by training/eval scripts.",
        "",
        "| model | mean trav IoU | median trav IoU | micro P | micro R | micro F1 | 3-class mIoU |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in models:
        row = row_metrics(m)
        md_lines.append(
            f"| {m} | {_cell(row['mean_iou_traversable_binary'])} | "
            f"{_cell(row['median_iou_traversable_binary'])} | {_cell(row['micro_precision'])} | "
            f"{_cell(row['micro_recall'])} | {_cell(row['micro_f1'])} | {_cell(row['mean_iou_3class'])} |",
        )
    (out_dir / "performance_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    logger.info("wrote %s", out_dir / "performance_summary.md")


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #


def assemble_mosaic_vertical(row_images: list[np.ndarray], *, gap_px: int = 6,
                              pad_color: tuple[int, int, int] = (32, 32, 32)) -> np.ndarray:
    mw = max(r.shape[1] for r in row_images)
    total_h = sum(r.shape[0] for r in row_images) + gap_px * (len(row_images) - 1)
    mosaic = np.full((total_h, mw, 3), pad_color, dtype=np.uint8)
    y = 0
    for r in row_images:
        h, w = r.shape[:2]
        x_off = (mw - w) // 2
        mosaic[y:y + h, x_off:x_off + w] = r
        y += h + gap_px
    return mosaic


def strip_row_band_with_metrics(row_bgr: np.ndarray, metrics_line: str) -> np.ndarray:
    if not metrics_line:
        return row_bgr
    h, w = row_bgr.shape[:2]
    n_splits = metrics_line.count(" | ") + 1
    band_h = max(28, min(72, 16 * n_splits + 10))
    band = np.zeros((band_h, w, 3), dtype=np.uint8)
    y = 16
    for ln in metrics_line.split(" | "):
        cv2.putText(band, ln[:118], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 235, 170), 1, cv2.LINE_AA)
        y += 17
        if y > band_h - 6:
            break
    return np.vstack([row_bgr, band])


def _resize_to_native(merged: torch.Tensor, native_hw: tuple[int, int]) -> torch.Tensor:
    """Not every SemanticModel.predict_logits() returns native resolution
    (SegFormer deliberately halves it) -- upsample defensively rather than
    assume."""
    h, w = native_hw
    if merged.shape[-2:] == (h, w):
        return merged
    return torch.nn.functional.interpolate(merged.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False)[0]


def _road_ground_channel_index(names: tuple[str, ...]) -> int:
    try:
        return names.index("road_ground")
    except ValueError as e:
        raise ValueError("This tool requires a semantic class named `road_ground` in config.yaml.") from e


def _first_available_sample_image() -> np.ndarray:
    """Grab one real frame for latency probing -- prefers a real dataset image
    (goes through each wrapper's real preprocessing) over a synthetic tensor."""
    seg_root = _REPO / "datasets" / "segmentation"
    candidates = [
        seg_root / "ORFD" / "training" / "image_data",
        seg_root / "off_road_zikim" / "val",
        seg_root / "FCDD" / "val" / "im",
    ]
    for c in candidates:
        if not c.is_dir():
            continue
        for p in sorted(c.rglob("*.png")) + sorted(c.rglob("*.jpg")):
            img = cv2.imread(str(p))
            if img is not None:
                return img
    logger.warning("No sample image found under datasets/segmentation/; using a synthetic frame.")
    return np.zeros((720, 1280, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Interactive picking
# --------------------------------------------------------------------------- #


def _pick_dataset_interactively() -> tuple[str, Path]:
    found, skipped = scan_segmentation_datasets(_REPO / "datasets" / "segmentation")
    if not found:
        raise SystemExit(
            "No recognized dataset under datasets/segmentation/. "
            f"Skipped: {skipped}. Pass --dataset/--dataset-root explicitly.",
        )
    for name, reason in skipped:
        logger.info("Skipping %s: %s", name, reason)
    options = [(d.kind, d.label) for d in found]
    kind = ask_choice("Which dataset?", options, default_idx=0)
    chosen = next(d for d in found if d.kind == kind)
    return chosen.kind, chosen.root


def _pick_models_interactively() -> list[str]:
    checkpoints = scan_semantic_checkpoints(_REPO / "weights" / "segmentation" / "orfd")
    all_choices = list(checkpoints) + list(BASELINE_CHOICES)
    if not all_choices:
        raise SystemExit("No checkpoints found under weights/segmentation/ and no baselines available.")
    options = [
        (f"{c.key}:{c.weights}" if c.weights else c.key, f"{c.key}  [{c.label}]")
        for c in all_choices
    ]
    default_idxs = tuple(range(min(3, len(options))))
    return ask_multi_choice("Which model(s)? (compare 2+ side by side)", options, default_idxs)


# --------------------------------------------------------------------------- #
# Latency-only mode (folds in the old benchmark_new_architectures.py)
# --------------------------------------------------------------------------- #


def run_latency_only(model_specs: list[str], cfg, hw, backend, output_dir: Path) -> int:
    sample_bgr = _first_available_sample_image()
    results: dict[str, dict[str, float]] = {}
    for spec in model_specs:
        key, explicit_w = parse_model_spec(spec)
        weights = resolve_weights(key, explicit_w, cfg)
        try:
            mdl = build_semantic_model(SemanticModelCfg(name=key, weights=weights), hw, backend)
            mdl.warmup(cfg.classes)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load %s: %s — skipping.", spec, e)
            continue
        lat = measure_forward_latency_ms(mdl, sample_frame=sample_bgr)
        inner = getattr(mdl, "_model", None)
        params_m = sum(p.numel() for p in inner.parameters()) / 1e6 if inner is not None else float("nan")
        results[spec] = {"params_m": round(params_m, 2), "latency_ms": round(lat, 2), "fps": round(1000.0 / lat, 1)}
        logger.info("%s: %.2f ms (%.1f FPS), %.2f M params", spec, lat, 1000.0 / lat, params_m)
        del mdl
        if hw.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not results:
        logger.error("No model produced a latency measurement.")
        return 2
    out_path = output_dir / "latency_benchmark.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("wrote %s", out_path)
    print(json.dumps(results, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["orfd", "zikim", "fcdd"], default=None,
                    help="Omit to pick interactively from what's on disk.")
    p.add_argument("--dataset-root", default=None,
                    help="Override the dataset directory (default: datasets/segmentation/<name>).")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--models", nargs="+", default=None,
                    help="Model keys, optionally 'key:weights_path'. Omit to pick interactively.")
    p.add_argument("--samples", type=int, default=None,
                    help="Random sample size; 0 = use all frames in natural order "
                         "(default: 20 for orfd/fcdd, 0/all for zikim).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--orfd-split", default="training", choices=["training", "validation", "testing"])
    p.add_argument("--zikim-val-subdir", default="m24")
    p.add_argument("--fcdd-split", default="val", choices=["train", "val", "test"])
    p.add_argument("--output-dir", default=None, help="Default: reports/segmentation/<dataset>_comparison")
    p.add_argument("--panel-w", type=int, default=320)
    p.add_argument("--single-mosaic", action="store_true", help="Stack all strips into one PNG.")
    p.add_argument("--stitch-video", dest="stitch_video", action="store_true", default=None,
                    help="Encode strips into an .mp4 instead of/alongside the mosaic "
                         "(auto-on for zikim's continuous-recording frames).")
    p.add_argument("--no-stitch-video", dest="stitch_video", action="store_false",
                    help="Disable, overriding the zikim auto-on default.")
    p.add_argument("--fps", type=float, default=4.0, help="Output FPS if --stitch-video.")
    p.add_argument("--no-performance-summary", action="store_true")
    p.add_argument("--latency-only", action="store_true",
                    help="Skip the dataset entirely; just measure params+latency for --models.")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    hw = cfg.hardware
    backend = PyTorchBackend()

    model_specs = args.models if args.models else _pick_models_interactively()

    if args.latency_only:
        out_dir = Path(args.output_dir) if args.output_dir else _REPO / "reports" / "segmentation"
        return run_latency_only(model_specs, cfg, hw, backend, out_dir)

    if args.dataset:
        dataset_root = Path(args.dataset_root) if args.dataset_root else _REPO / "datasets" / "segmentation" / (
            "ORFD" if args.dataset == "orfd" else "off_road_zikim" if args.dataset == "zikim" else "FCDD"
        )
        dataset_kind = args.dataset
    else:
        dataset_kind, dataset_root = _pick_dataset_interactively()

    samples = args.samples
    if samples is None:
        samples = 0 if dataset_kind == "zikim" else 20

    if dataset_kind == "orfd":
        frames = _orfd_frames(dataset_root, split=args.orfd_split, samples=samples, seed=args.seed)
    elif dataset_kind == "zikim":
        frames = _zikim_frames(dataset_root, val_subdir=args.zikim_val_subdir, samples=samples, seed=args.seed)
    else:
        frames = _fcdd_frames(dataset_root, split=args.fcdd_split, samples=samples, seed=args.seed)

    if not frames:
        logger.error("No usable frames for dataset=%s root=%s", dataset_kind, dataset_root)
        return 2

    has_gt = frames[0].gt_user is not None
    stitch_video = args.stitch_video if args.stitch_video is not None else (dataset_kind == "zikim")

    sem_classes = list(cfg.semantic_classes)
    if not sem_classes:
        logger.error("config has no semantic classes")
        return 2
    rg_idx = _road_ground_channel_index(tuple(c.name for c in sem_classes)) if has_gt else None

    out_dir = Path(args.output_dir) if args.output_dir else _REPO / "reports" / "segmentation" / f"{dataset_kind}_comparison"
    strips_dir = out_dir / "strips"
    strips_dir.mkdir(parents=True, exist_ok=True)

    pred_store: dict[str, dict[str, np.ndarray]] = {}
    perf_iou_by: defaultdict[str, list[float]] = defaultdict(list)
    perf_micro_by: defaultdict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    perf_n_frames: defaultdict[str, int] = defaultdict(int)
    miou_preds: defaultdict[str, list[torch.Tensor]] = defaultdict(list)
    miou_gts: list[torch.Tensor] = []
    models_ran: list[str] = []

    for spec in model_specs:
        key, explicit_w = parse_model_spec(spec)
        weights = resolve_weights(key, explicit_w, cfg)
        try:
            mdl = build_semantic_model(SemanticModelCfg(name=key, weights=weights), hw, backend)
            mdl.warmup(cfg.classes)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load %s: %s — skipping.", spec, e)
            continue

        ran_any = False
        for fr in frames:
            merged = mdl.predict_logits(fr.img_bgr)
            merged = _resize_to_native(merged, fr.img_bgr.shape[:2])
            mnp = merged.float().cpu().numpy()

            if has_gt:
                pred_mc = mnp.argmax(axis=0).astype(np.int64, copy=False)
                pred_trav = pred_mc == int(rg_idx)

                gt_trav, valid = _gt_trav_valid(fr.gt_user)
                perf_n_frames[spec] += 1
                pb, gb, vb = pred_trav, gt_trav, valid
                mic = perf_micro_by[spec]
                mic["tp"] += int((pb & gb & vb).sum())
                mic["fp"] += int((pb & ~gb & vb).sum())
                mic["fn"] += int((~pb & gb & vb).sum())
                iou_bin = _binary_traversable_iou(pb, gb, vb)
                if iou_bin is not None:
                    perf_iou_by[spec].append(iou_bin)

                # 3-class mIoU accumulation (road_ground channel = "traversable";
                # the other two config channels are assumed to align 1:1 with
                # non_traversable/sky by construction of _TRAV_CLASSES' order —
                # models compared here always expose exactly the 3 ORFD classes).
                miou_preds[spec].append(torch.from_numpy(pred_mc).unsqueeze(0))
                pred_store.setdefault(fr.strip_name, {})[spec] = pred_mc.astype(np.int8, copy=False)
            else:
                pred_mc = mnp.argmax(axis=0).astype(np.int8, copy=False)
                pred_store.setdefault(fr.strip_name, {})[spec] = pred_mc

            ran_any = True
            del merged, mnp

        del mdl
        if hw.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if ran_any:
            models_ran.append(spec)

    if has_gt:
        for fr in frames:
            miou_gts.append(torch.from_numpy(fr.gt_user.astype(np.int64)).unsqueeze(0))
        gts_cat = torch.cat(miou_gts, dim=0)
        miou_3class_by: dict[str, tuple[float, list[float]]] = {}
        for spec in models_ran:
            preds_cat = torch.cat(miou_preds[spec], dim=0)
            m3, per_class = compute_miou(preds_cat, gts_cat, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)
            miou_3class_by[spec] = (m3, per_class)
            logger.info("%s: 3-class mIoU=%.4f  binary trav mean IoU=%s", spec, m3,
                        f"{np.mean(perf_iou_by[spec]):.4f}" if perf_iou_by[spec] else "—")
    else:
        miou_3class_by = {}

    if not models_ran:
        logger.error("No model produced predictions.")
        return 2

    mosaic_rows: list[np.ndarray] = []
    for fr in frames:
        preds_vis = pred_store.get(fr.strip_name, {})
        gt_vis = None
        chunks: list[str] = []
        if has_gt:
            gt_trav, valid = _gt_trav_valid(fr.gt_user)
            gt_vis = np.where(fr.gt_user == IGNORE_INDEX, np.int8(-1), fr.gt_user.astype(np.int8))
            for kname in models_ran:
                pr = preds_vis.get(kname)
                if pr is None:
                    chunks.append(f"{kname} IoU=—")
                    continue
                iou_bb = _binary_traversable_iou(pr == 1, gt_trav, valid)
                chunks.append(f"{kname} trav_IoU={'%.3f' % iou_bb if iou_bb is not None else '—'}")

        title = f"{fr.img_path.parent.name}/{fr.img_path.name}"
        pane = strips_dir / f"_pane_{fr.strip_name}.png"
        render_panel(
            title=title, image_bgr=fr.img_bgr, gt_userclass=gt_vis,
            preds={k: preds_vis[k] for k in models_ran if k in preds_vis},
            user_classes=list(_TRAV_CLASSES), out_path=pane, target_w=args.panel_w,
        )
        strip = cv2.imread(str(pane))
        pane.unlink(missing_ok=True)
        if strip is None:
            continue
        banded = strip_row_band_with_metrics(strip, " | ".join(chunks))
        out_png = strips_dir / f"{fr.strip_name}.png"
        cv2.imwrite(str(out_png), banded)
        logger.info("wrote %s", out_png)
        mosaic_rows.append(banded)

    if not mosaic_rows:
        logger.error("No PNG strips generated.")
        return 2

    readme_lines = [
        f"Generated by scripts/segmentation/evaluation/compare_semantic_models.py --dataset {dataset_kind}",
        f"dataset_root={dataset_root}",
        f"models={models_ran}",
        f"has_gt={has_gt}",
    ]
    if dataset_kind == "fcdd":
        readme_lines.append("FCDD has its own 17-class GT but no verified collapse mapping yet -- qualitative only.")
    (out_dir / "README.txt").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    if has_gt and not args.no_performance_summary:
        _write_performance_artifacts(
            out_dir, dataset_key=dataset_kind, models=models_ran,
            iou_by=perf_iou_by, micro_by=perf_micro_by, n_frames=perf_n_frames,
            miou_3class_by=miou_3class_by,
        )

    if args.single_mosaic:
        mosaic = assemble_mosaic_vertical(mosaic_rows)
        mop = out_dir / f"{dataset_kind}_mosaic.png"
        cv2.imwrite(str(mop), mosaic)
        logger.info("wrote %s", mop)

    if stitch_video:
        first = mosaic_rows[0]
        h, w = first.shape[:2]
        video_path = out_dir / f"{dataset_kind}_video.mp4"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        for row in mosaic_rows:
            if row.shape[:2] != (h, w):
                row = cv2.resize(row, (w, h))
            writer.write(row)
        writer.release()
        logger.info("wrote %s (%d frames)", video_path, len(mosaic_rows))

    if hw.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

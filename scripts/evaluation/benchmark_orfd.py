"""Benchmark semantic segmentation models on the ORFD validation set.

Evaluates each registered model, computes comprehensive metrics, and
produces a side-by-side visual comparison of predictions.

Metrics reported
----------------
  iou_traversable      IoU for class 1 (traversable)
  iou_non_traversable  IoU for class 0
  mean_iou             Mean of the two class IoUs (mIoU)
  median_iou           Median of per-image mIoU scores
  f1_traversable       F1 / Dice for the traversable class
  precision_traversable
  recall_traversable
  latency_ms           Median inference latency over 100 frames (ms)
  fps                  1000 / latency_ms
  params_M             Model parameter count (millions)
  model_size_mb        Checkpoint file size on disk (MB), or 0 for HF models

Output files
------------
  reports/orfd_finetuned/performance_summary.json
  reports/orfd_finetuned/performance_summary.md
  reports/orfd_finetuned/qualitative/<idx>_comparison.png   (per sample)
  reports/orfd_finetuned/qualitative/mosaic.png             (grid overview)

Usage
-----
    # Evaluate all registered models (runs those whose checkpoint exists):
    python scripts/benchmark_orfd.py

    # Evaluate a specific subset:
    python scripts/benchmark_orfd.py --models segformer-b2-baseline segformer-b2-orfd

    # Use a different split:
    python scripts/benchmark_orfd.py --split testing
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from perception.datasets.orfd_torch import ORFDDataset, TRAIN_SIZE, _to_normalized_tensor
from perception.datasets.orfd_labels import binary_traversable_iou, orfd_eval_valid_mask

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark_orfd")

NUM_CLASSES   = 2
IGNORE_INDEX  = 255

# Class colours for overlay visualisation (BGR).
_CLASS_COLORS = {
    0: (60, 60, 60),     # non-traversable: dark grey
    1: (100, 100, 255),  # traversable: blue-ish (matches pipeline road_ground)
}

# Label for overlay panels.
_CLASS_NAMES = {0: "non-trav", 1: "traversable"}


# --------------------------------------------------------------------------- #
# Model registry                                                               #
# --------------------------------------------------------------------------- #


def _default_model_defs() -> list[dict[str, Any]]:
    """Return the list of models to benchmark (those whose checkpoint exists)."""
    defs = [
        {
            "key": "segformer-b2-baseline",
            "label": "SegFormer-B2\n(baseline)",
            "type": "segformer-ade20k",
            "hf_id": "nvidia/segformer-b2-finetuned-ade-512-512",
            "checkpoint": None,
        },
        {
            "key": "segformer-b2-final",
            "label": "SegFormer-B2\n(Final_Dataset)",
            "type": "segformer-finetuned",
            "hf_id": "nvidia/segformer-b2-finetuned-ade-512-512",
            "checkpoint": str(_ROOT / "weights" / "orfd" / "final_dataset" / "segformer-b2" / "best.pth"),
        },
        {
            "key": "segformer-b2-frozen",
            "label": "SegFormer-B2\n(frozen backbone)",
            "type": "segformer-finetuned",
            "hf_id": "nvidia/segformer-b2-finetuned-ade-512-512",
            "checkpoint": str(_ROOT / "weights" / "orfd" / "frozen_backbone" / "segformer-b2" / "best.pth"),
        },
        {
            "key": "segformer-b0-frozen",
            "label": "SegFormer-B0\n(frozen backbone)",
            "type": "segformer-finetuned",
            "hf_id": "nvidia/segformer-b0-finetuned-ade-512-512",
            "checkpoint": str(_ROOT / "weights" / "orfd" / "frozen_backbone" / "segformer-b0" / "best.pth"),
        },
        {
            "key": "segformer-b1-frozen",
            "label": "SegFormer-B1\n(frozen backbone)",
            "type": "segformer-finetuned",
            "hf_id": "nvidia/segformer-b1-finetuned-ade-512-512",
            "checkpoint": str(_ROOT / "weights" / "orfd" / "frozen_backbone" / "segformer-b1" / "best.pth"),
        },
        {
            "key": "segformer-b2-lora",
            "label": "SegFormer-B2\n(LoRA)",
            "type": "segformer-finetuned",
            "hf_id": "nvidia/segformer-b2-finetuned-ade-512-512",
            "checkpoint": str(_ROOT / "weights" / "orfd" / "lora" / "segformer-b2" / "best.pth"),
        },
        {
            "key": "segformer-b2-frozen-lora",
            "label": "SegFormer-B2\n(frozen backbone + LoRA)",
            "type": "segformer-finetuned",
            "hf_id": "nvidia/segformer-b2-finetuned-ade-512-512",
            "checkpoint": str(_ROOT / "weights" / "orfd" / "frozen_lora" / "segformer-b2" / "best.pth"),
        },
        {
            "key": "segformer-b4-final",
            "label": "SegFormer-B4\n(Final_Dataset)",
            "type": "segformer-finetuned",
            "hf_id": "nvidia/segformer-b4-finetuned-ade-512-512",
            "checkpoint": str(_ROOT / "weights" / "orfd" / "final_dataset" / "segformer-b4" / "best.pth"),
        },
        {
            "key": "auriganet-final",
            "label": "AurigaNet\n(ORFD fine-tuned)",
            "type": "auriganet-finetuned",
            "checkpoint": str(_ROOT / "weights" / "orfd" / "auriganet" / "best.pth"),
        },
    ]
    return defs


# --------------------------------------------------------------------------- #
# Model loaders                                                                #
# --------------------------------------------------------------------------- #


def load_segformer_baseline(hf_id: str, device: str, fp16: bool):
    """Load standard ADE20K SegFormer; returns (model, processor, predict_fn)."""
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    # ADE20K indices for traversable classes (path=52, dirt-track=91).
    TRAVERSABLE_INDICES = [52, 91]
    N_ADE = 150
    lut_traversable = torch.zeros(N_ADE)
    for i in TRAVERSABLE_INDICES:
        lut_traversable[i] = 1.0

    processor = SegformerImageProcessor.from_pretrained(hf_id)
    model = SegformerForSemanticSegmentation.from_pretrained(hf_id)
    model.eval().to(device)
    if fp16:
        model = model.half()
    lut_traversable = lut_traversable.to(device)

    @torch.no_grad()
    def predict(frame_bgr: np.ndarray) -> np.ndarray:
        """Return (H, W) uint8 prediction: 0=non-trav, 1=trav."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_bgr.shape[:2]
        inputs = processor(images=rgb, return_tensors="pt")
        pv = inputs["pixel_values"].to(device)
        if fp16:
            pv = pv.half()
        out = model(pixel_values=pv).logits  # (1, 150, H/4, W/4)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)[0]
        probs = torch.softmax(out.float(), dim=0)           # (150, H, W)
        trav_prob = (probs * lut_traversable[:, None, None]).sum(0)  # (H, W)
        # Traversable if it wins the argmax proxy: trav_prob > non-trav residual.
        pred = (trav_prob > 0.5).byte().cpu().numpy()
        return pred

    params = sum(p.numel() for p in model.parameters()) / 1e6
    return predict, params, 0.0


def load_segformer_finetuned(hf_id: str, checkpoint: str, device: str, fp16: bool):
    """Load fine-tuned SegFormer from a training checkpoint (2- or 3-class)."""
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt

    # Auto-detect number of output classes from the checkpoint so this works
    # for both the old 2-class checkpoints and the new 3-class (sky) checkpoints.
    n_classes = state_dict["decode_head.classifier.weight"].shape[0]

    processor = SegformerImageProcessor.from_pretrained(hf_id)
    model = SegformerForSemanticSegmentation.from_pretrained(
        hf_id, num_labels=n_classes, ignore_mismatched_sizes=True,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    if fp16:
        model = model.half()

    @torch.no_grad()
    def predict(frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_bgr.shape[:2]
        inputs = processor(images=rgb, return_tensors="pt")
        pv = inputs["pixel_values"].to(device)
        if fp16:
            pv = pv.half()
        out = model(pixel_values=pv).logits  # (1, n_classes, H/4, W/4)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)[0]
        # Class 1 = traversable in both 2-class and 3-class models;
        # sky (class 2) maps to 0 (non-traversable) automatically.
        return (out.argmax(0) == 1).byte().cpu().numpy()

    params = sum(p.numel() for p in model.parameters()) / 1e6
    size_mb = Path(checkpoint).stat().st_size / 1e6
    return predict, params, size_mb


def load_auriganet_finetuned(checkpoint: str, device: str, fp16: bool):
    """Load fine-tuned AurigaNet from a training checkpoint."""
    from perception.models.semantic._vendored.auriganet import AurigaNetArch

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("net", ckpt) if isinstance(ckpt, dict) else ckpt
    model = AurigaNetArch(num_seg_classes=3, with_detection=False)
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    if fp16:
        model = model.half()

    @torch.no_grad()
    def predict(frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
        x = resized.astype(np.float32) / 255.0
        x = (x - _MEAN) / _STD
        x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)
        if fp16:
            x = x.half()
        seg_logits, _, _ = model(x)  # (1, 3, 160, 160)
        seg_logits = F.interpolate(
            seg_logits.float(), size=(h, w), mode="bilinear", align_corners=False
        )[0]
        return (seg_logits.argmax(0) == 1).byte().cpu().numpy()

    params = sum(p.numel() for p in model.parameters()) / 1e6
    size_mb = Path(checkpoint).stat().st_size / 1e6
    return predict, params, size_mb


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #


def compute_metrics(
    predict_fn,
    dataset: ORFDDataset,
    device: str,
    n_latency: int = 100,
) -> dict[str, Any]:
    """Run predict_fn over the dataset and return the full metrics dict."""

    # --- Accuracy metrics ---
    iou_trav_list:      list[float] = []
    iou_nontrav_list:   list[float] = []
    per_image_miou:     list[float] = []
    tp = tn = fp = fn = 0

    pairs = dataset.pairs
    for img_path, gt_path in pairs:
        bgr = cv2.imread(str(img_path))
        gt_u8 = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if bgr is None or gt_u8 is None:
            continue

        pred = predict_fn(bgr)  # (H, W) uint8 {0, 1}
        # Resize GT to match prediction if needed.
        if pred.shape != gt_u8.shape:
            gt_u8 = cv2.resize(gt_u8, (pred.shape[1], pred.shape[0]),
                                interpolation=cv2.INTER_NEAREST)

        valid = orfd_eval_valid_mask(gt_u8)
        gt_trav = (gt_u8 == 255)
        pred_trav = pred.astype(bool)

        iou_t = binary_traversable_iou(pred_trav, gt_trav, valid)
        # IoU for non-traversable = swap the roles.
        iou_nt = binary_traversable_iou(~pred_trav, ~gt_trav, valid)

        if iou_t is not None:
            iou_trav_list.append(iou_t)
        if iou_nt is not None:
            iou_nontrav_list.append(iou_nt)
        if iou_t is not None and iou_nt is not None:
            per_image_miou.append((iou_t + iou_nt) / 2.0)

        # Precision / recall components.
        v = valid.ravel()
        pt = pred_trav.ravel() & v
        gt = gt_trav.ravel() & v
        tp += int((pt & gt).sum())
        fp += int((pt & ~gt).sum())
        fn += int((~pt & gt).sum())
        tn += int((~pt & ~gt).sum())

    iou_trav  = float(np.mean(iou_trav_list))   if iou_trav_list   else float("nan")
    iou_nontrav = float(np.mean(iou_nontrav_list)) if iou_nontrav_list else float("nan")
    mean_iou  = (iou_trav + iou_nontrav) / 2.0  if iou_trav_list and iou_nontrav_list else float("nan")
    median_iou = float(np.median(per_image_miou)) if per_image_miou else float("nan")

    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")

    # --- Latency ---
    logger.info("  Measuring latency over %d frames...", n_latency)
    sample_pairs = pairs[:n_latency] if len(pairs) >= n_latency else pairs * (n_latency // len(pairs) + 1)
    sample_pairs = sample_pairs[:n_latency]
    bgrs_for_lat = []
    for ip, _ in sample_pairs:
        bgr = cv2.imread(str(ip))
        if bgr is not None:
            bgrs_for_lat.append(bgr)

    # Warm-up.
    for bgr in bgrs_for_lat[:5]:
        _ = predict_fn(bgr)

    latencies = []
    for bgr in bgrs_for_lat:
        t0 = time.perf_counter()
        _ = predict_fn(bgr)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latency_ms = float(np.median(latencies))
    fps = 1000.0 / latency_ms if latency_ms > 0 else float("nan")

    return dict(
        iou_traversable=round(iou_trav,    4),
        iou_non_traversable=round(iou_nontrav, 4),
        mean_iou=round(mean_iou,           4),
        median_iou=round(median_iou,       4),
        f1_traversable=round(f1,           4),
        precision_traversable=round(prec,  4),
        recall_traversable=round(rec,      4),
        latency_ms=round(latency_ms,       2),
        fps=round(fps,                     1),
    )


# --------------------------------------------------------------------------- #
# Visualisation                                                                #
# --------------------------------------------------------------------------- #


def _colorise_pred(pred: np.ndarray, h: int, w: int) -> np.ndarray:
    """Convert (H, W) uint8 prediction to (H, W, 3) BGR overlay."""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in _CLASS_COLORS.items():
        canvas[pred == cls_id] = color
    return canvas


def _colorise_gt(gt_u8: np.ndarray, h: int, w: int) -> np.ndarray:
    """Convert ORFD fillcolor GT to colour overlay (BGR)."""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[gt_u8 == 255] = _CLASS_COLORS[1]   # traversable
    canvas[gt_u8 == 0]   = _CLASS_COLORS[0]   # non-traversable
    canvas[gt_u8 == 128] = (180, 180, 180)     # sky band: grey
    return canvas


def _add_label(img: np.ndarray, text: str, iou: float | None = None) -> np.ndarray:
    """Add a text banner at the top of a panel image."""
    out = img.copy()
    lines = text.split("\n")
    if iou is not None:
        lines.append(f"mIoU={iou:.3f}")
    y = 18
    for line in lines:
        cv2.putText(out, line, (4, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
        y += 16
    return out


def make_comparison_panel(
    bgr: np.ndarray,
    gt_u8: np.ndarray,
    predictions: list[tuple[np.ndarray, str, float | None]],  # (pred, label, miou)
    panel_w: int = 256,
) -> np.ndarray:
    """Build a single wide comparison image.

    Columns: [Input | GT | model_1 | model_2 | ...]
    """
    aspect = bgr.shape[0] / bgr.shape[1]
    ph = int(panel_w * aspect)

    def resize(img):
        return cv2.resize(img, (panel_w, ph), interpolation=cv2.INTER_LINEAR)

    # Input panel.
    inp_panel = resize(bgr)
    inp_panel = _add_label(inp_panel, "Input")

    # GT panel.
    gt_colour = resize(_colorise_gt(gt_u8, gt_u8.shape[0], gt_u8.shape[1]))
    gt_colour  = _add_label(gt_colour, "GT")

    panels = [inp_panel, gt_colour]
    for pred, label, miou in predictions:
        pred_colour = resize(_colorise_pred(
            pred, pred.shape[0], pred.shape[1],
        ))
        panels.append(_add_label(pred_colour, label, miou))

    return np.concatenate(panels, axis=1)


def build_qualitative(
    pairs_sample: list[tuple[Path, Path]],
    predict_fns: list[tuple[str, str, object]],  # (key, label, predict_fn)
    out_dir: Path,
    panel_w: int = 256,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_panels: list[np.ndarray] = []

    for i, (img_path, gt_path) in enumerate(pairs_sample):
        bgr   = cv2.imread(str(img_path))
        gt_u8 = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if bgr is None or gt_u8 is None:
            continue

        predictions: list[tuple[np.ndarray, str, float | None]] = []
        for _, label, predict_fn in predict_fns:
            pred = predict_fn(bgr)
            # Per-image mIoU for this model.
            valid  = orfd_eval_valid_mask(gt_u8)
            iou_t  = binary_traversable_iou(pred.astype(bool), gt_u8 == 255, valid)
            iou_nt = binary_traversable_iou(~pred.astype(bool), gt_u8 == 0,  valid)
            img_miou = (iou_t + iou_nt) / 2.0 if iou_t is not None and iou_nt is not None else None
            predictions.append((pred, label, img_miou))

        panel = make_comparison_panel(bgr, gt_u8, predictions, panel_w=panel_w)
        save_path = out_dir / f"{i:03d}_comparison.png"
        cv2.imwrite(str(save_path), panel)
        saved_panels.append(panel)
        logger.info("  Saved qualitative panel %d → %s", i, save_path.name)

    # Mosaic: stack all panels vertically.
    if saved_panels:
        mosaic = np.concatenate(saved_panels, axis=0)
        cv2.imwrite(str(out_dir / "mosaic.png"), mosaic)
        logger.info("Mosaic saved → %s", out_dir / "mosaic.png")


# --------------------------------------------------------------------------- #
# Markdown / JSON output                                                       #
# --------------------------------------------------------------------------- #


_MD_COLS = [
    ("Model",               "label"),
    ("mIoU",                "mean_iou"),
    ("Median IoU",          "median_iou"),
    ("IoU Traversable",     "iou_traversable"),
    ("IoU Non-Trav",        "iou_non_traversable"),
    ("F1 Traversable",      "f1_traversable"),
    ("Precision",           "precision_traversable"),
    ("Recall",              "recall_traversable"),
    ("Latency (ms)",        "latency_ms"),
    ("FPS",                 "fps"),
    ("Params (M)",          "params_M"),
    ("Size (MB)",           "model_size_mb"),
]


def write_reports(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = out_dir / "performance_summary.json"
    json_path.write_text(json.dumps(results, indent=2))
    logger.info("JSON report → %s", json_path)

    # Markdown table
    header = "| " + " | ".join(c for c, _ in _MD_COLS) + " |"
    sep    = "| " + " | ".join("---" for _ in _MD_COLS) + " |"
    rows   = []
    for r in results:
        row_vals = []
        for col_name, key in _MD_COLS:
            v = r.get(key, "—")
            if isinstance(v, float) and v != v:  # nan
                row_vals.append("—")
            elif isinstance(v, float):
                row_vals.append(f"{v:.4f}" if key not in ("latency_ms", "fps") else f"{v:.1f}")
            else:
                row_vals.append(str(v))
        rows.append("| " + " | ".join(row_vals) + " |")

    md = "\n".join([
        "# ORFD Benchmark: Fine-tuned Models\n",
        header, sep, *rows, "",
        "_Lower is better for latency; higher is better for all accuracy metrics._",
    ])
    md_path = out_dir / "performance_summary.md"
    md_path.write_text(md)
    logger.info("Markdown report → %s", md_path)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark models on ORFD")
    p.add_argument("--models", nargs="*", default=None,
                   help="Model keys to evaluate (default: all available)")
    p.add_argument("--data",   default=str(_ROOT / "datasets" / "Final_Dataset"),
                   help="Path to ORFD root")
    p.add_argument("--split",  default="validation",
                   choices=["validation", "testing"])
    p.add_argument("--qualitative-n", type=int, default=20,
                   help="Number of random samples for qualitative panels")
    p.add_argument("--panel-w", type=int, default=256,
                   help="Width (px) of each panel column in visual comparison")
    p.add_argument("--out",    default=str(_ROOT / "reports" / "orfd_finetuned"),
                   help="Output directory for reports")
    p.add_argument("--no-fp16", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16   = not args.no_fp16 and device == "cuda"
    out_dir = Path(args.out)

    model_defs = _default_model_defs()
    if args.models:
        model_defs = [m for m in model_defs if m["key"] in args.models]

    # Filter to models whose checkpoint exists.
    available: list[dict] = []
    for md in model_defs:
        ckpt = md.get("checkpoint")
        if ckpt and not Path(ckpt).exists():
            logger.warning("Skipping %s: checkpoint not found at %s", md["key"], ckpt)
            continue
        available.append(md)
    if not available:
        logger.error("No models available. Train first with scripts/train_orfd.py.")
        sys.exit(1)

    # Load dataset.
    dataset = ORFDDataset(args.data, split=args.split, augment=False)
    logger.info("Dataset: ORFD %s — %d samples.", args.split, len(dataset))

    # Build predict_fns for each model.
    predict_fns: list[tuple[str, str, object]] = []
    params_map:   dict[str, float] = {}
    size_map:     dict[str, float] = {}

    for md in available:
        key, label, mtype = md["key"], md["label"], md["type"]
        logger.info("Loading %s (%s)...", key, mtype)
        ckpt = md.get("checkpoint")
        hf_id = md.get("hf_id", "")

        if mtype == "segformer-ade20k":
            pfn, params, size = load_segformer_baseline(hf_id, device, fp16)
        elif mtype == "segformer-finetuned":
            pfn, params, size = load_segformer_finetuned(hf_id, ckpt, device, fp16)
        elif mtype == "auriganet-finetuned":
            pfn, params, size = load_auriganet_finetuned(ckpt, device, fp16)
        else:
            logger.warning("Unknown model type %s, skipping.", mtype)
            continue

        predict_fns.append((key, label, pfn))
        params_map[key] = params
        size_map[key]   = size

    # --- Run accuracy + latency metrics ---
    results: list[dict] = []
    for key, label, pfn in predict_fns:
        logger.info("Evaluating %s ...", key)
        metrics = compute_metrics(pfn, dataset, device)
        metrics["key"]   = key
        metrics["label"] = label.replace("\n", " ")
        metrics["params_M"]      = round(params_map[key], 2)
        metrics["model_size_mb"] = round(size_map[key],   1)
        results.append(metrics)

        logger.info(
            "  mIoU=%.4f  median=%.4f  F1=%.4f  latency=%.1fms  fps=%.1f",
            metrics["mean_iou"], metrics["median_iou"],
            metrics["f1_traversable"], metrics["latency_ms"], metrics["fps"],
        )

    write_reports(results, out_dir)

    # --- Qualitative comparison ---
    n = min(args.qualitative_n, len(dataset.pairs))
    sample_indices = sorted(random.sample(range(len(dataset.pairs)), n))
    sample_pairs = [dataset.pairs[i] for i in sample_indices]

    logger.info("Generating %d qualitative comparison panels...", n)
    build_qualitative(
        sample_pairs, predict_fns,
        out_dir=out_dir / "qualitative",
        panel_w=args.panel_w,
    )
    logger.info("Done. Reports in %s", out_dir)


if __name__ == "__main__":
    main()

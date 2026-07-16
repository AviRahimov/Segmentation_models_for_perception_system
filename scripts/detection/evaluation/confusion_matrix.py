#!/usr/bin/env python3
"""Confusion matrix — interactive survey over trained detection checkpoints.

Same interactive UX as the training scripts (numbered menu, '1,3' picks
several) — nothing hardcoded to any specific checkpoint list.

Deliberately does NOT re-derive anything leaderboard.py already produces:
per-class P/R at a fixed operating point (P40/R40/fp_img columns), best-F1
per-class thresholds (--thresholds), and an annotated false-positive gallery
(--fp-gallery) all already exist there. The one thing genuinely missing
elsewhere — which class gets confused as which other class (e.g. tank
predicted as truck) — is what this script adds: a class-agnostic greedy
box matcher (leaderboard.py's _ap_utils.py only has a per-class-filtered
matcher, which can't produce this) accumulated into an (N+1)x(N+1) matrix
(classes + a "background" row/col for false positives/negatives).

Each checkpoint is evaluated on its OWN native class scheme (not collapsed
to leaderboard's 2-class benchmark) and, where possible, its OWN validation
split (inferred from its weights/detection/{model}/{dataset_slug}/{recipe}/
path — falls back to the shared Detection_Dataset benchmark with a warning
when that inference fails, e.g. for old round1/exp sweep checkpoints that
don't carry a resolvable dataset-slug component).

Usage
-----
    python scripts/detection/evaluation/confusion_matrix.py
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / "scripts" / "detection" / "training"))

from _ap_utils import (  # noqa: E402
    RFDETR_2CLASS_NAMES,
    RFDETR_6CLASS_NAMES,
    GT,
    Pred,
    collect_predictions,
    collect_predictions_rfdetr,
    infer_rfdetr_profile,
    iou_xyxy,
    is_rfdetr_checkpoint,
    load_rfdetr_for_eval,
    load_yolo_gts,
)
from leaderboard import _discover_checkpoints, _label  # noqa: E402
from _survey_common import _ask, _confirm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("confusion_matrix")

_BACKGROUND = "background"


# =========================================================================== #
# Per-checkpoint dataset / class-scheme / threshold resolution                #
# =========================================================================== #

def _resolve_own_dataset(ckpt: Path) -> tuple[Path, Path] | None:
    """weights/detection/{model}/{dataset_slug}/{recipe}/best.pt -> matching
    datasets/{Name}/{valid|val}/{images,labels}, or None if unresolvable
    (e.g. round1/exp sweep checkpoints with no real dataset-slug component)."""
    det_root = _ROOT / "weights" / "detection"
    try:
        rel_parts = ckpt.relative_to(det_root).parts
    except ValueError:
        return None
    if len(rel_parts) < 3:
        return None
    dataset_slug = rel_parts[1].lower()
    datasets_root = _ROOT / "datasets"
    if not datasets_root.is_dir():
        return None
    match = next((d for d in datasets_root.iterdir()
                 if d.is_dir() and d.name.lower() == dataset_slug), None)
    if match is None:
        return None
    for split in ("valid", "val"):
        img_dir, lbl_dir = match / split / "images", match / split / "labels"
        if img_dir.is_dir() and lbl_dir.is_dir():
            return img_dir, lbl_dir
    return None


def _dataset_class_names(dataset_dir: Path) -> list[str] | None:
    import yaml
    for y in sorted(dataset_dir.glob("*.yaml")) + sorted(dataset_dir.glob("*.yml")):
        if y.name == "data.local.yaml":
            continue
        try:
            raw = yaml.safe_load(y.read_text()) or {}
        except Exception:
            continue
        names_raw = raw.get("names")
        if names_raw is None:
            continue
        if isinstance(names_raw, dict):
            return [str(names_raw[k]) for k in sorted(names_raw)]
        return [str(n) for n in names_raw]
    return None


def _profile_thresholds_by_index(profile_name: str, coco_offset: int) -> dict[int, float]:
    """config.yaml instance_profiles.{profile_name} -> {raw 0-indexed class id: threshold}.

    coco_offset: 1 for YOLO-family profiles (1-indexed coco_classes), 0 for
    the rfdetr_* profiles (0-indexed) — see rfdetr_2class's comment in
    config.yaml for why RF-DETR fine-tuned checkpoints differ here.
    """
    import yaml
    raw = yaml.safe_load((_ROOT / "config" / "config.yaml").read_text())
    entries = raw.get("instance_profiles", {}).get(profile_name, []) or []
    out: dict[int, float] = {}
    for e in entries:
        for c in e.get("coco_classes", []) or []:
            out[int(c) - coco_offset] = float(e.get("confidence_threshold", 0.25))
    return out


def _resolve_thresholds(ckpt: Path, classes: list[str], is_rf: bool) -> dict[str, float]:
    """Each model's own per-class thresholds from its matching instance_profiles
    entry in config.yaml (what's actually deployed) — not an arbitrary fixed conf."""
    if is_rf:
        profile = "rfdetr_6class" if "6class" in str(ckpt) else "rfdetr_2class"
        offset = 0
    elif set(classes) == set(RFDETR_6CLASS_NAMES):
        profile, offset = "6class", 1
    elif set(classes) == set(RFDETR_2CLASS_NAMES):
        profile, offset = "2class", 1
    else:
        profile = None
    if profile is None:
        logger.warning("  no matching instance_profiles entry for classes %s — using flat 0.25",
                       classes)
        return {c: 0.25 for c in classes}
    idx_thr = _profile_thresholds_by_index(profile, offset)
    return {classes[i]: t for i, t in idx_thr.items() if 0 <= i < len(classes)}


# =========================================================================== #
# Class-agnostic matching + accumulation                                      #
# =========================================================================== #

def class_agnostic_confusion(
    preds: list[Pred],
    gts: list[GT],
    classes: list[str],
    thresholds: dict[str, float],
    iou_thr: float = 0.5,
) -> np.ndarray:
    """(N+1)x(N+1) matrix, rows=true classes, cols=predicted classes, last
    row/col = background (unmatched GT = false negative, unmatched
    prediction = false positive). Unlike _ap_utils.py's _match_class (which
    filters to one class before matching), this matches each prediction
    against the best-IoU unmatched GT box of ANY class, so a wrong-class
    match shows up off-diagonal instead of as an unlinked FP+FN pair."""
    n = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    bg = n
    cm = np.zeros((n + 1, n + 1), dtype=int)

    preds_by_img: dict[str, list[Pred]] = defaultdict(list)
    for p in preds:
        if p.class_name not in idx:
            continue
        if p.score < thresholds.get(p.class_name, 0.25):
            continue
        preds_by_img[p.image_id].append(p)
    gts_by_img: dict[str, list[GT]] = defaultdict(list)
    for g in gts:
        gts_by_img[g.image_id].append(g)

    for img_id in set(preds_by_img) | set(gts_by_img):
        img_preds = sorted(preds_by_img.get(img_id, []), key=lambda p: -p.score)
        img_gts = gts_by_img.get(img_id, [])
        matched_gt: set[int] = set()
        for p in img_preds:
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(img_gts):
                if j in matched_gt:
                    continue
                v = iou_xyxy(p.box, g.box)
                if v > best_iou:
                    best_iou, best_j = v, j
            pred_idx = idx[p.class_name]
            if best_j >= 0 and best_iou >= iou_thr:
                matched_gt.add(best_j)
                true_idx = idx.get(img_gts[best_j].class_name, bg)
                cm[true_idx, pred_idx] += 1
            else:
                cm[bg, pred_idx] += 1  # false positive
        for j, g in enumerate(img_gts):
            if j not in matched_gt:
                true_idx = idx.get(g.class_name, bg)
                cm[true_idx, bg] += 1  # false negative
    return cm


def _plot_heatmap(cm: np.ndarray, labels: list[str], title: str,
                  out_path: Path, normalize: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = cm.astype(float)
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        data = np.divide(data, row_sums, out=np.zeros_like(data), where=row_sums != 0)

    size = max(5.0, 1.1 * len(labels))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=(1.0 if normalize else None))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.suptitle(title, fontsize=11)
    thresh = data.max() / 2 if data.max() else 0.0
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = f"{data[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            color = "white" if data[i, j] > thresh else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =========================================================================== #
# Per-checkpoint run                                                          #
# =========================================================================== #

def _run_one(ckpt: Path, label: str, out_root: Path) -> None:
    is_rf = is_rfdetr_checkpoint(ckpt)

    resolved = _resolve_own_dataset(ckpt)
    if resolved is None:
        logger.warning("  could not infer this checkpoint's own validation set from its "
                       "path — falling back to the shared benchmark "
                       "(datasets/Detection_Dataset/valid).")
        img_dir = _ROOT / "datasets" / "Detection_Dataset" / "valid" / "images"
        lbl_dir = _ROOT / "datasets" / "Detection_Dataset" / "valid" / "labels"
    else:
        img_dir, lbl_dir = resolved

    model = None
    if is_rf:
        classes = [c.name for c in infer_rfdetr_profile(ckpt)]
    else:
        from ultralytics import YOLO
        model = YOLO(str(ckpt))
        names = _dataset_class_names(img_dir.parent) or list((model.names or {}).values())
        classes = list((model.names or {}).values()) or names

    pairs = load_yolo_gts(img_dir, lbl_dir, classes)
    logger.info("  dataset: %s (%d images), classes: %s",
               img_dir.parent.parent.name, len(pairs), classes)

    thresholds = _resolve_thresholds(ckpt, classes, is_rf)
    conf_floor = min(thresholds.values()) if thresholds else 0.05
    identity_collapse = {c: c for c in classes}

    if is_rf:
        model = load_rfdetr_for_eval(ckpt, confidence_floor=conf_floor)
        preds, gts = collect_predictions_rfdetr(model, pairs, identity_collapse)
    else:
        preds, gts = collect_predictions(model, pairs, identity_collapse,
                                         imgsz=1280, conf=conf_floor, device="0")
    del model

    cm = class_agnostic_confusion(preds, gts, classes, thresholds)

    out_dir = out_root / label.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = classes + [_BACKGROUND]
    (out_dir / "counts.json").write_text(json.dumps({
        "label": label, "classes": labels, "matrix": cm.tolist(),
        "thresholds": thresholds,
    }, indent=2))
    _plot_heatmap(cm, labels, f"{label} — raw counts",
                 out_dir / "raw_counts.png", normalize=False)
    _plot_heatmap(cm, labels, f"{label} — row-normalized (recall)",
                 out_dir / "normalized.png", normalize=True)
    logger.info("  -> %s", out_dir.relative_to(_ROOT))


# =========================================================================== #
# Interactive mode                                                            #
# =========================================================================== #

def run_survey() -> None:
    print("=" * 70)
    print("Confusion matrix — interactive setup (Enter = recommended default)")
    print("=" * 70)

    ckpts = _discover_checkpoints()
    if not ckpts:
        logger.error("No checkpoints found under weights/detection/.")
        sys.exit(1)

    options = [(_label(c), "") for c in ckpts]
    picks = _ask(
        "1) Which checkpoint(s) to generate a confusion matrix for? "
        "(a queue like '1,3' runs several)",
        options, default_idx=0, multi=True,
    )
    selected = [ckpts[i] for i in picks]

    print(f"\nSelected {len(selected)} checkpoint(s):")
    for c in selected:
        print(f"  - {_label(c)}")
    if not _confirm("Proceed?"):
        print("Aborted — nothing generated.")
        return

    out_root = _ROOT / "reports" / "detection" / "confusion_matrices"
    for ckpt in selected:
        label = _label(ckpt)
        logger.info("")
        logger.info("=" * 70)
        logger.info("Confusion matrix: %s", label)
        logger.info("=" * 70)
        try:
            _run_one(ckpt, label, out_root)
        except Exception as exc:  # noqa: BLE001 — one failed checkpoint must not kill the queue
            logger.error("  failed: %s", exc)
            continue

    logger.info("")
    logger.info("Done — see %s", out_root.relative_to(_ROOT))


if __name__ == "__main__":
    run_survey()

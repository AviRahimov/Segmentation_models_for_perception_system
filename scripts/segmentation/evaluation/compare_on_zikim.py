#!/usr/bin/env python3
"""Side-by-side comparison on the off_road_zikim dataset's val split.

off_road_zikim uses a Cityscapes-style 5-class labeling scheme (see its
config_zikim.json) that does NOT match ORFD's 3-class scheme
(non_traversable/traversable/sky) the project's models are trained on.
This script converts zikim's color masks into the ORFD-equivalent scheme:

    road (128,64,128, "purple")  -> traversable
    sky  (70,130,180)            -> sky
    ground (153,102,0)           -> non_traversable
    unlabeled / other / anything else -> ignored (both are categorie="void"
    per config_zikim.json — not real ground-truth signal)

Runs N models (factory-driven, same convention as orfd_semantic_comparison.py)
over a val split's sequential frames, renders (input | GT | model predictions)
strips via compare_semantic_models.render_panel, and assembles them into a
real annotated .mp4 (these are frame sequences from a continuous recording,
not independently-sampled images) alongside a per-model mIoU readout.

Usage
-----
    PYTHONPATH=src python scripts/segmentation/evaluation/compare_on_zikim.py \\
        --val-dir datasets/segmentation/off_road_zikim/val/m24 \\
        --models mask2former-large mask2former-base segformer-b2 \\
        --output reports/segmentation/zikim_comparison/zikim_val_m24.mp4
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts" / "segmentation" / "training"))

from perception.config.loader import load_config
from perception.config.schema import ClassDef, SemanticModelCfg
from perception.models.backends.pytorch import PyTorchBackend
from perception.models.factory import SEMANTIC_DEFAULT_WEIGHTS, build_semantic_model

from _orfd_common import compute_miou

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("compare_on_zikim")

_ORFD_CLASSES: list[ClassDef] = [
    ClassDef(name="non_traversable", text_prompt="-", display_mode="mask_only",
             color_rgb=(220, 40, 40), is_semantic=True, native_indices={}),
    ClassDef(name="traversable", text_prompt="-", display_mode="mask_only",
             color_rgb=(40, 255, 140), is_semantic=True, native_indices={}),
    ClassDef(name="sky", text_prompt="-", display_mode="mask_only",
             color_rgb=(80, 160, 255), is_semantic=True, native_indices={}),
]

# zikim label name -> ORFD class index (None = ignore)
_ZIKIM_TO_ORFD = {
    "road": 1,       # traversable
    "sky": 2,        # sky
    "ground": 0,     # non_traversable
    "unlabeled": None,
    "other": None,
}


def _load_zikim_color_to_orfd(config_path: Path) -> dict[tuple[int, int, int], int | None]:
    labels = json.loads(config_path.read_text())["labels"]
    out: dict[tuple[int, int, int], int | None] = {}
    for name, spec in labels.items():
        rgb = tuple(int(v) for v in spec["color"])
        out[rgb] = _ZIKIM_TO_ORFD.get(name)
    return out


def zikim_color_mask_to_orfd(color_mask_rgb: np.ndarray,
                              color_to_orfd: dict[tuple[int, int, int], int | None]) -> np.ndarray:
    """(H,W,3) RGB color mask -> (H,W) int8 ORFD class indices, -1 = ignore."""
    h, w = color_mask_rgb.shape[:2]
    out = np.full((h, w), -1, dtype=np.int8)
    flat = color_mask_rgb.reshape(-1, 3)
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    class_per_uniq = np.full(len(uniq), -1, dtype=np.int8)
    for i, rgb in enumerate(uniq):
        cls = color_to_orfd.get(tuple(int(v) for v in rgb))
        if cls is not None:
            class_per_uniq[i] = cls
    out = class_per_uniq[inv].reshape(h, w)
    return out


def gather_zikim_frames(val_dir: Path) -> list[tuple[Path, Path, int]]:
    """Pair base images with their _color_mask.png, sorted by frame index."""
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val-dir", default="datasets/segmentation/off_road_zikim/val/m24")
    p.add_argument("--config-zikim", default="datasets/segmentation/off_road_zikim/config_zikim.json")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--models", nargs="+", default=["mask2former-large", "mask2former-base", "segformer-b2"])
    p.add_argument("--output", default="reports/segmentation/zikim_comparison/zikim_val.mp4")
    p.add_argument("--fps", type=float, default=4.0, help="Output video FPS (source is a sparse frame sequence)")
    p.add_argument("--panel-w", type=int, default=380)
    args = p.parse_args()

    val_dir = Path(args.val_dir).resolve()
    color_to_orfd = _load_zikim_color_to_orfd(Path(args.config_zikim).resolve())
    logger.info("zikim color->ORFD map: %s", color_to_orfd)

    pairs = gather_zikim_frames(val_dir)
    if not pairs:
        logger.error("No frames found under %s", val_dir)
        return 2

    cfg = load_config(args.config)
    hw = cfg.hardware
    backend = PyTorchBackend()

    _csm_spec = importlib.util.spec_from_file_location("_csm", _HERE / "compare_semantic_models.py")
    _csm = importlib.util.module_from_spec(_csm_spec)
    sys.modules["_csm"] = _csm
    _csm_spec.loader.exec_module(_csm)
    render_panel = _csm.render_panel

    models = {}
    for key in args.models:
        weights = SEMANTIC_DEFAULT_WEIGHTS.get(key.lower().strip(), "")
        if key.lower().strip() == cfg.models.semantic.name.lower().strip() and cfg.models.semantic.weights:
            weights = cfg.models.semantic.weights
        mdl = build_semantic_model(SemanticModelCfg(name=key, weights=weights), hw, backend)
        mdl.warmup(cfg.classes)
        models[key] = mdl
        logger.info("Loaded %s (weights=%s)", key, weights)

    out_dir = Path(args.output).resolve().parent
    frames_dir = out_dir / "_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    all_preds: dict[str, list[torch.Tensor]] = {k: [] for k in models}
    all_gts: list[torch.Tensor] = []
    frame_paths: list[Path] = []

    for img_path, color_mask_path, idx in pairs:
        img_bgr = cv2.imread(str(img_path))
        color_mask = cv2.imread(str(color_mask_path))
        if img_bgr is None or color_mask is None:
            continue
        color_mask_rgb = cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB)
        gt_orfd = zikim_color_mask_to_orfd(color_mask_rgb, color_to_orfd)

        preds_vis: dict[str, np.ndarray] = {}
        for key, mdl in models.items():
            logits = mdl.predict_logits(img_bgr)
            if logits.shape[-2:] != img_bgr.shape[:2]:
                logits = torch.nn.functional.interpolate(
                    logits.unsqueeze(0), size=img_bgr.shape[:2], mode="bilinear", align_corners=False,
                )[0]
            pred = logits.argmax(dim=0).cpu().numpy().astype(np.int8)
            preds_vis[key] = pred
            all_preds[key].append(torch.from_numpy(pred.astype(np.int64)).unsqueeze(0))

        all_gts.append(torch.from_numpy(gt_orfd.astype(np.int64)).unsqueeze(0))

        out_png = frames_dir / f"frame_{idx:06d}.png"
        render_panel(
            title=f"{img_path.name}  (frame {idx})",
            image_bgr=img_bgr,
            gt_userclass=gt_orfd,
            preds=preds_vis,
            user_classes=_ORFD_CLASSES,
            out_path=out_png,
            target_w=args.panel_w,
        )
        frame_paths.append(out_png)

    if not frame_paths:
        logger.error("No frames rendered.")
        return 2

    for key in models:
        preds_cat = torch.cat(all_preds[key], dim=0)
        gts_cat = torch.cat(all_gts, dim=0)
        miou, per_class = compute_miou(preds_cat, gts_cat, num_classes=3, ignore_index=-1)
        logger.info("%s: mIoU=%.4f  per-class=%s", key, miou,
                    [f"{v:.3f}" if v == v else "nan" for v in per_class])

    first = cv2.imread(str(frame_paths[0]))
    h, w = first.shape[:2]
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    for fp in frame_paths:
        writer.write(cv2.imread(str(fp)))
    writer.release()
    logger.info("Saved %d frames -> %s", len(frame_paths), out_path)

    for fp in frame_paths:
        fp.unlink(missing_ok=True)
    frames_dir.rmdir()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

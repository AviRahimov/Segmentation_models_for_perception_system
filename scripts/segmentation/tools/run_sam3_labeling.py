#!/usr/bin/env python3
"""Auto-label the Gaza-domain image set with SAM3, one prompt per fine-grained
class (see SAM3_FINEGRAINED_NAMES) -- run this in .venv-sam3 (see
requirements-sam3.txt), not the main venv (transformers==4.46.3 there
predates SAM3 support).

Writes ONLY into a review area -- never touches the real training dataset.
For each image: a JSON instance manifest (this repo's own schema, consumed by
rasterize_sam3_labels.py) plus a rendered overlay panel under _preview/ for
manual review. Survival of that overlay file after you delete rejects *is*
the approval signal for promote_gaza_labels.py -- mirrors the exact review
workflow already proven for the inpainting hard-negative pipeline.

Per-class confidence thresholds are independently tunable (DEFAULT_THRESHOLDS
below) -- the whole reason this runs locally instead of only through
Roboflow's Auto Label UI: real per-instance confidence scores and raw mask
logits are available here, not just a final thresholded polygon.

Never prompts SAM3 for "non_traversable" or anything abstract -- confirmed
by two independent sources to cause over-segmentation that no amount of
threshold tuning fixes. Every prompt here is a concrete noun.

Usage
-----
    source .venv-sam3/bin/activate
    python scripts/segmentation/tools/run_sam3_labeling.py \\
        --images-dir "/home/avi/Documents/GitLab/WaterMark_Remover/sota_inpainting_lab/comparisons/final_full_run_flux" \\
        --output-dir datasets/segmentation/gaza_domain_review
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[3]
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "scripts" / "segmentation"))
sys.path.insert(0, str(_HERE))

from _class_catalogues_loader import SAM3_FINEGRAINED_NAMES  # noqa: E402
from rasterize_sam3_labels import (  # noqa: E402
    TRAVERSABLE_COARSE_INDEX, apply_sand_deferral, composite_instances, to_coarse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("run_sam3_labeling")

# Independently tunable per prompt -- SAM3's own docs use 0.5 as a generic
# default; start there for every class and adjust per-class once you've
# looked at real _preview/ output (e.g. "animal" may need a lower bar to
# get anything at all, "building" a higher one if it over-fires on rubble).
DEFAULT_THRESHOLDS: dict[str, float] = {name: 0.5 for name in SAM3_FINEGRAINED_NAMES}

# "ground" (speculative, generic traversable fallback) at 0.5 was found to
# regress otherwise-good images -- it fired confidently enough on things
# that aren't actually traversable to win pixels it shouldn't. Restricting
# it to only its most confident detections (>0.9, strictly -- 0.90 itself
# does not pass) trades away most of its coverage but keeps only the
# detections least likely to be spurious; still being evaluated against
# real review, may be dropped entirely if this isn't a good enough trade.
DEFAULT_THRESHOLDS["ground"] = 0.91 + 1e-6
DEFAULT_THRESHOLDS["sand"] = 0.6

_MIN_CONTOUR_AREA_PX = 30  # drop tiny noise contours when converting mask -> polygon(s)

# Sasha Trubetskoy's 20-distinct-colors set -- chosen for maximum pairwise
# perceptual distinctness, not aesthetics. An earlier ad-hoc palette put
# sand/low_vegetation/animal all in variations of red, which is exactly the
# "similar colors confusing me" problem flagged during real review.
# Listed here as RGB (the commonly published form), converted to BGR tuples
# for cv2 (which draws B,G,R) -- road=blue, dirt_road=green, gravel_road=
# yellow, sand=orange, sky=lavender, building=cyan, rubble=red, high_veg=
# purple, low_veg=lime, vehicle=teal, person=magenta, animal=pink, tent=grey,
# tarp=mint, rock=olive, ground=navy.
_PALETTE = [
    (200, 130, 0), (75, 180, 60), (25, 225, 255), (48, 130, 245),
    (255, 190, 220), (240, 240, 70), (75, 25, 230), (180, 30, 145),
    (60, 245, 210), (128, 128, 0), (230, 50, 240), (212, 190, 250),
    (128, 128, 128), (195, 255, 170), (0, 128, 128), (128, 0, 0),
]


def mask_to_polygons(mask: np.ndarray) -> list[list[list[float]]]:
    """Binary HxW mask -> list of polygons (each a list of [x, y] pairs).

    External contours only (RETR_EXTERNAL) -- same simplification already
    used elsewhere in this repo (_inpaint_common.build_mask's convex-hull
    approach) to avoid jagged internal-hole boundaries; acceptable here
    since the rasterizer fills polygons solid regardless.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for c in contours:
        if cv2.contourArea(c) < _MIN_CONTOUR_AREA_PX:
            continue
        polygons.append(c.reshape(-1, 2).tolist())
    return polygons


def _title_bar(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _composite_fill(image_bgr: np.ndarray, label_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Solid color per pixel = the class that actually won that pixel (no
    outlines, no text) -- this is what rasterize_sam3_labels.py writes to
    the label PNG, just rendered as a picture instead of raw class indices."""
    overlay = image_bgr.copy()
    for idx in range(len(SAM3_FINEGRAINED_NAMES)):
        region = label_map == idx
        if region.any():
            overlay[region] = _PALETTE[idx % len(_PALETTE)]
    return cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)


def _diagnostic_panel(image_bgr: np.ndarray, instances: list[dict], label_map: np.ndarray) -> np.ndarray:
    """Winning composite fill + every instance's outline and confidence,
    even losing ones -- shows not just what won, but why: two overlapping
    outlines with different confidence numbers printed right next to them."""
    blended = _composite_fill(image_bgr, label_map)
    for inst in instances:
        if inst["class_name"] not in SAM3_FINEGRAINED_NAMES:
            continue
        poly = np.array(inst["polygon"], dtype=np.int32)
        idx = SAM3_FINEGRAINED_NAMES.index(inst["class_name"])
        color = _PALETTE[idx % len(_PALETTE)]
        cv2.polylines(blended, [poly], isClosed=True, color=color, thickness=1)
        m = cv2.moments(poly)
        cx = int(m["m10"] / m["m00"]) if m["m00"] else int(poly[:, 0].mean())
        cy = int(m["m01"] / m["m00"]) if m["m00"] else int(poly[:, 1].mean())
        text = f"{inst['confidence']:.2f}"
        cv2.putText(blended, text, (cx - 12, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(blended, text, (cx - 12, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    color, 1, cv2.LINE_AA)
    return blended


def _traversable_panel(image_bgr: np.ndarray, label_map: np.ndarray) -> np.ndarray:
    """Grayscale everything except the final TRAVERSABLE region (bright
    solid green) -- isolates exactly the shape of the path the model will
    be taught is drivable, the actual thing that matters for the freespace
    task, separate from which fine-grained class produced it."""
    coarse = to_coarse(label_map)
    gray = cv2.cvtColor(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    traversable = coarse == TRAVERSABLE_COARSE_INDEX
    overlay = gray.copy()
    overlay[traversable] = (60, 220, 60)  # bright green, BGR
    return cv2.addWeighted(overlay, 0.55, gray, 0.45, 0)


def _build_legend(instances: list[dict], label_map: np.ndarray, conf_map: np.ndarray, width: int = 260) -> np.ndarray:
    class_confidences: dict[str, list[float]] = {}
    for idx, name in enumerate(SAM3_FINEGRAINED_NAMES):
        region = label_map == idx
        if region.any():
            class_confidences[name] = conf_map[region].tolist()
    seen_classes = list(class_confidences)

    legend_h = 22 * (len(seen_classes) + 1)
    legend = np.zeros((legend_h, width, 3), dtype=np.uint8)
    cv2.putText(legend, f"{len(instances)} instances", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    for i, name in enumerate(seen_classes):
        idx = SAM3_FINEGRAINED_NAMES.index(name)
        color = _PALETTE[idx % len(_PALETTE)]
        confs = class_confidences[name]
        y = 22 * (i + 1) + 16
        cv2.rectangle(legend, (6, y - 12), (18, y), color, -1)
        label_text = f"{name} ({min(confs):.2f}-{max(confs):.2f})"
        cv2.putText(legend, label_text, (24, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return legend


def render_preview(image_bgr: np.ndarray, instances: list[dict]) -> np.ndarray:
    """Three panels side by side, left to right:
      1. diagnostic  -- winning fill + every instance's outline/confidence
                        (shows what won an overlap and by how much)
      2. won classes -- the same winning fill alone, no outlines/text
                        (exactly what rasterize_sam3_labels.py writes out)
      3. traversable -- only the final coarse "traversable" region highlighted,
                        everything else grayscale (the actual training target)
    plus a legend (per-class confidence range in this image) sized to match.
    """
    h, w = image_bgr.shape[:2]
    label_map, conf_map, label_map2, conf_map2 = composite_instances(instances, h, w)
    final_map = apply_sand_deferral(label_map, label_map2, conf_map2)

    # Panel 1 deliberately shows the RAW rank-1 decision (sand included, un-
    # deferred) -- true diagnostic info about what SAM3 actually decided.
    # Panels 2/3 show the FINAL decision after sand's own traversable vote
    # is deferred to the second-most-confident detection (see
    # apply_sand_deferral()'s docstring) -- what actually gets exported.
    panel1 = _title_bar(_diagnostic_panel(image_bgr, instances, label_map), "1: diagnostic (all detections)")
    panel2 = _title_bar(_composite_fill(image_bgr, final_map), "2: won classes (= exported label)")
    panel3 = _title_bar(_traversable_panel(image_bgr, final_map), "3: traversable only")
    row = cv2.hconcat([panel1, panel2, panel3])

    legend = _build_legend(instances, label_map, conf_map)
    if legend.shape[0] < row.shape[0]:
        legend = cv2.copyMakeBorder(legend, 0, row.shape[0] - legend.shape[0], 0, 0,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return cv2.hconcat([row, legend[: row.shape[0]]])


def label_image(model, processor, image_bgr: np.ndarray, thresholds: dict[str, float]) -> list[dict]:
    """Runs one SAM3 forward pass per prompt class, returns a flat list of
    {class_name, confidence, polygon} instances across all classes."""
    image_rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    h, w = image_bgr.shape[:2]
    instances: list[dict] = []

    for class_name in SAM3_FINEGRAINED_NAMES:
        prompt = class_name.replace("_", " ")
        inputs = processor(images=image_rgb, text=prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_instance_segmentation(
            outputs, threshold=thresholds[class_name], mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        masks = results["masks"]
        scores = results["scores"]
        for mask, score in zip(masks, scores):
            mask_np = mask.cpu().numpy().astype(np.uint8)
            if mask_np.shape != (h, w):
                mask_np = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
            for polygon in mask_to_polygons(mask_np):
                instances.append({
                    "class_name": class_name,
                    "confidence": float(score),
                    "polygon": polygon,
                })
    return instances


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images-dir", required=True, help="Directory of the 214 unlabeled Gaza-domain images")
    p.add_argument("--output-dir", default="datasets/segmentation/gaza_domain_review",
                   help="Review area -- never the real training dataset")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap, for a quick smoke test")
    args = p.parse_args()

    images_dir = Path(args.images_dir)
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = _ROOT / out_dir
    manifests_dir = out_dir / "manifests"
    preview_dir = out_dir / "_preview"
    images_out_dir = out_dir / "images"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    images_out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if args.max_images:
        image_paths = image_paths[: args.max_images]
    logger.info("Found %d images in %s", len(image_paths), images_dir)

    logger.info("Loading facebook/sam3 ...")
    from transformers import Sam3Model, Sam3Processor
    model = Sam3Model.from_pretrained("facebook/sam3", device_map="auto")
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model.eval()

    for img_path in tqdm(image_paths, desc="SAM3 labeling"):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            logger.warning("Could not read %s -- skipping", img_path)
            continue
        h, w = image_bgr.shape[:2]
        instances = label_image(model, processor, image_bgr, DEFAULT_THRESHOLDS)

        manifest = {"image": img_path.name, "width": w, "height": h, "instances": instances}
        (manifests_dir / f"{img_path.stem}.json").write_text(json.dumps(manifest))

        preview = render_preview(image_bgr, instances)
        cv2.imwrite(str(preview_dir / f"{img_path.stem}.jpg"), preview)

        # Copy (not move) the original into the review area too, so the whole
        # review folder is self-contained -- promote_gaza_labels.py never
        # needs to reach back into the original 214-image source directory.
        shutil.copy2(img_path, images_out_dir / img_path.name)

    logger.info("Done: %d manifests + preview panels -> %s", len(image_paths), out_dir)
    logger.info("Next: run rasterize_sam3_labels.py --manifests-dir %s --output-dir %s, "
                "then look through %s by eye, delete any panel you don't trust, then run "
                "promote_gaza_labels.py --from-preview", manifests_dir, out_dir / "labels", preview_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

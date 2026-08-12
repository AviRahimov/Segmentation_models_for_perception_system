#!/usr/bin/env python3
"""Rasterize per-image SAM3 instance annotations into dense fine-grained
semantic label PNGs (+ optional per-pixel confidence maps).

No COCO/polygon-to-dense-mask converter previously existed anywhere in this
repo (confirmed by direct search) -- this is new code, built on the same
``cv2.fillPoly`` primitive already used for a different purpose in
``scripts/detection/tools/_inpaint_common.py``.

Input contract -- one small JSON "instance manifest" per image (this repo's
own format, written by the SAM3 labeling step in
``scripts/segmentation/tools/run_sam3_labeling.py``, and also what a Roboflow
COCO-instance export gets converted into by ``coco_to_manifests()`` below so
both label sources funnel through the exact same rasterizer):

    {
      "image": "some_frame.png", "width": 900, "height": 506,
      "instances": [
        {"class_name": "dirt_road", "confidence": 0.87, "polygon": [[x1,y1], ...]},
        ...
      ]
    }

Overlap resolution is **confidence-based, not order-based**: where two
instances' polygons overlap, the higher-confidence one wins that pixel,
regardless of upload order or class (see composite_instances()). An earlier
version used a fixed class-index z-order, which let a low-confidence
spurious object detection silently overwrite a correct high-confidence
ground detection underneath it -- caught by manual review of real output.

Pixels no instance ever claims are left at IGNORE_INDEX (255) -- never
force-defaulted to non_traversable. A coverage gap is not evidence of
anything; see the plan's rationale for never prompting SAM3 for
"non_traversable" directly.

Usage
-----
    python scripts/segmentation/tools/rasterize_sam3_labels.py \\
        --manifests-dir datasets/segmentation/gaza_domain_review/manifests \\
        --output-dir datasets/segmentation/gaza_domain_review/labels

    # Roboflow COCO-instance export instead of this repo's own manifests:
    python scripts/segmentation/tools/rasterize_sam3_labels.py \\
        --coco-json datasets/segmentation/gaza_domain_review/roboflow_export/annotations.coco.json \\
        --output-dir datasets/segmentation/gaza_domain_review/labels
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts" / "segmentation"))
sys.path.insert(0, str(_ROOT / "scripts" / "segmentation" / "training"))

from _class_catalogues_loader import COARSE_CLASSES, FINE_TO_COARSE, SAM3_FINEGRAINED_NAMES  # noqa: E402
from _orfd_common import IGNORE_INDEX  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("rasterize_sam3_labels")

_NAME_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(SAM3_FINEGRAINED_NAMES)}
TRAVERSABLE_COARSE_INDEX = COARSE_CLASSES.index("traversable")

# Fine-grained class index -> coarse index (non_traversable=0/traversable=1/
# sky=2, matching ORFDDataset.CLASSES), IGNORE_INDEX passes through unchanged
# since it's outside the 0..len(SAM3_FINEGRAINED_NAMES) range this LUT covers.
_COARSE_OF_FINE = np.full(256, IGNORE_INDEX, dtype=np.uint8)
for _i, _name in enumerate(SAM3_FINEGRAINED_NAMES):
    _COARSE_OF_FINE[_i] = COARSE_CLASSES.index(FINE_TO_COARSE[_name])


def to_coarse(label_map: np.ndarray) -> np.ndarray:
    """Fine-grained label_map (from composite_instances/rasterize_manifest) -> coarse (0/1/2/255)."""
    return _COARSE_OF_FINE[label_map]


def coco_to_manifests(coco_json_path: Path) -> list[dict]:
    """Convert a Roboflow/COCO instance-segmentation export into this
    module's own per-image manifest schema (list of dicts, one per image).

    COCO "segmentation" is a list of polygons (each a flat [x1,y1,x2,y2,...]
    list); RLE segmentations are not handled here since SAM3-via-Roboflow
    exports polygons, not RLE, for this project's use.
    """
    data = json.loads(coco_json_path.read_text())
    cat_names = {cat["id"]: cat["name"] for cat in data["categories"]}

    by_image: dict[int, dict] = {}
    for img in data["images"]:
        by_image[img["id"]] = {
            "image": img["file_name"], "width": img["width"], "height": img["height"],
            "instances": [],
        }

    skipped_rle = 0
    for ann in data["annotations"]:
        seg = ann.get("segmentation")
        if not seg or isinstance(seg, dict):  # dict == RLE, not a polygon list
            skipped_rle += 1
            continue
        class_name = cat_names.get(ann["category_id"])
        if class_name not in _NAME_TO_INDEX:
            continue  # not one of our SAM3_FINEGRAINED_NAMES prompts -- ignore silently
        for flat_poly in seg:
            polygon = list(zip(flat_poly[0::2], flat_poly[1::2]))
            by_image[ann["image_id"]]["instances"].append({
                "class_name": class_name,
                "confidence": float(ann.get("score", 1.0)),
                "polygon": polygon,
            })

    if skipped_rle:
        logger.warning("Skipped %d RLE-encoded annotations (polygons only supported)", skipped_rle)
    return list(by_image.values())


_SAND_INDEX = _NAME_TO_INDEX["sand"]


def composite_instances(instances: list[dict], h: int, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Confidence-aware compositing, shared by the rasterizer and the
    review-preview renderer so what a human sees in _preview/ is exactly
    what ends up in the promoted label PNG.

    Where two instances' polygons overlap, the HIGHER-CONFIDENCE one wins
    for those pixels -- not whichever has the higher SAM3_FINEGRAINED_NAMES
    index. An earlier version used a fixed class-index z-order (traversable-
    adjacent painted first, objects on top unconditionally), which let a
    low-confidence spurious object detection (e.g. a weak "building"/
    "rubble" hit on a small rock or tent) silently overwrite a correct,
    high-confidence ground detection underneath it -- caught by manual
    review of real output, not a theoretical concern.

    Also tracks the second-most-confident detection per pixel
    (label_map2/conf_map2, IGNORE_INDEX/0 wherever fewer than two
    detections ever touched a pixel) -- needed for apply_sand_deferral()
    below: real review found "sand" also fires confidently on steep dune
    slopes, not just genuinely flat drivable ground, so its own traversable
    vote isn't trusted directly anymore.

    Returns (label_map, conf_map, label_map2, conf_map2), each (H, W).
    """
    label_map = np.full((h, w), IGNORE_INDEX, dtype=np.uint8)
    conf_map = np.zeros((h, w), dtype=np.float32)
    label_map2 = np.full((h, w), IGNORE_INDEX, dtype=np.uint8)
    conf_map2 = np.zeros((h, w), dtype=np.float32)

    for inst in instances:
        if inst["class_name"] not in _NAME_TO_INDEX:
            continue
        class_idx = _NAME_TO_INDEX[inst["class_name"]]
        poly = np.array(inst["polygon"], dtype=np.int32)
        if poly.ndim != 2 or poly.shape[0] < 3:
            continue
        inst_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(inst_mask, [poly], 1)
        painted = inst_mask.astype(bool)
        confidence = float(inst.get("confidence", 1.0))

        becomes_rank1 = painted & (confidence > conf_map)
        # Whatever was rank-1 gets bumped down to rank-2 wherever a
        # stronger detection takes over rank-1 for that pixel.
        label_map2[becomes_rank1] = label_map[becomes_rank1]
        conf_map2[becomes_rank1] = conf_map[becomes_rank1]
        label_map[becomes_rank1] = class_idx
        conf_map[becomes_rank1] = confidence

        becomes_rank2 = painted & ~becomes_rank1 & (confidence > conf_map2)
        label_map2[becomes_rank2] = class_idx
        conf_map2[becomes_rank2] = confidence

    return label_map, conf_map, label_map2, conf_map2


def apply_sand_deferral(label_map: np.ndarray, label_map2: np.ndarray, conf_map2: np.ndarray) -> np.ndarray:
    """Wherever "sand" wins a pixel, defer to the second-most-confident
    detection there instead -- sand's own traversable vote isn't trusted
    directly (confirmed to also claim steep dune slopes, not just flat
    drivable ground). If no second detection exists for that pixel, leave
    it IGNORE_INDEX rather than force a guess either way. Returns a new
    array; does not mutate label_map."""
    out = label_map.copy()
    is_sand = label_map == _SAND_INDEX
    has_second = is_sand & (conf_map2 > 0)
    out[has_second] = label_map2[has_second]
    out[is_sand & ~has_second] = IGNORE_INDEX
    return out


def rasterize_manifest(manifest: dict) -> tuple[np.ndarray, np.ndarray]:
    """Returns (label_map uint8 (H,W), post sand-deferral -- this is the
    label actually written to the PNG -- confidence_map float32 (H,W),
    0 where ignored)."""
    label_map, conf_map, label_map2, conf_map2 = composite_instances(
        manifest["instances"], manifest["height"], manifest["width"])
    final_map = apply_sand_deferral(label_map, label_map2, conf_map2)
    # Confidence map follows the same deferral so a sand->X pixel reports
    # X's own confidence, not sand's.
    final_conf = np.where(final_map == label_map, conf_map, conf_map2)
    final_conf[final_map == IGNORE_INDEX] = 0.0
    return final_map, final_conf


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifests-dir", help="Directory of this repo's own per-image JSON manifests")
    src.add_argument("--coco-json", help="Path to a Roboflow/COCO instance-segmentation export")
    p.add_argument("--output-dir", required=True, help="Where to write <stem>.png + <stem>_conf.npy")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.coco_json:
        manifests = coco_to_manifests(Path(args.coco_json))
    else:
        manifests_dir = Path(args.manifests_dir)
        manifests = [json.loads(f.read_text()) for f in sorted(manifests_dir.glob("*.json"))]

    logger.info("Rasterizing %d image manifests -> %s", len(manifests), out_dir)
    for manifest in manifests:
        stem = Path(manifest["image"]).stem
        label_map, conf_map = rasterize_manifest(manifest)
        cv2.imwrite(str(out_dir / f"{stem}.png"), label_map)
        np.save(out_dir / f"{stem}_conf.npy", conf_map)
        n_ignored = int((label_map == IGNORE_INDEX).sum())
        coverage = 1.0 - n_ignored / label_map.size
        logger.info("  %-60s %d instances  coverage=%.1f%%", stem, len(manifest["instances"]), coverage * 100)

    logger.info("Done: %d label PNGs + confidence maps written to %s", len(manifests), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

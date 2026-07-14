"""Manifest-driven dataset builder — merge/remap multiple YOLO sources.

One reproducible command per dataset variant; the manifest (YAML, version-
controlled under config/detection/datasets/) fully describes sources, class
mappings, filters, oversampling, and split policy:

    python scripts/detection/tools/build_dataset.py \\
        --manifest config/detection/datasets/merged_2class.yaml

Manifest schema
---------------
    name: merged_2class
    output: datasets/merged_2class
    classes: ["Military Vehicle", "person"]   # target scheme, index = class id
    min_box_px: 8                # drop boxes smaller than this (pixels, noise)
    resplit: {val_fraction: 0.05, seed: 42}   # for sources pooled below
    sources:
      - path: datasets/Some_Dataset
        format: yolo_splits      # dirs with images/ + labels/ per split + yaml
        include_splits: {train: train, valid: val}   # src split -> target split
        # 'pool' as target = collect then re-split by `resplit`
        oversample_train: 10     # duplicate factor for this source's train images
        map: {src_class_name: target_class_name, ...}   # absent = dropped
        negatives: {from_classes: [names...], fraction: 0.08}
        filename_exclude: [substr, ...]
      - path: .../class_dir_dataset
        format: class_dirs       # images_awaiting/<Class>/ + labels_awaiting/<Class>/
        map: {DirName: target_class_name, ...}

Notes
-----
- Class resolution is by NAME (source yaml `names:` / directory names), never
  by raw id — robust to id reshuffles between exports.
- `class_dirs` sources encode the class in the directory name; label-file ids
  are ignored (verrckter convention: every label line has id 0).
- Oversampling writes suffixed symlinks (img__dupN.ext) + copied label files.
- Output: {out}/{train,val}/{images,labels} + data.yaml + build_manifest.json
  (provenance: manifest copy, git commit, per-source counts).
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("build_dataset")

_IMG_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


# --------------------------------------------------------------------------- #
# Label parsing / rewriting                                                    #
# --------------------------------------------------------------------------- #

@dataclass
class _Item:
    """One image + its remapped label lines, destined for a target split."""
    img: Path
    lines: list[str]          # remapped label lines ('' lines never included)
    target: str               # 'train' | 'val' | 'pool'
    source: str               # source dataset name (for stats)
    is_negative: bool = False


def _resolve_names(names_raw: Any) -> list[str]:
    if isinstance(names_raw, dict):
        return [str(names_raw[k]) for k in sorted(names_raw)]
    return [str(n) for n in (names_raw or [])]


def _remap_lines(
    text: str,
    id_to_target: dict[int, int],
    min_box_px: float,
    img_wh: tuple[int, int] | None,
) -> tuple[list[str], Counter, int]:
    """Remap label lines by id map; returns (kept, per-src-id counts, n_tiny)."""
    kept: list[str] = []
    seen: Counter = Counter()
    n_tiny = 0
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        seen[cid] += 1
        if cid not in id_to_target:
            continue
        if img_wh is not None and min_box_px > 0 and len(parts) == 5:
            w_px = float(parts[3]) * img_wh[0]
            h_px = float(parts[4]) * img_wh[1]
            if w_px < min_box_px or h_px < min_box_px:
                n_tiny += 1
                continue
        kept.append(" ".join([str(id_to_target[cid])] + parts[1:]))
    return kept, seen, n_tiny


def _image_size(path: Path) -> tuple[int, int] | None:
    try:
        import cv2
        img = cv2.imread(str(path))
        if img is None:
            return None
        h, w = img.shape[:2]
        return (w, h)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Source readers                                                               #
# --------------------------------------------------------------------------- #

def _read_yolo_splits(src_cfg: dict, target_classes: list[str],
                      min_box_px: float, stats: dict) -> list[_Item]:
    src_dir = _ROOT / src_cfg["path"]
    name = src_dir.name

    # Resolve source class names from its yaml (any *.yaml with names:)
    names: list[str] | None = None
    for y in sorted(src_dir.glob("*.yaml")) + sorted(src_dir.glob("*.yml")):
        try:
            raw = yaml.safe_load(y.read_text()) or {}
        except Exception:
            continue
        if isinstance(raw, dict) and "names" in raw:
            names = _resolve_names(raw["names"])
            break
    if names is None:
        raise SystemExit(f"{name}: no yaml with 'names:' found in {src_dir}")

    cls_map: dict[str, str] = src_cfg.get("map", {}) or {}
    unknown = [k for k in cls_map if k not in names]
    if unknown:
        raise SystemExit(f"{name}: map keys not in source classes: {unknown} "
                         f"(source has {names})")
    id_to_target = {
        names.index(src_name): target_classes.index(tgt_name)
        for src_name, tgt_name in cls_map.items()
    }
    neg_cfg = src_cfg.get("negatives") or {}
    neg_source_ids = {names.index(n) for n in neg_cfg.get("from_classes", [])
                      if n in names}
    excl = [s.lower() for s in (src_cfg.get("filename_exclude") or [])]

    items: list[_Item] = []
    negatives_pool: list[_Item] = []
    src_counts: Counter = Counter()
    n_tiny_total = 0

    for src_split, tgt_split in (src_cfg.get("include_splits") or {}).items():
        img_dir = src_dir / src_split / "images"
        lbl_dir = src_dir / src_split / "labels"
        if not img_dir.is_dir():
            raise SystemExit(f"{name}: missing split dir {img_dir}")
        for img in sorted(p for p in img_dir.iterdir()
                          if p.suffix.lower() in _IMG_SUFFIXES):
            if excl and any(s in img.name.lower() for s in excl):
                stats.setdefault("filename_excluded", Counter())[name] += 1
                continue
            lbl = lbl_dir / img.with_suffix(".txt").name
            text = lbl.read_text() if lbl.exists() else ""
            wh = _image_size(img) if min_box_px > 0 else None
            kept, seen, n_tiny = _remap_lines(text, id_to_target, min_box_px, wh)
            src_counts.update(seen)
            n_tiny_total += n_tiny
            if kept:
                items.append(_Item(img, kept, tgt_split, name))
            elif seen.keys() & neg_source_ids:
                negatives_pool.append(_Item(img, [], tgt_split, name, is_negative=True))

    # Hard negatives budget (only train-destined images count)
    if neg_cfg:
        budget = int(sum(1 for i in items if i.target == "train")
                     * float(neg_cfg.get("fraction", 0.08)))
        chosen = negatives_pool[:budget] if budget else []
        items.extend(chosen)
        stats.setdefault("negatives", {})[name] = len(chosen)

    stats.setdefault("instances_by_source", {})[name] = {
        names[cid]: n for cid, n in sorted(src_counts.items())
    }
    stats.setdefault("tiny_boxes_dropped", Counter())[name] = n_tiny_total
    return items


def _read_class_dirs(src_cfg: dict, target_classes: list[str],
                     min_box_px: float, stats: dict) -> list[_Item]:
    """images_awaiting/<Class>/ + labels_awaiting/<Class>/ layout.

    The class is the DIRECTORY name; label-file ids are ignored (they are all
    0 in the verrckter export).
    """
    src_dir = _ROOT / src_cfg["path"]
    name = src_cfg.get("alias") or src_dir.parent.parent.name
    img_root = src_dir / "images_awaiting"
    lbl_root = src_dir / "labels_awaiting"
    if not img_root.is_dir() or not lbl_root.is_dir():
        raise SystemExit(f"{name}: expected images_awaiting/ + labels_awaiting/ under {src_dir}")

    cls_map: dict[str, str] = src_cfg.get("map", {}) or {}
    excl = [s.lower() for s in (src_cfg.get("filename_exclude") or [])]
    items: list[_Item] = []
    src_counts: Counter = Counter()
    n_tiny_total = 0

    for class_dir in sorted(p for p in img_root.iterdir() if p.is_dir()):
        cls_name = class_dir.name
        if cls_name not in cls_map:
            stats.setdefault("dirs_dropped", {}).setdefault(name, []).append(cls_name)
            continue
        target_id = target_classes.index(cls_map[cls_name])
        for img in sorted(p for p in class_dir.iterdir()
                          if p.suffix.lower() in _IMG_SUFFIXES):
            if excl and any(s in img.name.lower() for s in excl):
                stats.setdefault("filename_excluded", Counter())[name] += 1
                continue
            lbl = lbl_root / cls_name / img.with_suffix(".txt").name
            if not lbl.exists():
                continue
            wh = _image_size(img) if min_box_px > 0 else None
            # ids in file are meaningless — map every valid line to target_id
            kept: list[str] = []
            for line in lbl.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                src_counts[cls_name] += 1
                if wh is not None and min_box_px > 0 and len(parts) == 5:
                    if (float(parts[3]) * wh[0] < min_box_px
                            or float(parts[4]) * wh[1] < min_box_px):
                        n_tiny_total += 1
                        continue
                kept.append(" ".join([str(target_id)] + parts[1:]))
            if kept:
                items.append(_Item(img, kept, "pool", name))

    stats.setdefault("instances_by_source", {})[name] = dict(sorted(src_counts.items()))
    stats.setdefault("tiny_boxes_dropped", Counter())[name] = n_tiny_total
    return items


# --------------------------------------------------------------------------- #
# Assembly                                                                     #
# --------------------------------------------------------------------------- #

def _unique_dest(base_dir: Path, img: Path, tag: str) -> Path:
    """Collision-safe destination name: <tag>__<original-name>."""
    return base_dir / f"{tag}__{img.name}"


def _place(item: _Item, out_root: Path, split: str, tag: str,
           copy_images: bool, dup_idx: int | None = None) -> None:
    img_dir = out_root / split / "images"
    lbl_dir = out_root / split / "labels"
    stem_tag = tag if dup_idx is None else f"{tag}__dup{dup_idx}"
    dest_img = _unique_dest(img_dir, item.img, stem_tag)
    if copy_images:
        shutil.copy2(item.img, dest_img)
    else:
        dest_img.symlink_to(item.img.resolve())
    (lbl_dir / (dest_img.stem + ".txt")).write_text(
        "\n".join(item.lines) + ("\n" if item.lines else "")
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Build a merged dataset from a YAML manifest")
    p.add_argument("--manifest", required=True)
    p.add_argument("--copy-images", action="store_true", dest="copy_images")
    args = p.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = _ROOT / manifest_path
    manifest = yaml.safe_load(manifest_path.read_text())

    target_classes: list[str] = [str(c) for c in manifest["classes"]]
    min_box_px = float(manifest.get("min_box_px", 0))
    resplit = manifest.get("resplit") or {}
    val_fraction = float(resplit.get("val_fraction", 0.05))
    seed = int(resplit.get("seed", 42))

    out_root = Path(manifest["output"])
    if not out_root.is_absolute():
        out_root = _ROOT / out_root
    if out_root.exists():
        shutil.rmtree(out_root)
    for split in ("train", "val"):
        (out_root / split / "images").mkdir(parents=True)
        (out_root / split / "labels").mkdir(parents=True)

    stats: dict = {}
    all_items: list[_Item] = []
    oversample: dict[str, int] = {}

    for src_cfg in manifest["sources"]:
        fmt = src_cfg.get("format", "yolo_splits")
        if fmt == "yolo_splits":
            items = _read_yolo_splits(src_cfg, target_classes, min_box_px, stats)
        elif fmt == "class_dirs":
            items = _read_class_dirs(src_cfg, target_classes, min_box_px, stats)
        else:
            raise SystemExit(f"Unknown source format: {fmt}")
        src_name = items[0].source if items else src_cfg["path"]
        oversample[src_name] = int(src_cfg.get("oversample_train", 1))
        all_items.extend(items)
        logger.info("source %-35s → %5d images", src_name, len(items))

    # Re-split pooled items deterministically.
    pooled = [i for i in all_items if i.target == "pool"]
    if pooled:
        rng = random.Random(seed)
        rng.shuffle(pooled)
        n_val = int(len(pooled) * val_fraction)
        for i, item in enumerate(pooled):
            item.target = "val" if i < n_val else "train"
        logger.info("re-split %d pooled images → %d val / %d train",
                    len(pooled), n_val, len(pooled) - n_val)

    # Write output; oversampling applies to train items only.
    split_counts: Counter = Counter()
    for item in all_items:
        tag = item.source
        _place(item, out_root, item.target, tag, args.copy_images)
        split_counts[item.target] += 1
        if item.target == "train" and not item.is_negative:
            for d in range(1, oversample.get(item.source, 1)):
                _place(item, out_root, "train", tag, args.copy_images, dup_idx=d)
                split_counts["train_dups"] += 1

    # data.yaml
    names_yaml = "\n".join(f"  {i}: {n}" for i, n in enumerate(target_classes))
    (out_root / "data.yaml").write_text(
        f"# Generated by build_dataset.py from {manifest_path.name}\n"
        f"path: {out_root.resolve()}\n"
        f"train: train/images\nval: val/images\n"
        f"nc: {len(target_classes)}\nnames:\n{names_yaml}\n"
    )

    # Provenance
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                cwd=_ROOT, capture_output=True, text=True,
                                check=False).stdout.strip()
    except Exception:
        commit = "unknown"
    provenance = {
        "manifest_file": str(manifest_path.relative_to(_ROOT)),
        "manifest": manifest,
        "git_commit": commit,
        "split_counts": dict(split_counts),
        "stats": {k: (dict(v) if isinstance(v, Counter) else v)
                  for k, v in stats.items()},
    }
    (out_root / "build_manifest.json").write_text(json.dumps(provenance, indent=2))

    # Final class distribution
    dist: Counter = Counter()
    for split in ("train", "val"):
        for lbl in (out_root / split / "labels").iterdir():
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if parts:
                    dist[(split, target_classes[int(parts[0])])] += 1

    logger.info("")
    logger.info("Dataset %s written to %s", manifest["name"], out_root)
    logger.info("  train images: %d (+%d oversample dups)   val images: %d",
                split_counts["train"], split_counts["train_dups"], split_counts["val"])
    for (split, cls), n in sorted(dist.items()):
        logger.info("  %-5s %-18s %6d instances", split, cls, n)
    for src, n in (stats.get("tiny_boxes_dropped") or {}).items():
        if n:
            logger.info("  tiny boxes dropped (%s): %d", src, n)
    for src, n in (stats.get("negatives") or {}).items():
        logger.info("  hard negatives kept (%s): %d", src, n)
    for src, n in (stats.get("filename_excluded") or {}).items():
        logger.info("  filename-excluded (%s): %d", src, n)
    logger.info("data.yaml → %s", out_root / "data.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Phase 3 — side-by-side comparison of every inpainting backend tried.

Builds one wide panel per (image, variant): [original | mask overlay | LaMa |
ZITS], reusing _inpaint_common.py's mask-building so the overlay reflects the
exact mask each model actually received (both models ran on identical
sampled images + identical masks -- the only fair comparison).

Usage
-----
    .venv-inpaint/bin/python3 scripts/detection/tools/compare_inpaint_models.py

Output: datasets/Detection_Dataset_inpaint_review_comparison/<variant>__<stem>.jpg
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _inpaint_common import SRC_DIR, build_mask, build_variants, parse_label_lines  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("compare_inpaint_models")

_ROOT = Path(__file__).resolve().parents[3]
_MODELS = {
    "LaMa": _ROOT / "datasets/Detection_Dataset_inpaint_review",
    "ZITS": _ROOT / "datasets/Detection_Dataset_inpaint_review_zits",
}
_OUT_DIR = _ROOT / "datasets/Detection_Dataset_inpaint_review_comparison"


def _label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _mask_overlay(orig_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    red = np.zeros_like(orig_bgr)
    red[:, :, 2] = 255
    alpha = (mask.astype(np.float32) / 255.0)[..., None] * 0.5
    return (orig_bgr * (1 - alpha) + red * alpha).astype(np.uint8)


def main() -> int:
    for name, path in _MODELS.items():
        if not path.exists():
            logger.error("Missing model output dir for %s: %s", name, path)
            return 1

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    lbl_dir = SRC_DIR / "labels"

    # Discover (variant, stem) pairs from the first model's output -- both
    # models ran on the identical sampled set, so either works as the index.
    first_root = next(iter(_MODELS.values()))
    pairs: list[tuple[str, str]] = []
    for variant_dir in sorted(first_root.iterdir()):
        if not variant_dir.is_dir() or variant_dir.name.startswith("_"):
            continue
        img_dir = variant_dir / "images"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.iterdir()):
            pairs.append((variant_dir.name, img_path.stem))

    logger.info("Building %d comparison panel(s) across %d model(s)", len(pairs), len(_MODELS))
    n_missing = 0
    for variant, stem in pairs:
        orig_path = SRC_DIR / "images" / f"{stem}.jpg"
        orig_bgr = cv2.imread(str(orig_path))
        if orig_bgr is None:
            logger.warning("Could not read original image for %s, skipping", stem)
            continue
        h, w = orig_bgr.shape[:2]

        lines = parse_label_lines(lbl_dir / f"{stem}.txt", w, h)
        variants = {v: (polys, protect) for v, polys, _, protect in build_variants(lines)}
        polys, protect = variants[variant]
        mask = build_mask(polys, h, w, protect_polys=protect)

        panels = [_label(orig_bgr, "original"), _label(_mask_overlay(orig_bgr, mask), "mask")]
        for name, root in _MODELS.items():
            candidates = list((root / variant / "images").glob(f"{stem}.*"))
            if not candidates:
                logger.warning("Missing %s output for %s/%s", name, variant, stem)
                n_missing += 1
                panels.append(_label(np.zeros_like(orig_bgr), f"{name} (missing)"))
                continue
            result_bgr = cv2.imread(str(candidates[0]))
            panels.append(_label(result_bgr, name))

        panel = np.hstack(panels)
        cv2.imwrite(str(_OUT_DIR / f"{variant}__{stem}.jpg"), panel)

    if n_missing:
        logger.warning("%d panel(s) had a missing model output -- see warnings above", n_missing)
    logger.info("Done. %d comparison panels -> %s", len(pairs), _OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

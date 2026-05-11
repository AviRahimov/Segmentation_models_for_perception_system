#!/usr/bin/env python3
"""ORFD freespace comparison: SegFormer-B2 / B4 / DDRNet vs ORFD traversable GT.

Pairs ``training/image_data/<id>.png`` with ``training/gt_image/<id>_fillcolor.png``.
By default (omit ``--legacy-multiclass``) uses **fillcolor encoding** measured on
the upstream ORFD release: **bright / 255 = traversable path**, **0 = blocked**,
**128 = sky band omitted from IoU** (so DDRNet/SegFormer sky false positives
outside the drivable mask do not inflate scores). Tune ``orfd_trav_gray`` under
``orfd_semantic_comparison`` in ``config/config.yaml`` if your mirror swaps encodings.

Official dataset layout reference:
https://github.com/chaytonmin/Off-Road-Freespace-Detection

Writes (default):

* ``<output-dir>/strips/orfd_<stem>.png`` and ``goose_<scenario>_<stem>.png``
  — one horizontal strip per frame with traversable-binary IoU in the band.
* ``<output-dir>/README.txt`` — run summary.

Optional ``--single-mosaic`` concatenates strips into ``orfd_mosaic.png``.
Extra GOOSE val sampling, ORFD gray value, and optional YOLOE mask subtraction
are configured under ``orfd_semantic_comparison`` in ``config.yaml``.

Examples::

    PYTHONPATH=src python scripts/orfd_semantic_comparison.py \\
        --training-root datasets/orfd/training --samples 20

    PYTHONPATH=src python scripts/orfd_semantic_comparison.py \\
        --single-mosaic --sanity-write-gt-raw
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent


_COMPARE_SEMANTIC_MOD: object | None = None


def _compare_semantic_module():
    """Load ``compare_semantic_models.py`` once (render_panel + GOOSE helpers)."""
    global _COMPARE_SEMANTIC_MOD
    if _COMPARE_SEMANTIC_MOD is None:
        sys.path.insert(0, str(_REPO / "src"))
        spec = importlib.util.spec_from_file_location(
            "_csm", _HERE / "compare_semantic_models.py",
        )
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _COMPARE_SEMANTIC_MOD = mod
    return _COMPARE_SEMANTIC_MOD


def render_panel(*args, **kwargs):
    return _compare_semantic_module().render_panel(*args, **kwargs)

from perception.config.loader import load_config  # noqa: E402
from perception.config.schema import ClassDef, InstanceModelCfg, SemanticModelCfg  # noqa: E402
from perception.datasets.orfd_labels import (  # noqa: E402
    ORFD_NON_TRAVERSABLE_GRAY,
    ORFD_SKY_BAND_GRAY,
    ORFD_TRAVERSABLE_GRAY,
    binary_traversable_iou,
    orfd_eval_valid_mask,
    orfd_traversable_gt_mask,
)
from perception.models.backends.pytorch import PyTorchBackend  # noqa: E402
from perception.models.factory import (  # noqa: E402
    SEMANTIC_DEFAULT_WEIGHTS,
    build_instance_model,
    build_semantic_model,
)
from perception.models.instance.base import InstanceModel  # noqa: E402

logger = logging.getLogger("orfd_semantic_comparison")

#: Deprecated: legacy remap for ``--legacy-multiclass``.
ORFD_GRAY_TO_CLASS: dict[int, str] = {
    ORFD_NON_TRAVERSABLE_GRAY: "sand_gravel",
    ORFD_SKY_BAND_GRAY: "sky",
    ORFD_TRAVERSABLE_GRAY: "road_ground",
}

_TRAV_CLASSES: tuple[ClassDef, ...] = (
    ClassDef(
        name="Traversable",
        text_prompt="-",
        display_mode="mask_only",
        color_rgb=(40, 255, 140),
        is_semantic=True,
        native_indices={},
    ),
)


def gather_orfd_pairs(training_root: Path) -> list[tuple[Path, Path]]:
    """Align ``image_data/*.png`` with ``gt_image/<id>_fillcolor.png``."""
    img_dir = training_root / "image_data"
    gt_dir = training_root / "gt_image"
    if not img_dir.is_dir() or not gt_dir.is_dir():
        logger.error(
            "Expected ORFD layout under %s: image_data/ and gt_image/ missing.",
            training_root,
        )
        return []

    imgs = {p.stem: p for p in img_dir.glob("*.png")}
    stems = [
        s
        for s in sorted(imgs.keys())
        if (gt_dir / f"{s}_fillcolor.png").is_file()
    ]
    pairs = [(imgs[s], gt_dir / f"{s}_fillcolor.png") for s in stems]
    logger.info("Found %d image/GT pairs under %s", len(pairs), training_root)
    return pairs


@dataclass(frozen=True)
class FreespaceFrame:
    """One evaluation frame with precomputed freespace masks."""

    strip_name: str  # basename for ``strips/<strip_name>.png`` (globally unique)
    img_path: Path
    img_bgr: np.ndarray
    gt_trav: np.ndarray  # bool
    valid: np.ndarray  # bool — exclude sky/ambiguous from IoU/scoring


def gather_goose_scenario_pairs(
    goose_root: Path,
    scenario_dir_name: str,
) -> list[tuple[Path, Path]]:
    csm = _compare_semantic_module()
    scenario = goose_root / "images" / "val" / scenario_dir_name
    if not scenario.is_dir():
        logger.warning("GOOSE val scenario folder missing: %s", scenario)
        return []
    out = [
        pair
        for pair in csm.goose_ex_val_pairs(goose_root)
        if pair[0].parent.name == scenario_dir_name
    ]
    logger.info(
        "GOOSE %s: %d image/label pairs under %s",
        scenario_dir_name,
        len(out),
        goose_root,
    )
    return out


def goose_freespace_masks(
    label_u8: np.ndarray,
    fine_to_cat: np.ndarray,
    *,
    trav_cat_idxs: tuple[int, ...],
    sky_cat_idx: int,
    void_cat_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """GOOSE fine labelids → binary traversable GT + valid mask (sky/void excluded).

    ``spot_scenario03`` (and much off-road footage) annotates gravel/dirt as
    category **terrain**, not **road**; ``road`` may be absent in every pixel.
    Default traversable categories come from ``config.yaml`` →
    ``orfd_semantic_comparison.goose.traversable_categories``.
    """
    lbl = label_u8.astype(np.int32, copy=False)
    max_f = int(fine_to_cat.shape[0])
    oob = (lbl < 0) | (lbl >= max_f)
    goose_cat = np.full(lbl.shape, -1, dtype=np.int16)
    goose_cat[~oob] = fine_to_cat[lbl[~oob]].astype(np.int16, copy=False)

    t_idx = np.array(trav_cat_idxs, dtype=np.int16)
    gt_trav = np.isin(goose_cat, t_idx)
    valid = (goose_cat >= 0) & (goose_cat != int(sky_cat_idx)) & (goose_cat != int(void_cat_idx))
    return gt_trav, valid


def goose_trav_category_indices(
    goose_12_names: tuple[str, ...],
    categories: tuple[str, ...],
) -> tuple[int, ...]:
    """Map validated GOOSE coarse names (order as in config) to channel indices."""
    m = {n: i for i, n in enumerate(goose_12_names)}
    return tuple(m[c] for c in categories)


def _fs_name_key(prefix: str, stem: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_.-]+", "_", f"{prefix}_{stem}")
    return safe[:200] if len(safe) > 200 else safe


def subtract_instance_masks_from_traversable(
    pred_trav: np.ndarray,
    img_bgr: np.ndarray,
    instance_model: InstanceModel,
    *,
    dilate_px: int = 0,
) -> np.ndarray:
    """Clear traversable predictions under YOLOE instance segmentation masks."""
    out = np.asarray(pred_trav, dtype=bool, order="C").copy()
    for det in instance_model.predict(img_bgr):
        mk = det.mask
        if mk is None:
            continue
        block = mk.astype(bool, copy=False)
        if dilate_px > 0:
            k = max(1, int(dilate_px) * 2 + 1)
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            block = cv2.dilate(block.astype(np.uint8), ker) > 0
        out &= ~block
    return out


def orfd_gt_to_user_indices(
    gt_u8: np.ndarray,
    class_order: tuple[str, ...],
    gray_to_name: dict[int, str] = ORFD_GRAY_TO_CLASS,
) -> np.ndarray:
    """Greyscale ORFD GT → (H,W) int8 user-class indices (-1 ignored)."""
    if gt_u8.ndim != 2:
        raise ValueError(f"Expected single-channel GT, shape={gt_u8.shape}")
    name_to_idx = {n: i for i, n in enumerate(class_order)}
    out = np.full(gt_u8.shape, -1, dtype=np.int8)
    for v_int, nm in gray_to_name.items():
        j = name_to_idx.get(nm)
        if j is None:
            continue
        out[gt_u8 == v_int] = j
    return out


def per_image_mean_iou(
    pred: np.ndarray,
    gt_user: np.ndarray,
    n_classes: int,
) -> float | None:
    """Mean IoU over user classes present in GT (-1 ignores)."""
    valid = gt_user >= 0
    if not valid.any():
        return None
    intersections: list[float] = []
    for c in range(n_classes):
        g = valid & (gt_user == c)
        if not g.any():
            continue
        inter = np.logical_and(pred == c, g).sum(dtype=np.float64)
        uni = np.logical_or(pred == c, g).astype(bool) & valid
        uni = uni.sum(dtype=np.float64)
        if uni > 0:
            intersections.append(float(inter / uni))

    return float(np.mean(intersections)) if intersections else None


def assemble_mosaic_vertical(
    row_images: list[np.ndarray],
    *,
    gap_px: int = 6,
    pad_color: tuple[int, int, int] = (32, 32, 32),
) -> np.ndarray:
    if not row_images:
        raise ValueError("assemble_mosaic_vertical: empty row list")

    mw = max(r.shape[1] for r in row_images)
    total_h = sum(r.shape[0] for r in row_images) + gap_px * (len(row_images) - 1)
    mosaic = np.full((total_h, mw, 3), pad_color, dtype=np.uint8)
    y = 0
    for r in row_images:
        h, w = r.shape[:2]
        x_off = (mw - w) // 2
        mosaic[y : y + h, x_off : x_off + w] = r
        y += h + gap_px
    return mosaic


def footer_freespace_strip(width: int, height: int) -> np.ndarray:
    """Footer for binary freespace mode."""
    bar = np.full((height, width, 3), (28, 28, 28), dtype=np.uint8)
    cv2.rectangle(bar, (0, 0), (width - 1, height - 1), (140, 140, 140), 1)

    lines = [
        "ORFD fillcolor (measured upstream): gray=255 traversable path;",
        "gray=0 blocked; gray=128 sky band (ignored in IoU — not scored).",
        "Panels: predicted road_ground (traversable) vs GT path; excludes sky pixels.",
        (
            "https://github.com/chaytonmin/Off-Road-Freespace-Detection — "
            "use --sanity-write-gt-raw if mirror encoding differs."
        ),
    ]

    y = 22
    for ln in lines:
        cv2.putText(
            bar, ln[:105], (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.41, (220, 220, 220), 1, cv2.LINE_AA,
        )
        y += 18
        if y > height - 8:
            break
    return bar


def footer_legend_legacy(
    width: int,
    height: int,
    *,
    user_classes: list[ClassDef],
    gray_to_name: dict[int, str],
) -> np.ndarray:
    """Footer for deprecated multi-class remap mode."""
    bar = np.full((height, width, 3), (28, 28, 28), dtype=np.uint8)
    rev_line = "; ".join(
        f"gray={gv} -> {cls}" for gv, cls in sorted(gray_to_name.items())
    )
    pale_line = "User classes | " + "; ".join(
        f"{c.name} [{c.color_rgb[0]}, {c.color_rgb[1]}, {c.color_rgb[2]}]"
        for c in user_classes
    )

    cv2.rectangle(bar, (0, 0), (width - 1, height - 1), (140, 140, 140), 1)

    cv2.putText(
        bar, "(legacy remap) GT grey → YAML class names:", (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 200, 140), 1, cv2.LINE_AA,
    )
    y = 42
    for chunk in (
        rev_line[:100],
        rev_line[100:200],
        pale_line[:100],
        pale_line[100:200],
        pale_line[200:300],
    ):
        if not chunk.strip():
            continue
        cv2.putText(
            bar, chunk, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
            (200, 200, 200), 1, cv2.LINE_AA,
        )
        y += 16
        if y > height - 6:
            break
    return bar


def strip_row_band_with_metrics(row_bgr: np.ndarray, metrics_line: str) -> np.ndarray:
    h, w = row_bgr.shape[:2]
    n_splits = metrics_line.count(" | ") + 1
    band_h = max(28, min(72, 16 * n_splits + 10))
    band = np.zeros((band_h, w, 3), dtype=np.uint8)
    y = 16
    for ln in metrics_line.split(" | "):
        cv2.putText(
            band, ln[:118], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
            (235, 235, 170), 1, cv2.LINE_AA,
        )
        y += 17
        if y > band_h - 6:
            break
    return np.vstack([row_bgr, band])


def gt_sanity_panel(gt_gray_u8: np.ndarray) -> np.ndarray:
    """Left: grayscale view; Right: RGB legend 0=R 128=G 255=B."""
    g = gt_gray_u8.astype(np.uint8, copy=False)
    vis = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)

    fc = np.zeros((*g.shape, 3), dtype=np.uint8)
    fc[g == ORFD_NON_TRAVERSABLE_GRAY] = (0, 0, 255)
    fc[g == ORFD_TRAVERSABLE_GRAY] = (0, 255, 0)
    fc[g == ORFD_SKY_BAND_GRAY] = (255, 0, 0)
    uniq = np.unique(g)
    for u in uniq:
        uu = int(u)
        if uu in (
            ORFD_NON_TRAVERSABLE_GRAY,
            ORFD_TRAVERSABLE_GRAY,
            ORFD_SKY_BAND_GRAY,
        ):
            continue
        fc[g == u] = (255, 0, 255)

    lh, lw = g.shape[:2]
    mono3 = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    lbl = mono3.copy()
    cv2.putText(lbl, "GT gray", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (240, 240, 240), 1, cv2.LINE_AA)
    lc = fc.copy()
    cv2.putText(lc, "BGR: 0=blocked 128=sky(exc) 255=path", (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    uniq_text = ",".join(str(int(x)) for x in uniq[:15])
    if uniq.size > 15:
        uniq_text += ",..."
    cv2.putText(
        lc, f"unique={uniq_text}", (6, lh - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 250), 1, cv2.LINE_AA,
    )

    return np.hstack([lbl, lc])


def _road_ground_channel_index(names: tuple[str, ...]) -> int:
    try:
        return names.index("road_ground")
    except ValueError as e:
        raise ValueError(
            "Freespace-binary mode requires a semantic class named "
            "`road_ground` in config.yaml."
        ) from e


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--training-root",
        default="datasets/orfd/training",
        help="ORFD folder containing image_data/ and gt_image/",
    )
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--output-dir", default="reports/orfd_comparison")
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--models",
        nargs="+",
        default=("segformer-b2", "segformer-b4", "ddrnet"),
        help="Semantic model keys.",
    )
    p.add_argument(
        "--panel-w",
        type=int,
        default=320,
        help="Per-column width in render_panel strips.",
    )
    p.add_argument(
        "--single-mosaic",
        action="store_true",
        help="Also concatenate all strips vertically into orfd_mosaic.png",
    )
    p.add_argument(
        "--legacy-multiclass",
        action="store_true",
        help="Old four-way ORFD remap + multi-class overlays (debug only).",
    )
    p.add_argument(
        "--traversable-threshold",
        type=float,
        default=None,
        metavar="P",
        help=(
            "Override config orfd_semantic_comparison.freespace_merged_prob_floor "
            "when set: traversable where merged P(road_ground)≥P. Omit to use YAML."
        ),
    )
    p.add_argument(
        "--sanity-write-gt-raw",
        action="store_true",
        help="Dump up to 2 GT sanity composites (gray | colour legend).",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    freespace_binary = not args.legacy_multiclass

    cfg = load_config(args.config)
    occ = cfg.orfd_semantic_comparison
    trav_tau = (
        args.traversable_threshold
        if args.traversable_threshold is not None
        else occ.freespace_merged_prob_floor
    )
    if trav_tau is not None and not (0.0 < float(trav_tau) < 1.0):
        logger.error("traversable probability floor must lie in (0, 1); check CLI or YAML.")
        return 2
    sem_classes = list(cfg.semantic_classes)
    if not sem_classes:
        logger.error("config has no semantic classes")
        return 2

    class_names = tuple(c.name for c in sem_classes)
    n_uc = len(sem_classes)
    hw = cfg.hardware

    rg_idx: int | None = None
    if freespace_binary:
        rg_idx = _road_ground_channel_index(class_names)

    pairs = gather_orfd_pairs(Path(args.training_root).resolve())
    if not pairs:
        return 2

    rng = random.Random(args.seed)
    picks = rng.sample(pairs, k=min(args.samples, len(pairs)))
    stem_to_gt: dict[str, Path] = {im.stem: gtp for im, gtp in pairs}

    out_dir = Path(args.output_dir)
    strips_dir = out_dir / "strips"
    out_dir.mkdir(parents=True, exist_ok=True)
    strips_dir.mkdir(parents=True, exist_ok=True)

    otg_kw = {"orfd_trav_gray": int(occ.orfd_trav_gray)}

    frames_freespace: list[FreespaceFrame] = []
    frames_legacy: list[tuple[Path, np.ndarray, np.ndarray]] = []

    for img_path, gt_path in picks:
        img_bgr = cv2.imread(str(img_path))
        gt_raw = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
        if img_bgr is None or gt_raw is None:
            logger.warning("Read failed %s / %s; skip.", img_path, gt_path)
            continue

        if gt_raw.ndim == 3:
            gt_gray = gt_raw[..., 0].astype(np.uint8, copy=False)
        else:
            gt_gray = gt_raw.astype(np.uint8, copy=False)

        if gt_gray.shape[:2] != img_bgr.shape[:2]:
            gt_gray = cv2.resize(
                gt_gray,
                (img_bgr.shape[1], img_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        if freespace_binary:
            gt_tr = orfd_traversable_gt_mask(gt_gray, **otg_kw)
            vm = orfd_eval_valid_mask(gt_gray, **otg_kw)
            frames_freespace.append(
                FreespaceFrame(
                    strip_name=_fs_name_key("orfd", img_path.stem),
                    img_path=img_path,
                    img_bgr=img_bgr,
                    gt_trav=gt_tr,
                    valid=vm,
                ),
            )
        else:
            gu = orfd_gt_to_user_indices(gt_gray, class_names)
            frames_legacy.append((img_path, img_bgr, gu))

    if freespace_binary and occ.goose.samples > 0:
        g_root = Path(occ.goose.ex_root).resolve()
        csv_p = Path(occ.goose.label_csv)
        if not csv_p.is_file():
            logger.warning("GOOSE CSV missing (%s); skipping extra samples.", csv_p)
        else:
            csm_mod = _compare_semantic_module()
            g_names = csm_mod.GOOSE_CATEGORY_NAMES
            fine_lut = csm_mod.build_fine_to_category_lut(csv_p.resolve())
            trav_idx = goose_trav_category_indices(
                g_names,
                occ.goose.traversable_categories,
            )
            sky_ix = g_names.index("sky")
            void_ix = g_names.index("void")
            gpairs = gather_goose_scenario_pairs(g_root, occ.goose.scenario_dir)
            if gpairs:
                gs = min(occ.goose.samples, len(gpairs))
                goose_picks = rng.sample(gpairs, k=gs)
                pref = f"goose_{occ.goose.scenario_dir}"
                for img_path, lbl_path in goose_picks:
                    ib = cv2.imread(str(img_path))
                    lbl_raw = cv2.imread(str(lbl_path), cv2.IMREAD_UNCHANGED)
                    if ib is None or lbl_raw is None:
                        logger.warning(
                            "GOOSE read failed %s / %s; skip.",
                            img_path,
                            lbl_path,
                        )
                        continue
                    if lbl_raw.ndim == 3:
                        lbl_u = lbl_raw[..., 0].astype(np.uint8, copy=False)
                    else:
                        lbl_u = lbl_raw.astype(np.uint8, copy=False)
                    if lbl_u.shape[:2] != ib.shape[:2]:
                        lbl_u = cv2.resize(
                            lbl_u,
                            (ib.shape[1], ib.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    gt_tr_g, vm_g = goose_freespace_masks(
                        lbl_u,
                        fine_lut,
                        trav_cat_idxs=trav_idx,
                        sky_cat_idx=sky_ix,
                        void_cat_idx=void_ix,
                    )
                    frames_freespace.append(
                        FreespaceFrame(
                            strip_name=_fs_name_key(pref, img_path.stem),
                            img_path=img_path,
                            img_bgr=ib,
                            gt_trav=gt_tr_g,
                            valid=vm_g,
                        ),
                    )
            else:
                logger.warning(
                    "No GOOSE image/label pairs under scenario %r.",
                    occ.goose.scenario_dir,
                )
    elif occ.goose.samples > 0 and not freespace_binary:
        logger.warning(
            "orfd_semantic_comparison.goose.samples=%d ignored (--legacy-multiclass).",
            occ.goose.samples,
        )

    primary = frames_freespace if freespace_binary else frames_legacy

    if not primary:
        logger.error("No usable frames.")
        return 2

    if args.sanity_write_gt_raw:
        sdir = out_dir / "sanity_gt"
        sdir.mkdir(parents=True, exist_ok=True)
        if freespace_binary:
            orfd_vis = [
                fr
                for fr in frames_freespace
                if fr.strip_name.startswith("orfd_")
            ][:2]
            for fr in orfd_vis:
                gtp = stem_to_gt.get(fr.img_path.stem)
                if gtp is None:
                    continue
                raw = cv2.imread(str(gtp), cv2.IMREAD_UNCHANGED)
                if raw is None:
                    continue
                gg = raw[..., 0] if raw.ndim == 3 else raw
                out_s = sdir / f"gt_sanity_{fr.img_path.stem}.png"
                cv2.imwrite(str(out_s), gt_sanity_panel(np.asarray(gg, dtype=np.uint8)))
                logger.info("wrote sanity %s", out_s)
        else:
            for img_path, _, _gt in frames_legacy[:2]:
                gg_path = stem_to_gt.get(img_path.stem)
                raw = (
                    cv2.imread(str(gg_path), cv2.IMREAD_UNCHANGED) if gg_path else None
                )
                if raw is None:
                    continue
                gg = raw[..., 0] if raw.ndim == 3 else raw
                out_s = sdir / f"gt_sanity_{img_path.stem}.png"
                cv2.imwrite(str(out_s), gt_sanity_panel(np.asarray(gg, dtype=np.uint8)))
                logger.info("wrote sanity %s", out_s)

    backend = PyTorchBackend()

    subtract_model: InstanceModel | None = None
    im_sub = occ.instance_mask_subtraction
    dilate_px = max(0, int(im_sub.dilate_px))
    if freespace_binary and im_sub.subtract_from_traversable:
        try:
            subtract_model = build_instance_model(cfg.models.instance, hw, backend)
            subtract_model.warmup(cfg.classes)
            logger.info(
                "YOLOE mask subtraction enabled (dilate_px=%d).",
                dilate_px,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Instance model load/warmup failed (%s); continuing without subtraction.",
                e,
            )
            subtract_model = None

    default_sem_name = cfg.models.semantic.name.lower().strip()
    yaml_weights_override = cfg.models.semantic.weights or ""

    def _weights_for_key(key: str) -> str:
        lk = key.lower().strip()
        merged_w = SEMANTIC_DEFAULT_WEIGHTS.get(lk, "")
        use_yaml = lk == default_sem_name and bool(yaml_weights_override)
        return yaml_weights_override if use_yaml else merged_w

    def _ddrnet_ckpt_exists(w_str: str) -> bool:
        if not w_str.strip():
            return False
        pt = Path(w_str)
        return pt.is_file() if pt.is_absolute() else (_REPO / pt).is_file()

    model_candidates: list[str] = []
    for key in args.models:
        lk = key.lower().strip()
        w_str = _weights_for_key(key)
        if lk == "ddrnet" and w_str and not _ddrnet_ckpt_exists(w_str):
            logger.warning("DDRNet weights missing (%r); skipping.", w_str)
            continue
        model_candidates.append(key)

    pred_store: dict[str, dict[str, np.ndarray]] = {}
    sums: dict[str, list[float]] = {}

    for sem_key in model_candidates:
        w_str = _weights_for_key(sem_key)
        try:
            mdl = build_semantic_model(
                SemanticModelCfg(name=sem_key, weights=w_str), hw, backend,
            )
            mdl.warmup(cfg.classes)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load %s: %s — skipping.", sem_key, e)
            continue

        sums[sem_key] = []
        ran_frames = False

        assert rg_idx is not None or not freespace_binary

        if freespace_binary:
            for fr in frames_freespace:
                fk = fr.strip_name
                merged = mdl.predict_logits(fr.img_bgr)
                mnp = merged.float().cpu().numpy()
                if trav_tau is None:
                    pred_mc = mnp.argmax(axis=0).astype(np.int32, copy=False)
                    pred_bool = pred_mc == int(rg_idx)
                else:
                    pred_bool = mnp[int(rg_idx)] >= float(trav_tau)
                if subtract_model is not None:
                    pred_bool = subtract_instance_masks_from_traversable(
                        pred_bool,
                        fr.img_bgr,
                        subtract_model,
                        dilate_px=dilate_px,
                    )
                pred_vis = np.where(pred_bool, np.int8(0), np.int8(-1))

                iou_bin = binary_traversable_iou(
                    pred_bool, fr.gt_trav, fr.valid,
                )
                if iou_bin is not None:
                    sums[sem_key].append(iou_bin)

                pred_store.setdefault(fk, {})[sem_key] = pred_vis
                ran_frames = True

                del merged, mnp
        else:
            for img_path, img_bgr, gt_user in frames_legacy:
                stem = img_path.stem
                merged = mdl.predict_logits(img_bgr)
                pred_mc = merged.argmax(dim=0).cpu().numpy().astype(np.int8)

                mu = per_image_mean_iou(pred_mc, gt_user, n_uc)
                if mu is not None:
                    sums[sem_key].append(mu)

                pred_store.setdefault(stem, {})[sem_key] = pred_mc
                ran_frames = True

                del merged

        del mdl
        if hw.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not ran_frames:
            sums.pop(sem_key, None)

    framing_keys_ok = (
        [fr.strip_name for fr in frames_freespace]
        if freespace_binary
        else [ip.stem for ip, _, __ in frames_legacy]
    )
    models_ran = [
        k
        for k in sums
        if framing_keys_ok
        and all(k in pred_store.get(strip_k, {}) for strip_k in framing_keys_ok)
    ]

    if not models_ran:
        logger.error("No model produced predictions.")
        return 2

    mosaic_rows: list[np.ndarray] = []

    for trip in primary:
        if freespace_binary:
            fr = trip  # type: ignore[assignment]
            img_path = fr.img_path
            img_bgr = fr.img_bgr
            strip_key = fr.strip_name
            preds_vis = pred_store.get(strip_key, {})

            gt_tr = fr.gt_trav
            gt_vis = np.where(gt_tr, 0, -1).astype(np.int8)

            if fr.strip_name.startswith("goose_"):
                subtitle = (
                    "binary GOOSE GT ("
                    + ",".join(occ.goose.traversable_categories)
                    + "; sky|void excl)"
                )
            else:
                subtitle = (
                    f"binary traversable (ORFD path gray={occ.orfd_trav_gray})"
                )

            chunks = []
            for kname in models_ran:
                pr_full = preds_vis.get(kname)
                if pr_full is None:
                    chunks.append(f"{kname} IoU=—")
                    continue
                iou_bb = binary_traversable_iou(
                    pr_full == 0,
                    gt_tr,
                    fr.valid,
                )
                chunks.append(
                    f"{kname} trav_IoU={'%.3f' % iou_bb if iou_bb is not None else '—'}",
                )

            title = img_path.parent.name + "/" + img_path.name
            pane = strips_dir / f"_strip_{strip_key}.png"
            render_panel(
                title=title + " | " + subtitle,
                image_bgr=img_bgr,
                gt_userclass=gt_vis,
                preds={k: preds_vis[k] for k in models_ran if k in preds_vis},
                user_classes=list(_TRAV_CLASSES),
                out_path=pane,
                target_w=args.panel_w,
            )
            strip = cv2.imread(str(pane))
            pane.unlink(missing_ok=True)
            out_key = strip_key
        else:
            img_path, img_bgr, third = trip  # type: ignore[misc]
            stem = img_path.stem
            preds_vis = pred_store.get(stem, {})
            gt_user = third

            chunks = []
            for kname in models_ran:
                pr_full = preds_vis.get(kname)
                iou_bb = (
                    per_image_mean_iou(pr_full, gt_user, n_uc)
                    if pr_full is not None
                    else None
                )
                chunks.append(
                    f"{kname} IoU={'%.3f' % iou_bb if iou_bb is not None else '—'}",
                )

            title = img_path.parent.name + "/" + img_path.name
            pane = strips_dir / f"_strip_{stem}.png"
            render_panel(
                title=title,
                image_bgr=img_bgr,
                gt_userclass=gt_user,
                preds={k: preds_vis[k] for k in models_ran if k in preds_vis},
                user_classes=sem_classes,
                out_path=pane,
                target_w=args.panel_w,
            )
            strip = cv2.imread(str(pane))
            pane.unlink(missing_ok=True)
            out_key = stem

        if strip is None:
            continue

        metrics_line = " | ".join(chunks if chunks else [])
        banded = strip_row_band_with_metrics(strip, metrics_line)
        out_png = strips_dir / f"{out_key}.png"
        cv2.imwrite(str(out_png), banded)
        logger.info("wrote %s", out_png)
        mosaic_rows.append(banded)

    if not mosaic_rows:
        logger.error("No PNG strips generated.")
        return 2

    mean_bits = []
    for k in models_ran:
        vals = sums.get(k, [])
        mean_bits.append(
            f"{k}: mean={'%.3f' % float(np.mean(vals))}" if vals else f"{k}: mean=—"
        )

    readme_lines = [
        "Generated by scripts/orfd_semantic_comparison.py",
        f"training_root={Path(args.training_root).resolve()}",
        f"orfd_trav_gray={occ.orfd_trav_gray}",
        f"goose.samples={occ.goose.samples} scenario={occ.goose.scenario_dir} "
        f"trav_categories={list(occ.goose.traversable_categories)}",
        (
            "instance_mask_subtraction="
            f"{im_sub.subtract_from_traversable}"
            + (
                f" dilate_px={im_sub.dilate_px}"
                if im_sub.subtract_from_traversable
                else ""
            )
        ),
        f"freespace_binary={freespace_binary} traversable_prob_floor={trav_tau}",
        f"mean metrics: {'; '.join(mean_bits)}",
        "",
        "ORFD fillcolor: https://github.com/chaytonmin/Off-Road-Freespace-Detection",
        "Individual PNG strips under: strips/",
    ]
    (out_dir / "README.txt").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    if args.single_mosaic:
        core = assemble_mosaic_vertical(mosaic_rows)
        mw = core.shape[1]
        head_h = 34
        head = np.zeros((head_h, mw, 3), dtype=np.uint8)
        hl = "; ".join(mean_bits)
        cv2.putText(head, hl[:118], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (248, 248, 240), 1, cv2.LINE_AA)
        fh = footer_freespace_strip(mw, 96) if freespace_binary else footer_legend_legacy(
            mw, 110, user_classes=sem_classes, gray_to_name=ORFD_GRAY_TO_CLASS
        )

        mosaic_full = np.vstack([head, core, fh])
        mop = out_dir / "orfd_mosaic.png"
        cv2.imwrite(str(mop), mosaic_full)
        logger.info("wrote %s", mop)

    if hw.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

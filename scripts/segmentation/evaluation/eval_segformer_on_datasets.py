#!/usr/bin/env python3
"""Qualitative SegFormer-B2 eval on RUGD / ORFD off-road imagery.

Loads the project config as-is, builds *only* the semantic head (no
YOLOE / no full pipeline), samples a deterministic subset of frames
from each dataset under ``datasets/`` and writes side-by-side
visualisations and an index.md to ``outputs/segformer_eval/``.

Each panel shows:
    1. the original frame
    2. the user-class colour overlay at alpha 0.5 (classes whose
       ``display_mode == "none"`` — e.g. ``sky`` — are absorbed but
       not drawn, matching the production renderer's behaviour)
    3. a text panel with the per-user-class pixel-coverage breakdown
       and the top-5 native ADE20K classes the raw model picked
       *before* the user-class merge

Coverage numbers and native top-K stats are aggregated into
``index.md`` so the breakdowns are inspectable without opening the
PNGs.

Usage
-----

    source .venv/bin/activate
    PYTHONPATH=src python scripts/eval_segformer_on_datasets.py --samples 24
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str((_HERE.parents[2] / "src").resolve()))

from perception.config.loader import load_config  # noqa: E402
from perception.config.schema import ClassDef  # noqa: E402
from perception.models.backends.factory import build_backend  # noqa: E402
from perception.models.factory import build_semantic_model  # noqa: E402
from perception.models.semantic.segformer import SegFormerSemanticModel  # noqa: E402

logger = logging.getLogger("eval_segformer")

IMG_EXTS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


# --------------------------------------------------------------------------- #
# Image walking & sampling                                                    #
# --------------------------------------------------------------------------- #


def _gather_images(root: Path, exts: Sequence[str] = IMG_EXTS) -> list[Path]:
    """Recursively collect image paths under ``root``, sorted by name.

    Returns ``[]`` if ``root`` does not exist or contains no images,
    so the caller can cleanly skip a missing dataset.
    """
    if not root.is_dir():
        return []
    files: list[Path] = []
    for ext in exts:
        files.extend(root.rglob(f"*{ext}"))
    # Skip the .hf-cache duplicates created by the RUGD downloader.
    files = [f for f in files if ".hf-cache" not in f.parts]
    files.sort()
    return files


def _sample(images: Sequence[Path], n: int, seed: int = 42) -> list[Path]:
    if not images:
        return []
    rng = random.Random(seed)
    n = min(n, len(images))
    return rng.sample(list(images), n)


# --------------------------------------------------------------------------- #
# Inference helpers                                                           #
# --------------------------------------------------------------------------- #


def _native_argmax_counts(
    sem: SegFormerSemanticModel,
    frame_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a *separate* low-res forward pass and return (counts, frac).

    Operates on the raw 150-channel ADE20K argmax so we can report
    "what the model actually saw before the user-class merge". We do
    this at the model's native low-res output (H/4, W/4) rather than
    upsampling, because:
        * the per-pixel ADE20K class distribution is well-approximated
          at low res for argmax counting;
        * upsampling 150 channels to ~700 px tall RUGD frames burns
          memory and adds nothing to the qualitative interpretation.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inputs = sem._processor(images=rgb, return_tensors="pt")  # noqa: SLF001
    pixel_values = inputs["pixel_values"].to(sem._device)  # noqa: SLF001
    if sem._fp16:  # noqa: SLF001
        pixel_values = pixel_values.half()

    with torch.inference_mode():
        outputs = sem._model(pixel_values=pixel_values)  # noqa: SLF001
    logits = outputs.logits[0]  # (150, h/4, w/4)
    arg = torch.argmax(logits, dim=0).flatten().to(torch.int64).cpu().numpy()
    n = arg.size
    n_classes = int(logits.shape[0])
    counts = np.bincount(arg, minlength=n_classes)
    frac = counts.astype(np.float64) / max(n, 1)
    return counts, frac


def _ade20k_id2label(sem: SegFormerSemanticModel) -> dict[int, str]:
    """Return ``{int_id: label}`` for the 150 ADE20K classes."""
    raw = getattr(sem._model.config, "id2label", None)  # noqa: SLF001
    out: dict[int, str] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                out[int(k)] = str(v)
            except (TypeError, ValueError):
                continue
    return out


# --------------------------------------------------------------------------- #
# Visualisation                                                               #
# --------------------------------------------------------------------------- #


def _build_overlay(
    frame_bgr: np.ndarray,
    user_argmax: np.ndarray,
    sem_classes: Sequence[ClassDef],
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend a colour mask onto ``frame_bgr``.

    Classes whose ``display_mode == "none"`` are *not* drawn, but they
    still consume their pixels in the argmax — so e.g. sky pixels stay
    transparent and show through to the original image.
    """
    h, w = frame_bgr.shape[:2]
    color = np.zeros_like(frame_bgr)
    drawn_mask = np.zeros((h, w), dtype=bool)

    for j, c in enumerate(sem_classes):
        if c.display_mode == "none":
            continue
        cls_mask = user_argmax == j
        if not cls_mask.any():
            continue
        # ClassDef colours are RGB; OpenCV is BGR.
        bgr = (int(c.color_rgb[2]), int(c.color_rgb[1]), int(c.color_rgb[0]))
        color[cls_mask] = bgr
        drawn_mask |= cls_mask

    blended = frame_bgr.copy()
    if drawn_mask.any():
        # Only blend over pixels that have a drawn class — keeps "none"
        # regions (e.g. sky) untouched.
        idx = drawn_mask
        blended[idx] = (
            frame_bgr[idx].astype(np.float32) * (1.0 - alpha)
            + color[idx].astype(np.float32) * alpha
        ).clip(0, 255).astype(np.uint8)
    return blended


def _user_class_coverage(
    user_argmax: np.ndarray,
    sem_classes: Sequence[ClassDef],
) -> dict[str, float]:
    """Return ``{class_name: fraction in [0, 1]}``."""
    n = user_argmax.size
    counts = np.bincount(user_argmax.ravel(), minlength=len(sem_classes))
    return {c.name: float(counts[j]) / max(n, 1) for j, c in enumerate(sem_classes)}


def _format_user_breakdown(coverage: dict[str, float]) -> str:
    items = sorted(coverage.items(), key=lambda kv: kv[1], reverse=True)
    return ", ".join(f"{n} {p * 100:.1f}%" for n, p in items)


def _format_native_top(
    frac: np.ndarray, id2label: dict[int, str], k: int = 5
) -> list[tuple[int, str, float]]:
    order = np.argsort(-frac)[:k]
    return [
        (int(i), id2label.get(int(i), f"class_{int(i)}"), float(frac[int(i)]))
        for i in order
    ]


def _wrap_text(s: str, width: int = 64) -> list[str]:
    out: list[str] = []
    for raw_line in s.split("\n"):
        cur: list[str] = []
        cur_len = 0
        for tok in raw_line.split(", "):
            if cur_len and cur_len + len(tok) + 2 > width:
                out.append(", ".join(cur))
                cur = [tok]
                cur_len = len(tok)
            else:
                cur.append(tok)
                cur_len += len(tok) + 2
        if cur:
            out.append(", ".join(cur))
    return out


def _make_text_panel(
    width: int,
    height: int,
    title: str,
    body_lines: Iterable[str],
) -> np.ndarray:
    panel = np.full((height, width, 3), 24, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 0.65
    body_scale = 0.5
    title_thick = 2
    body_thick = 1
    margin_x = 14
    y = 30

    cv2.putText(
        panel, title, (margin_x, y), font, title_scale, (255, 255, 255), title_thick,
        cv2.LINE_AA,
    )
    y += 24
    cv2.line(panel, (margin_x, y), (width - margin_x, y), (90, 90, 90), 1)
    y += 18
    for line in body_lines:
        if y > height - 8:
            break
        cv2.putText(
            panel, line, (margin_x, y), font, body_scale, (220, 220, 220), body_thick,
            cv2.LINE_AA,
        )
        y += 18
    return panel


def _compose_panel(
    frame_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    title: str,
    body_lines: Sequence[str],
) -> np.ndarray:
    """[original | overlay | text] horizontally."""
    h, w = frame_bgr.shape[:2]
    text_w = max(420, w // 2)
    text_panel = _make_text_panel(text_w, h, title, body_lines)
    return np.hstack([frame_bgr, overlay_bgr, text_panel])


# --------------------------------------------------------------------------- #
# Per-image driver                                                            #
# --------------------------------------------------------------------------- #


def _evaluate_one(
    sem: SegFormerSemanticModel,
    sem_classes: Sequence[ClassDef],
    img_path: Path,
    id2label: dict[int, str],
) -> dict | None:
    frame = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if frame is None:
        logger.warning("Could not read %s; skipping.", img_path)
        return None

    t0 = time.perf_counter()
    merged = sem.predict_logits(frame)  # (C_user, H, W)
    user_argmax = torch.argmax(merged, dim=0).to(torch.int64).cpu().numpy()
    t1 = time.perf_counter()

    native_counts, native_frac = _native_argmax_counts(sem, frame)
    t2 = time.perf_counter()

    coverage = _user_class_coverage(user_argmax, sem_classes)
    native_top5 = _format_native_top(native_frac, id2label, k=5)

    overlay = _build_overlay(frame, user_argmax, sem_classes, alpha=0.5)
    title = img_path.name
    body = [
        "User-class coverage:",
        *_wrap_text(_format_user_breakdown(coverage), width=58),
        "",
        "Native ADE20K top-5 (pre-merge):",
        *[f"  #{i:3d} {name:<22s} {p * 100:5.1f}%" for i, name, p in native_top5],
        "",
        f"Inference: predict={1000 * (t1 - t0):.1f} ms,",
        f"           native={1000 * (t2 - t1):.1f} ms",
        f"Frame: {frame.shape[1]}x{frame.shape[0]}",
    ]
    panel = _compose_panel(frame, overlay, title, body)

    return {
        "path": img_path,
        "panel": panel,
        "coverage": coverage,
        "native_counts": native_counts,
        "predict_ms": 1000 * (t1 - t0),
        "native_ms": 1000 * (t2 - t1),
    }


# --------------------------------------------------------------------------- #
# Top-level driver                                                            #
# --------------------------------------------------------------------------- #


def _aggregate_native(
    counts_list: Sequence[np.ndarray], k: int, id2label: dict[int, str]
) -> list[tuple[int, str, float]]:
    if not counts_list:
        return []
    stack = np.stack(counts_list, axis=0).astype(np.float64)
    per_image_frac = stack / np.clip(stack.sum(axis=1, keepdims=True), 1, None)
    mean_frac = per_image_frac.mean(axis=0)
    order = np.argsort(-mean_frac)[:k]
    return [
        (int(i), id2label.get(int(i), f"class_{int(i)}"), float(mean_frac[int(i)]))
        for i in order
    ]


def _aggregate_user(
    cov_list: Sequence[dict[str, float]], sem_classes: Sequence[ClassDef]
) -> dict[str, float]:
    out: dict[str, float] = {c.name: 0.0 for c in sem_classes}
    if not cov_list:
        return out
    for cov in cov_list:
        for k, v in cov.items():
            out[k] = out.get(k, 0.0) + v
    return {k: v / len(cov_list) for k, v in out.items()}


def _write_index(
    out_root: Path,
    per_dataset: dict[str, dict],
    sem_classes: Sequence[ClassDef],
) -> Path:
    lines: list[str] = []
    lines.append("# SegFormer-B2 qualitative eval")
    lines.append("")
    lines.append(
        "Generated by `scripts/eval_segformer_on_datasets.py`. Each row "
        "links to a side-by-side panel "
        "(original / colour overlay / text). The user-class set is "
        "merged from ADE20K via `config/config.yaml` "
        f"(`{', '.join(c.name for c in sem_classes)}`)."
    )
    lines.append("")

    for ds, info in per_dataset.items():
        n = info["n"]
        cov = info["avg_coverage"]
        nat = info["native_top10"]
        records = info["records"]
        lines.append(f"## {ds}")
        lines.append("")
        lines.append(f"- Source dir: `{info['root']}`")
        lines.append(f"- Available frames: {info['available']}")
        lines.append(f"- Sampled: {n}")
        if info.get("median_predict_ms") is not None:
            lines.append(
                f"- Median per-frame inference: predict {info['median_predict_ms']:.1f} ms, "
                f"native {info['median_native_ms']:.1f} ms"
            )
        lines.append("")
        lines.append("### Average user-class pixel coverage")
        lines.append("")
        lines.append("| class | mean coverage |")
        lines.append("|---|---|")
        for c in sem_classes:
            lines.append(f"| `{c.name}` | {cov.get(c.name, 0.0) * 100:.2f}% |")
        lines.append("")
        lines.append("### Top-10 native ADE20K classes (mean over samples)")
        lines.append("")
        lines.append("| ADE20K id | name | mean fraction |")
        lines.append("|---|---|---|")
        for i, name, p in nat:
            lines.append(f"| {i} | `{name}` | {p * 100:.2f}% |")
        lines.append("")
        if records:
            lines.append("### Per-image panels")
            lines.append("")
            for r in records:
                rel = r["panel_path"].relative_to(out_root)
                lines.append(f"- [{rel}]({rel.as_posix()})")
            lines.append("")
        else:
            lines.append("_No samples produced._")
            lines.append("")

    idx = out_root / "index.md"
    idx.write_text("\n".join(lines))
    return idx


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--datasets-root", default="./datasets")
    p.add_argument("--out", default="./outputs/segformer_eval")
    p.add_argument("--samples", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--datasets",
        nargs="+",
        default=("rugd", "orfd"),
        help="Per-dataset subdirs to scan under --datasets-root.",
    )
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    sem_classes = list(cfg.semantic_classes)
    if not sem_classes:
        logger.error("No semantic classes configured; nothing to evaluate.")
        return 2

    backend = build_backend(cfg.hardware.use_tensorrt)
    sem = build_semantic_model(cfg.models.semantic, cfg.hardware, backend)
    if not isinstance(sem, SegFormerSemanticModel):
        logger.error(
            "This script currently understands SegFormer only "
            "(got %s).", type(sem).__name__,
        )
        return 2
    sem.warmup(cfg.classes)
    id2label = _ade20k_id2label(sem)
    logger.info(
        "Loaded SegFormer with %d native classes; %d user classes: %s",
        len(id2label), len(sem_classes), [c.name for c in sem_classes],
    )

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    per_dataset: dict[str, dict] = {}

    for ds in args.datasets:
        ds_root = Path(args.datasets_root) / ds
        images = _gather_images(ds_root)
        sampled = _sample(images, args.samples, args.seed)
        ds_out = out_root / ds
        ds_out.mkdir(parents=True, exist_ok=True)

        records: list[dict] = []
        cov_list: list[dict[str, float]] = []
        nat_counts_list: list[np.ndarray] = []
        predict_ms: list[float] = []
        native_ms: list[float] = []

        if not sampled:
            logger.warning(
                "Dataset %r: no images found under %s; skipping.", ds, ds_root,
            )
        for i, p in enumerate(sampled, 1):
            r = _evaluate_one(sem, sem_classes, p, id2label)
            if r is None:
                continue
            panel_path = ds_out / f"{p.stem}.png"
            cv2.imwrite(str(panel_path), r["panel"])
            r["panel_path"] = panel_path
            records.append(r)
            cov_list.append(r["coverage"])
            nat_counts_list.append(r["native_counts"])
            predict_ms.append(r["predict_ms"])
            native_ms.append(r["native_ms"])
            logger.info(
                "[%s %d/%d] %s -> %s",
                ds, i, len(sampled), p.name, panel_path.name,
            )

        avg_cov = _aggregate_user(cov_list, sem_classes)
        nat_top10 = _aggregate_native(nat_counts_list, k=10, id2label=id2label)
        per_dataset[ds] = {
            "root": str(ds_root),
            "available": len(images),
            "n": len(records),
            "records": records,
            "avg_coverage": avg_cov,
            "native_top10": nat_top10,
            "median_predict_ms": float(np.median(predict_ms)) if predict_ms else None,
            "median_native_ms": float(np.median(native_ms)) if native_ms else None,
        }

    idx = _write_index(out_root, per_dataset, sem_classes)
    logger.info("Wrote index: %s", idx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

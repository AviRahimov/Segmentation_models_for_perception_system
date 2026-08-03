"""Shared helpers for the segmentation evaluation scripts: panel rendering,
latency measurement, interactive dataset/checkpoint picking.

Extracted from the old GOOSE-Ex-only ``compare_semantic_models.py`` (render_panel,
measure_forward_latency_ms — unchanged) plus new dataset/checkpoint scanning that
mirrors ``scripts/detection/training/_survey_common.py``'s interactive-pick pattern,
adapted for segmentation's heterogeneous per-dataset GT formats (unlike detection's
uniform YOLO ``data.yaml``, there's no single manifest convention here).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from perception.config.schema import ClassDef
from perception.models.semantic.base import SemanticModel

logger = logging.getLogger("semantic_eval_common")


# --------------------------------------------------------------------------- #
# Panel rendering (moved verbatim from the old GOOSE-only compare_semantic_models.py)
# --------------------------------------------------------------------------- #


def render_panel(
    *,
    title: str,
    image_bgr: np.ndarray,
    gt_userclass: np.ndarray | None,
    preds: dict[str, np.ndarray],   # name -> (H, W) int8 user-class index, -1 = unassigned
    user_classes: list[ClassDef],
    out_path: Path,
    target_w: int = 480,
) -> None:
    """Compose a (input | model_1 | model_2 | ... | GT) horizontal strip.

    ``gt_userclass`` may be ``None`` for a qualitative-only (no ground truth)
    dataset — the GT column is simply omitted.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    H_in, W_in = image_bgr.shape[:2]
    th = int(target_w * H_in / W_in)
    img_resized = cv2.resize(image_bgr, (target_w, th), interpolation=cv2.INTER_AREA)

    panels: list[tuple[str, np.ndarray]] = []
    panels.append(("input", img_resized))

    palette = np.zeros((len(user_classes) + 1, 3), dtype=np.uint8)
    for i, c in enumerate(user_classes):
        # ClassDef.color_rgb is (R, G, B); cv2 wants BGR.
        palette[i] = c.color_rgb[::-1]
    palette[-1] = (0, 0, 0)  # unassigned

    def _colorise(seg: np.ndarray) -> np.ndarray:
        seg2 = seg.copy()
        seg2[seg2 < 0] = len(user_classes)  # last palette slot
        rgb = palette[seg2]
        return cv2.resize(rgb, (target_w, th), interpolation=cv2.INTER_NEAREST)

    for name, pred in preds.items():
        rendered = _colorise(pred)
        # Light overlay on input for readability.
        blend = cv2.addWeighted(img_resized, 0.4, rendered, 0.6, 0.0)
        panels.append((name, blend))

    if gt_userclass is not None:
        rendered = _colorise(gt_userclass)
        blend = cv2.addWeighted(img_resized, 0.4, rendered, 0.6, 0.0)
        panels.append(("GT", blend))

    # Draw text label per panel.
    labelled = []
    for name, panel in panels:
        canvas = panel.copy()
        cv2.rectangle(canvas, (0, 0), (target_w, 26), (0, 0, 0), -1)
        cv2.putText(canvas, name, (8, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        labelled.append(canvas)

    strip = np.concatenate(labelled, axis=1)

    # Title bar
    title_bar = np.zeros((28, strip.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_bar, title, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    out = np.concatenate([title_bar, strip], axis=0)
    cv2.imwrite(str(out_path), out)


# --------------------------------------------------------------------------- #
# Latency (moved verbatim)
# --------------------------------------------------------------------------- #


def measure_forward_latency_ms(
    model: SemanticModel,
    *,
    sample_frame: np.ndarray,
    n_warm: int = 20,
    n_iter: int = 100,
) -> float:
    """Return median forward-pass latency in ms over ``n_iter`` runs.

    Warmup iterations are excluded; CUDA events are used so the measurement
    captures GPU work only (not Python or CPU prep). The wrapper's
    ``predict_logits`` path includes preprocessing -- which is what the
    application actually measures, so we time the whole call here.
    """
    for _ in range(n_warm):
        model.predict_logits(sample_frame)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(n_iter):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model.predict_logits(sample_frame)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


# --------------------------------------------------------------------------- #
# Small interactive-prompt helpers (self-contained; mirrors the spirit of
# scripts/detection/training/_survey_common.py's _ask()/_confirm(), kept as
# its own copy so segmentation/detection tooling stay independently editable)
# --------------------------------------------------------------------------- #


def ask_choice(question: str, options: list[tuple[str, str]], default_idx: int = 0) -> str:
    """Print a numbered menu of (value, label) options; return the chosen value.

    Enter alone picks ``default_idx``. Non-interactive stdin (e.g. under a
    test runner) falls back to the default without blocking.
    """
    print(f"\n{question}")
    for i, (_, label) in enumerate(options):
        marker = " (default)" if i == default_idx else ""
        print(f"  [{i}] {label}{marker}")
    try:
        raw = input(f"Choice [0-{len(options) - 1}], Enter={default_idx}: ").strip()
    except (EOFError, OSError):
        raw = ""
    if not raw:
        return options[default_idx][0]
    try:
        idx = int(raw)
        if 0 <= idx < len(options):
            return options[idx][0]
    except ValueError:
        pass
    print(f"Unrecognised choice {raw!r}; using default.")
    return options[default_idx][0]


def ask_multi_choice(
    question: str, options: list[tuple[str, str]], default_idxs: tuple[int, ...],
) -> list[str]:
    """Same as ``ask_choice`` but accepts a comma-separated list of indices."""
    print(f"\n{question}")
    for i, (_, label) in enumerate(options):
        marker = " (default)" if i in default_idxs else ""
        print(f"  [{i}] {label}{marker}")
    default_str = ",".join(str(i) for i in default_idxs)
    try:
        raw = input(f"Choice(s), comma-separated [0-{len(options) - 1}], Enter={default_str}: ").strip()
    except (EOFError, OSError):
        raw = ""
    if not raw:
        return [options[i][0] for i in default_idxs]
    picked: list[str] = []
    for tok in raw.split(","):
        tok = tok.strip()
        try:
            idx = int(tok)
            if 0 <= idx < len(options):
                picked.append(options[idx][0])
        except ValueError:
            continue
    if not picked:
        print(f"Unrecognised choice {raw!r}; using default.")
        return [options[i][0] for i in default_idxs]
    return picked


# --------------------------------------------------------------------------- #
# Dataset scanning — segmentation datasets don't share one manifest format
# (unlike detection's YOLO data.yaml), so this classifies by known layout
# markers instead of parsing a uniform config file.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DatasetChoice:
    kind: str          # "orfd" | "zikim" | "fcdd"
    name: str          # directory name under datasets/segmentation/
    root: Path
    label: str         # human-readable, shown in the interactive menu


def scan_segmentation_datasets(root: Path) -> tuple[list[DatasetChoice], list[tuple[str, str]]]:
    """Classify each subdir of ``root`` by known segmentation-dataset layout.

    Returns ``(found, skipped)`` where ``skipped`` is ``[(dirname, reason)]``
    for anything that isn't a recognized image+GT (or image-only) layout —
    e.g. a folder of raw videos, which belongs to compare_on_raw_video.py
    instead.
    """
    found: list[DatasetChoice] = []
    skipped: list[tuple[str, str]] = []
    if not root.is_dir():
        return found, skipped

    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if (d / "training" / "image_data").is_dir() and (d / "training" / "gt_image").is_dir():
            found.append(DatasetChoice(
                kind="orfd", name=d.name, root=d,
                label=f"{d.name} (ORFD-style, labeled 3-class freespace)",
            ))
        elif (d / "config_zikim.json").is_file():
            found.append(DatasetChoice(
                kind="zikim", name=d.name, root=d,
                label=f"{d.name} (Cityscapes-style color GT, converted to 3-class)",
            ))
        elif (d / "_classes.csv").is_file() and any(
            (d / split / "im").is_dir() for split in ("train", "val", "test")
        ):
            found.append(DatasetChoice(
                kind="fcdd", name=d.name, root=d,
                label=f"{d.name} (has its own class scheme, no verified collapse "
                      f"mapping yet -- qualitative only, GT ignored)",
            ))
        else:
            skipped.append((d.name, "no recognized labeled/qualitative image-dataset layout"))

    return found, skipped


# --------------------------------------------------------------------------- #
# Checkpoint scanning — enumerate real files on disk under weights/segmentation
# so the interactive model picker offers actual trained recipes (frozen vs
# full-finetune vs LoRA vs distilled), not just one entry per architecture.
# --------------------------------------------------------------------------- #

#: Directory-name prefix -> factory key it should be loaded as.
_DIRNAME_TO_KEY: dict[str, str] = {
    "auriganet": "auriganet",
    "upernet": "upernet",
    "dinov2-large": "dinov2-large",
    "dinov2": "dinov2",
    "mask2former-large-lora": "mask2former-large",
    "mask2former-large": "mask2former-large",
    "mask2former": "mask2former-base",
    "distilled_segformer-b2": "segformer-b2",
}


@dataclass(frozen=True)
class CheckpointChoice:
    key: str        # factory key to build with
    weights: str    # local .pth path, or "" for the key's HF-hub default
    label: str      # human-readable recipe description


def _infer_key(rel_dir: str) -> str | None:
    first_seg = rel_dir.split("/")[0]
    if first_seg in _DIRNAME_TO_KEY:
        return _DIRNAME_TO_KEY[first_seg]
    # frozen_backbone/segformer-b2, lora/segformer-b2, full_finetune_heavy_aug/segformer-b4, ...
    parts = rel_dir.split("/")
    for p in parts:
        if p.startswith("segformer-b"):
            return p
    return None


def scan_semantic_checkpoints(weights_root: Path) -> list[CheckpointChoice]:
    """Walk ``weights_root/**/best.pth`` and infer a usable factory key for each."""
    choices: list[CheckpointChoice] = []
    if not weights_root.is_dir():
        return choices
    for ckpt in sorted(weights_root.rglob("best.pth")):
        rel_dir = str(ckpt.parent.relative_to(weights_root))
        key = _infer_key(rel_dir)
        if key is None:
            logger.debug("Skipping %s: no known factory key for this layout.", ckpt)
            continue
        choices.append(CheckpointChoice(key=key, weights=str(ckpt), label=rel_dir))
    return choices


#: Always-offered HF-hub ADE20K baselines (no ORFD fine-tuning) — useful as a
#: "what if we never fine-tuned" reference point in the interactive picker.
BASELINE_CHOICES: tuple[CheckpointChoice, ...] = tuple(
    CheckpointChoice(key=k, weights="", label=f"{k} (ADE20K baseline, not fine-tuned)")
    for k in ("segformer-b0", "segformer-b1", "segformer-b2", "segformer-b4")
)


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Split a ``--models`` entry: ``"key"`` or ``"key:weights_path"``."""
    if ":" in spec:
        key, _, weights = spec.partition(":")
        return key.strip(), weights.strip()
    return spec.strip(), ""


def resolve_weights(key: str, explicit_weights: str, cfg) -> str:
    """Weights for a factory key: explicit override > config.yaml's active
    model override (if this key is the currently-configured one) > the
    factory's own default (HF-hub baseline or "")."""
    from perception.models.factory import SEMANTIC_DEFAULT_WEIGHTS

    if explicit_weights:
        return explicit_weights
    lk = key.lower().strip()
    if lk == cfg.models.semantic.name.lower().strip() and cfg.models.semantic.weights:
        return cfg.models.semantic.weights
    return SEMANTIC_DEFAULT_WEIGHTS.get(lk, "")

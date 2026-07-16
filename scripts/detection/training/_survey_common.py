"""Shared, architecture-agnostic helpers for the detection training surveys.

Used by both ``train_detector.py`` (YOLO family) and
``train_detector_rfdetr.py`` (RF-DETR family) — dataset scanning, the
numbered-prompt UX, checkpoint discovery, and experiment-log provenance are
identical regardless of which trainer actually runs. Anything that depends
on a specific training stack (recipes, freeze schedules, the Ultralytics
progress-bar callback wiring, RF-DETR's ``TrainConfig``) stays in its own
script.
"""
from __future__ import annotations

import json
import logging
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

_ROOT = Path(__file__).resolve().parents[3]

logger = logging.getLogger("survey_common")

_IMG_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================================== #
# Dataset scanning                                                            #
# =========================================================================== #

@dataclass
class _DatasetInfo:
    name: str                     # directory name, e.g. "yolo_dataset_auto_labeled"
    dir: Path
    yaml_path: Path               # original yaml
    n_train: int
    n_val: int
    class_names: list[str]
    needs_local_yaml: bool        # original yaml has a broken absolute path:
    train_rel: str
    val_rel: str


def _count_images(d: Path) -> int:
    if not d.is_dir():
        return 0
    return sum(1 for p in d.iterdir() if p.suffix.lower() in _IMG_SUFFIXES)


def _resolve_split_dir(dataset_dir: Path, rel: str) -> Path | None:
    """Resolve a train/val entry against the dataset dir, tolerating the
    Roboflow ``../`` convention (same trick as compare_detection_models.py)."""
    cand = (dataset_dir / rel).resolve()
    if cand.is_dir():
        return cand
    stripped = rel
    while stripped.startswith("../"):
        stripped = stripped[3:]
    cand = (dataset_dir / stripped).resolve()
    return cand if cand.is_dir() else None


def _scan_datasets(root: Path) -> tuple[list[_DatasetInfo], list[tuple[str, str]]]:
    """Return (trainable datasets sorted by train size desc, skipped [(name, reason)])."""
    found: list[_DatasetInfo] = []
    skipped: list[tuple[str, str]] = []

    for ds_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        yamls = sorted(ds_dir.glob("*.yaml")) + sorted(ds_dir.glob("*.yml"))
        # Ignore local yamls we generated ourselves — the original is scanned.
        yamls = [y for y in yamls if y.name != "data.local.yaml"]
        if not yamls:
            skipped.append((ds_dir.name, "no dataset yaml"))
            continue

        info: _DatasetInfo | None = None
        for y in yamls:
            try:
                raw = yaml.safe_load(y.read_text()) or {}
            except Exception:
                continue
            if not isinstance(raw, dict) or "train" not in raw or "val" not in raw:
                continue
            train_dir = _resolve_split_dir(ds_dir, str(raw["train"]))
            val_dir = _resolve_split_dir(ds_dir, str(raw["val"]))
            if train_dir is None or val_dir is None:
                continue

            names_raw = raw.get("names", [])
            if isinstance(names_raw, dict):
                class_names = [str(names_raw[k]) for k in sorted(names_raw)]
            else:
                class_names = [str(n) for n in names_raw]

            # A stale absolute path: (e.g. from the machine that generated the
            # labels) breaks Ultralytics — flag for data.local.yaml generation.
            ds_path = raw.get("path")
            needs_fix = bool(ds_path) and Path(str(ds_path)).is_absolute() \
                and not Path(str(ds_path)).exists()

            info = _DatasetInfo(
                name=ds_dir.name,
                dir=ds_dir,
                yaml_path=y,
                n_train=_count_images(train_dir),
                n_val=_count_images(val_dir),
                class_names=class_names,
                needs_local_yaml=needs_fix,
                train_rel=str(raw["train"]),
                val_rel=str(raw["val"]),
            )
            break

        if info is None:
            skipped.append((ds_dir.name, "no yaml with resolvable train/val splits"))
        elif info.n_train == 0:
            skipped.append((ds_dir.name, "train split contains no images"))
        else:
            found.append(info)

    found.sort(key=lambda i: -i.n_train)
    return found, skipped


def _training_yaml(info: _DatasetInfo) -> Path:
    """Return the yaml to pass to Ultralytics, fixing a stale path: if needed."""
    if not info.needs_local_yaml:
        return info.yaml_path
    local = info.dir / "data.local.yaml"
    names_yaml = "\n".join(f"  {i}: {n}" for i, n in enumerate(info.class_names))
    local.write_text(
        f"# Auto-generated by train_detector.py — {info.yaml_path.name} has a\n"
        f"# stale absolute path: from the machine that produced the dataset.\n"
        f"path: {info.dir.resolve()}\n"
        f"train: {info.train_rel}\n"
        f"val: {info.val_rel}\n"
        f"nc: {len(info.class_names)}\n"
        f"names:\n{names_yaml}\n"
    )
    logger.info("Wrote corrected dataset yaml: %s", local)
    return local


# =========================================================================== #
# Survey prompt helpers                                                       #
# =========================================================================== #

def _ask(question: str, options: list[tuple[str, str]], default_idx: int = 0,
         multi: bool = False) -> list[int]:
    """Numbered prompt. Enter → default. ``multi`` allows '1,3'.

    Returns a list of selected indices (length 1 unless multi).
    On EOF (piped input exhausted) the default is chosen.
    """
    print(f"\n{question}")
    for i, (label, desc) in enumerate(options):
        marker = "  (recommended)" if i == default_idx else ""
        print(f"  {i + 1}. {label}{marker}")
        if desc:
            print(f"       {desc}")
    hint = "e.g. 1,3" if multi else "number"
    while True:
        try:
            raw = input(f"Choice [{hint}; Enter = {default_idx + 1}]: ").strip()
        except EOFError:
            print(f"(no input — using default {default_idx + 1})")
            return [default_idx]
        if not raw:
            return [default_idx]
        try:
            picks = [int(t) - 1 for t in raw.replace(" ", "").split(",")] if multi \
                else [int(raw) - 1]
        except ValueError:
            print("  Please enter option number(s).")
            continue
        if all(0 <= p < len(options) for p in picks):
            return picks
        print(f"  Options are 1..{len(options)}.")


def _ask_int(question: str, presets: list[tuple[int, str]], default_idx: int = 0) -> int:
    """Numbered presets plus free-typed positive integer. Enter → default preset."""
    print(f"\n{question}")
    for i, (value, desc) in enumerate(presets):
        marker = "  (recommended)" if i == default_idx else ""
        print(f"  {i + 1}. {value}{marker}")
        if desc:
            print(f"       {desc}")
    while True:
        try:
            raw = input(f"Choice or custom value [Enter = {default_idx + 1}]: ").strip()
        except EOFError:
            print(f"(no input — using default {presets[default_idx][0]})")
            return presets[default_idx][0]
        if not raw:
            return presets[default_idx][0]
        try:
            n = int(raw)
        except ValueError:
            print("  Please enter a number.")
            continue
        if 1 <= n <= len(presets):
            return presets[n - 1][0]
        if n > 0:
            return n  # custom value (e.g. epochs=37)
        print("  Value must be positive.")


def _confirm(prompt: str) -> bool:
    """Y/n confirm. Enter → yes. EOF → no (never start training on missing input)."""
    try:
        raw = input(f"{prompt} [Y/n]: ").strip().lower()
    except EOFError:
        print("(no input — aborting)")
        return False
    return raw in ("", "y", "yes")


def _scan_checkpoints() -> list[tuple[str, Path]]:
    """Discover existing best.pt checkpoints → [(label, path)], label like
    'yolo11m/exp/freeze10_aug_clean'."""
    det_root = _ROOT / "weights" / "detection"
    out: list[tuple[str, Path]] = []
    if not det_root.is_dir():
        return out
    for ckpt in sorted(det_root.glob("**/best.pt")):
        if ckpt.parent.name == "weights":  # skip Ultralytics' inner weights/ copies
            continue
        label = "/".join(ckpt.relative_to(det_root).parts[:-1])
        out.append((label, ckpt))
    return out


# =========================================================================== #
# Shared result reporting / provenance                                       #
# =========================================================================== #

def _print_ranking(results: list[tuple[str, float, float]], title: str) -> None:
    logger.info("")
    logger.info("=" * 70)
    logger.info(title)
    logger.info("%-30s  %8s  %10s", "Run", "mAP50", "mAP50-95")
    logger.info("-" * 55)
    for name, m50, m5095 in sorted(
            results, key=lambda x: -(x[1] if x[1] == x[1] else float("-inf"))):
        logger.info("%-30s  %8.4f  %10.4f", name, m50, m5095)
    logger.info("=" * 70)


def _log_experiment(record: dict[str, Any]) -> None:
    """Append one run record to reports/detection/experiments.jsonl."""
    out = _ROOT / "reports" / "detection" / "experiments.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                cwd=_ROOT, capture_output=True, text=True,
                                check=False).stdout.strip()
    except Exception:
        commit = "unknown"
    record = {"ts": datetime.now().isoformat(timespec="seconds"),
              "git_commit": commit, **record}
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

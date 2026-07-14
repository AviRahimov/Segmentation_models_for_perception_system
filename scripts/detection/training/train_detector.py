"""General detection training — interactive survey or classic CLI.

Interactive mode (no arguments)
-------------------------------
    python scripts/detection/training/train_detector.py

Scans datasets/ for trainable YOLO datasets and walks through a short survey
(dataset, models, recipe, starting weights, epochs, advanced settings).
Every question is numbered; pressing Enter picks the recommended default.
Multiple models can be queued with e.g. ``1,3`` and are trained sequentially.
Output: weights/detection/{model}/{dataset_slug}/{recipe}/best.pt

Classic CLI mode (any argument)
-------------------------------
    python scripts/detection/training/train_detector.py --model yolo26m --variants all
    python scripts/detection/training/train_detector.py --model yolo11m --variants freeze10 aug_clean

Behaves exactly like the former train_exp.py hyperparameter sweep:
output goes to weights/detection/{model}/exp/{variant}/.
Round 2 naming stays reserved for continual learning.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_detector")

# ---------------------------------------------------------------------------
# Model registry (standard YOLO only — no YOLOE; YOLOE uses YOLOEPETrainer)
# Value is either pretrained weights, or (architecture yaml, pretrained
# weights) — the yaml builds a custom architecture (e.g. P2 small-object
# head) and matching pretrained layers are transferred via .load().
# ---------------------------------------------------------------------------
_MODELS: dict[str, str | tuple[str, str]] = {
    "yolo26n": "yolo26n.pt",
    "yolo26s": "yolo26s.pt",
    "yolo26m": "yolo26m.pt",
    "yolo26l": "yolo26l.pt",
    "yolo11n": "yolo11n.pt",
    "yolo11s": "yolo11s.pt",
    "yolo11m": "yolo11m.pt",
    "yolo11l": "yolo11l.pt",
}

# Survey model order: recommended first (sweep results, 2026-07).
_SURVEY_MODEL_ORDER = ["yolo11m", "yolo11s", "yolo26m",
                       "yolo26s", "yolo11n", "yolo26l"]
_SURVEY_MODEL_NOTES = {
    "yolo11m": "sweep winner — best accuracy (mAP50 0.715 on real val)",
    "yolo11s": "best speed/accuracy — Jetson candidate, half the compute of 11m",
    "yolo26m": "best YOLO26 variant",
    "yolo26s": "small YOLO26",
    "yolo11n": "nano — fastest, lowest accuracy",
    "yolo26l": "largest — slowest, rarely worth it on small datasets",
}

# Per-model freeze default (same as round1, used by aug_clean variant)
_FREEZE_DEFAULTS: dict[str, int] = {
    "yolo26n": 10,
    "yolo26s":  8,
    "yolo26m":  6,
    "yolo26l":  4,
    "yolo11n":  7,
    "yolo11s":  6,
    "yolo11m":  5,
    "yolo11l":  4,
}

# ---------------------------------------------------------------------------
# Hyperparameter recipes
# freeze=None → use _FREEZE_DEFAULTS for the model (same as round1)
# ---------------------------------------------------------------------------
_VARIANTS: dict[str, dict[str, Any]] = {
    "freeze0": {
        "freeze": 0,
        "mosaic": 1.0,
        "mixup": 0.2,
        "copy_paste": 0.15,
        "description": "Full fine-tune, round1 augmentation — lower bound reference",
    },
    "freeze10": {
        "freeze": 10,
        "mosaic": 1.0,
        "mixup": 0.2,
        "copy_paste": 0.15,
        "description": "Backbone frozen at layer 10 — research sweet-spot for ~150-image datasets",
    },
    "freeze21": {
        "freeze": 21,
        "mosaic": 1.0,
        "mixup": 0.2,
        "copy_paste": 0.15,
        "description": "Backbone + neck frozen — head-only training",
    },
    "aug_clean": {
        "freeze": None,
        "mosaic": 0.5,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "description": "Round1 freeze, clean augmentation (no blending — avoids boundary corruption)",
    },
    "freeze10_aug_clean": {
        "freeze": 10,
        "mosaic": 0.5,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "description": "freeze=10 + clean augmentation — sweep winner on the 157-image dataset",
    },
    "noaug": {
        "freeze": None,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        # extra train_kwargs overrides beyond the standard recipe knobs
        "extra": {"erasing": 0.0, "degrees": 0.0, "translate": 0.0,
                  "scale": 0.0, "flipud": 0.0},
        "description": "No augmentation (only horizontal flip) — for large / pre-augmented datasets",
    },
}

# Shared base hyperparams (same as round1, recipes only override what changes)
_BASE_KWARGS: dict[str, Any] = {
    "epochs":        150,
    "imgsz":         640,
    "batch":         16,
    "lr0":           2e-4,
    "lrf":           0.01,
    "weight_decay":  5e-4,
    "warmup_epochs": 5,
    "close_mosaic":  30,
    "optimizer":     "AdamW",
    "patience":      20,
    "save_period":   20,
    "workers":       8,
    "degrees":       10.0,
    "translate":     0.2,
    "scale":         0.6,
    "flipud":        0.15,
    "fliplr":        0.5,
    "erasing":       0.4,
    "val":           True,
    "plots":         True,
    "verbose":       False,
    "exist_ok":      True,
}

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


def _recommend_recipe(n_train: int) -> tuple[str, str]:
    """Dataset-size-aware recipe recommendation → (variant_name, reason)."""
    if n_train < 300:
        return "freeze10_aug_clean", f"{n_train} train images — heavy regularization (sweep winner)"
    if n_train <= 1000:
        return "freeze10", f"{n_train} train images — frozen backbone, full augmentation"
    if n_train <= 5000:
        return "aug_clean", f"{n_train} train images — enough data for a less-frozen backbone"
    return "noaug", f"{n_train} train images — large dataset, augmentation adds little"


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


# Preference order for the "auto" starting-weights option: real-data sweep
# winner first, then the round1 baseline.
_AUTO_CKPT_PREFERENCE = ("exp/freeze10_aug_clean/best.pt", "round1/best.pt")


def _auto_checkpoint(model_name: str) -> Path | None:
    """Best existing real-data checkpoint for a model, or None."""
    for rel in _AUTO_CKPT_PREFERENCE:
        p = _ROOT / "weights" / "detection" / model_name / rel
        if p.exists():
            return p
    return None


# =========================================================================== #
# Interactive mode                                                            #
# =========================================================================== #

def run_survey() -> None:
    print("=" * 70)
    print("Detection training — interactive setup (Enter = recommended default)")
    print("=" * 70)

    # ---- Q1: dataset ------------------------------------------------------
    datasets, skipped = _scan_datasets(_ROOT / "datasets")
    if not datasets:
        logger.error("No trainable datasets found under datasets/.")
        sys.exit(1)
    for name, reason in skipped:
        print(f"  (skipped {name}: {reason})")

    ds_options = [
        (info.name,
         f"{info.n_train} train / {info.n_val} val — classes: {', '.join(info.class_names)}")
        for info in datasets
    ]
    # Recommend the largest dataset that matches the pipeline's 2-class scheme;
    # a raw multi-class source (e.g. the 12-class Kaggle original) would train a
    # model incompatible with the config's coco_classes mapping.
    canonical = ["Military Vehicle", "person"]
    ds_default = next((i for i, info in enumerate(datasets)
                       if info.class_names == canonical), 0)
    ds_idx = _ask("1) Which dataset to train on?", ds_options, default_idx=ds_default)[0]
    dataset = datasets[ds_idx]

    # ---- Q2: models (queue) -------------------------------------------------
    model_options = [(m, _SURVEY_MODEL_NOTES.get(m, "")) for m in _SURVEY_MODEL_ORDER]
    model_picks = _ask("2) Which models to train? (a queue like '1,3' runs sequentially)",
                       model_options, default_idx=0, multi=True)
    models = [_SURVEY_MODEL_ORDER[i] for i in model_picks]

    # ---- Q3: recipe -----------------------------------------------------------
    rec_name, rec_reason = _recommend_recipe(dataset.n_train)
    recipe_names = list(_VARIANTS.keys())
    recipe_options = []
    for name in recipe_names:
        desc = _VARIANTS[name]["description"]
        if name == rec_name:
            desc += f"  [{rec_reason}]"
        recipe_options.append((name, desc))
    recipe_picks = _ask("3) Which training recipes? (a queue like '2,4' runs each per model)",
                        recipe_options, default_idx=recipe_names.index(rec_name),
                        multi=True)
    recipes = [recipe_names[i] for i in recipe_picks]

    # ---- Q4: starting weights --------------------------------------------------
    ckpts = _scan_checkpoints()
    init_options = [
        ("COCO-pretrained base",
         "official Ultralytics weights for each queued model"),
        ("Auto: best existing checkpoint per model",
         "fine-tune each queued model from its own real-data checkpoint "
         "(exp/freeze10_aug_clean, else round1); lr is lowered automatically"),
    ]
    init_options += [(label, "") for label, _ in ckpts]
    init_idx = _ask("4) Starting weights?", init_options, default_idx=0)[0]
    init_mode = "coco" if init_idx == 0 else "auto" if init_idx == 1 else "manual"
    manual_ckpt: Path | None = ckpts[init_idx - 2][1] if init_mode == "manual" else None
    manual_ckpt_model = ckpts[init_idx - 2][0].split("/")[0] if manual_ckpt else None

    def _resolve_init(model_name: str) -> tuple[str, bool]:
        """→ (weights to load, is_finetune). Falls back to COCO with a note."""
        if init_mode == "auto":
            auto = _auto_checkpoint(model_name)
            if auto is not None:
                return str(auto), True
            logger.warning("%s: no existing checkpoint found — using COCO base.", model_name)
        elif init_mode == "manual":
            if manual_ckpt_model == model_name:
                return str(manual_ckpt), True
            logger.warning("%s: selected checkpoint is a %s — falling back to COCO base.",
                           model_name, manual_ckpt_model)
        return _MODELS[model_name], False

    # ---- Q5: epochs ---------------------------------------------------------
    epochs = _ask_int(
        "5) How many epochs? (or type any number)",
        [(150, "standard — early stopping (patience=20) usually ends sooner"),
         (50,  "quick sanity run"),
         (300, "long — for large datasets / slow convergence")],
        default_idx=0,
    )

    # ---- Q6: advanced -----------------------------------------------------------
    adv_idx = _ask("6) Advanced settings?",
                   [("Use defaults", "batch=16, imgsz=640, device=0, seed=42"),
                    ("Customize",    "enter each value manually")],
                   default_idx=0)[0]
    batch, imgsz, device, seed = 16, 640, "0", 42
    if adv_idx == 1:
        def _read(prompt: str, default: Any, cast=int) -> Any:
            try:
                raw = input(f"  {prompt} [{default}]: ").strip()
            except EOFError:
                return default
            return cast(raw) if raw else default
        batch = _read("batch", batch)
        imgsz = _read("imgsz", imgsz)
        device = _read("device", device, cast=str)
        seed = _read("seed", seed)

    # ---- Summary + confirm --------------------------------------------------
    dataset_slug = dataset.name.lower()
    # (model, base_weights, is_finetune) resolved up front so the summary is exact.
    model_inits = [(m, *_resolve_init(m)) for m in models]

    print("\n" + "=" * 70)
    print("Training plan")
    print(f"  dataset:  {dataset.name}  ({dataset.n_train} train / {dataset.n_val} val)")
    print(f"  models:   {', '.join(models)}")
    print(f"  recipes:  {', '.join(recipes)}")
    for m, base_weights, is_ft in model_inits:
        tag = "fine-tune (lr0=5e-5, warmup=2)" if is_ft else "COCO base"
        print(f"  init:     {m}: {base_weights}  [{tag}]")
    print(f"  epochs:   {epochs}   batch: {batch}   imgsz: {imgsz}   device: {device}   seed: {seed}")
    print(f"  queue:    {len(models)} model(s) x {len(recipes)} recipe(s) = {len(models) * len(recipes)} training run(s)")
    for m, _, is_ft in model_inits:
        for r in recipes:
            run_name = f"{r}_ft" if is_ft else r
            print(f"  output:   weights/detection/{m}/{dataset_slug}/{run_name}/")
    print("=" * 70)
    if not _confirm("Start training?"):
        print("Aborted — nothing trained.")
        return

    # ---- Train queue ----------------------------------------------------------
    seed_everything(seed)
    data_yaml = _training_yaml(dataset)
    results_summary: list[tuple[str, float, float]] = []

    for model_name, base_weights, is_ft in model_inits:
        for recipe_name in recipes:
            recipe = _VARIANTS[recipe_name]
            freeze = recipe["freeze"] if recipe["freeze"] is not None \
                else _FREEZE_DEFAULTS.get(model_name, 8)

            # _ft suffix keeps fine-tune runs from overwriting COCO-init runs.
            run_name = f"{recipe_name}_ft" if is_ft else recipe_name
            out_dir = _ROOT / "weights" / "detection" / model_name / dataset_slug / run_name
            out_dir.mkdir(parents=True, exist_ok=True)

            train_kwargs: dict[str, Any] = {
                **_BASE_KWARGS,
                "data":       str(data_yaml),
                "epochs":     epochs,
                "imgsz":      imgsz,
                "batch":      batch,
                "device":     device,
                "seed":       seed,
                "freeze":     freeze,
                "mosaic":     recipe["mosaic"],
                "mixup":      recipe["mixup"],
                "copy_paste": recipe["copy_paste"],
                **recipe.get("extra", {}),
                "project":    str(out_dir.parent),
                "name":       run_name,
            }
            if is_ft:
                # Gentle fine-tune: protect the pretrained real-data features
                # from being blown away during the first epochs.
                train_kwargs["lr0"] = 5e-5
                train_kwargs["warmup_epochs"] = 2

            logger.info("")
            logger.info("=" * 70)
            logger.info("Training %s on %s  (recipe=%s, freeze=%d, epochs=%d%s)",
                        model_name, dataset.name, run_name, freeze, epochs,
                        ", fine-tune lr0=5e-5" if is_ft else "")
            logger.info("=" * 70)
            run_label = f"{model_name}/{run_name}"
            try:
                map50, map5095 = _run_variant(base_weights, train_kwargs, out_dir,
                                              model_name, run_label)
            except Exception as exc:  # noqa: BLE001 — one failed run must not kill the queue
                logger.error("Run %s FAILED: %s", run_label, exc)
                logger.error("Continuing with the next queued run.")
                results_summary.append((f"{run_label} [FAILED]",
                                        float("nan"), float("nan")))
                _log_experiment({
                    "mode": "survey", "model": model_name, "dataset": dataset.name,
                    "recipe": run_name, "run_dir": str(out_dir.relative_to(_ROOT)),
                    "error": str(exc),
                })
                continue
            results_summary.append((run_label, map50, map5095))
            _log_experiment({
                "mode": "survey", "model": model_name, "dataset": dataset.name,
                "recipe": run_name, "run_dir": str(out_dir.relative_to(_ROOT)),
                "epochs": epochs, "imgsz": imgsz, "batch": batch, "seed": seed,
                "init": str(base_weights),
                "train_mAP50": round(map50, 5) if map50 == map50 else None,
                "train_mAP50_95": round(map5095, 5) if map5095 == map5095 else None,
            })

    _print_ranking(results_summary, title=f"QUEUE COMPLETE — {dataset.name}")


# =========================================================================== #
# Classic CLI mode (former train_exp.py sweep)                                #
# =========================================================================== #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detection training — run with NO arguments for the interactive survey"
    )
    p.add_argument("--model", required=True, choices=list(_MODELS.keys()),
                   help="Model to train (e.g. yolo26m, yolo11m)")
    p.add_argument("--variants", nargs="+", default=["all"],
                   metavar="VARIANT",
                   help=f"Variants to run (default: all). Choices: {list(_VARIANTS)}")
    p.add_argument("--data", default="datasets/Detection_Dataset/data.yaml",
                   help="Path to YOLO data.yaml")
    p.add_argument("--device", default="0",
                   help="CUDA device index or 'cpu' (default: 0)")
    p.add_argument("--seed",   type=int, default=42)
    return p.parse_args()


def main_cli() -> None:
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)
        logger.info("Global seed: %d", args.seed)

    # Resolve which variants to run
    if args.variants == ["all"]:
        variants_to_run = list(_VARIANTS.keys())
    else:
        unknown = [v for v in args.variants if v not in _VARIANTS]
        if unknown:
            logger.error("Unknown variants: %s. Valid choices: %s", unknown, list(_VARIANTS))
            sys.exit(1)
        variants_to_run = args.variants

    # Resolve data path
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = _ROOT / data_path
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")

    base_weights = _MODELS[args.model]
    default_freeze = _FREEZE_DEFAULTS.get(args.model, 8)

    logger.info("Model:    %s  (base weights: %s)", args.model, base_weights)
    logger.info("Variants: %s", variants_to_run)
    logger.info("Data:     %s", data_path)

    results_summary: list[tuple[str, float, float]] = []

    for variant_name in variants_to_run:
        variant = _VARIANTS[variant_name]
        freeze = variant["freeze"] if variant["freeze"] is not None else default_freeze

        out_dir = _ROOT / "weights" / "detection" / args.model / "exp" / variant_name
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("")
        logger.info("=" * 70)
        logger.info("Variant: %s  |  freeze=%d  mosaic=%.1f  mixup=%.2f  copy_paste=%.2f",
                    variant_name, freeze, variant["mosaic"], variant["mixup"], variant["copy_paste"])
        logger.info("  %s", variant["description"])
        logger.info("  Output: %s", out_dir)
        logger.info("=" * 70)

        train_kwargs: dict[str, Any] = {
            **_BASE_KWARGS,
            "data":       str(data_path),
            "device":     args.device,
            "seed":       args.seed,
            "freeze":     freeze,
            "mosaic":     variant["mosaic"],
            "mixup":      variant["mixup"],
            "copy_paste": variant["copy_paste"],
            **variant.get("extra", {}),
            "project":    str(out_dir.parent),
            "name":       variant_name,
        }

        try:
            map50, map5095 = _run_variant(base_weights, train_kwargs, out_dir,
                                          args.model, variant_name)
        except Exception as exc:  # noqa: BLE001 — keep sweeping remaining variants
            logger.error("Variant %s FAILED: %s — continuing.", variant_name, exc)
            results_summary.append((f"{variant_name} [FAILED]",
                                    float("nan"), float("nan")))
            _log_experiment({
                "mode": "cli", "model": args.model, "dataset": data_path.parent.name,
                "recipe": variant_name, "run_dir": str(out_dir.relative_to(_ROOT)),
                "error": str(exc),
            })
            continue
        results_summary.append((variant_name, map50, map5095))
        _log_experiment({
            "mode": "cli", "model": args.model, "dataset": data_path.parent.name,
            "recipe": variant_name, "run_dir": str(out_dir.relative_to(_ROOT)),
            "epochs": train_kwargs["epochs"], "imgsz": train_kwargs["imgsz"],
            "batch": train_kwargs["batch"], "seed": args.seed,
            "init": str(base_weights),
            "train_mAP50": round(map50, 5) if map50 == map50 else None,
            "train_mAP50_95": round(map5095, 5) if map5095 == map5095 else None,
        })

    _print_ranking(results_summary, title=f"SWEEP COMPLETE — {args.model}")
    logger.info("Run summarize_exp.py to compare with round1 baseline.")


# =========================================================================== #
# Shared training internals                                                   #
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


def _attach_progress(model: Any, run_label: str, epochs: int) -> dict:
    """Replace Ultralytics' raw log tables with two clean tqdm bars.

    Returns a state dict updated by the callbacks (best mAP, best epoch).
    Falls back silently to native logs if the callback API is unavailable.
    """
    state: dict[str, Any] = {"best": 0.0, "best_ep": 0, "epoch_bar": None,
                             "batch_bar": None, "ok": False}
    try:
        from ultralytics.utils import LOGGER as ultra_logger

        epoch_bar = tqdm(total=epochs, desc=run_label, unit="ep",
                         position=0, leave=True, dynamic_ncols=True)
        state["epoch_bar"] = epoch_bar

        def on_train_epoch_start(trainer):
            if state["batch_bar"] is not None:
                state["batch_bar"].close()
            state["batch_bar"] = tqdm(
                total=len(trainer.train_loader),
                desc=f"epoch {trainer.epoch + 1}/{epochs}",
                unit="it", position=1, leave=False, dynamic_ncols=True,
            )

        def on_train_batch_end(trainer):
            bar = state["batch_bar"]
            if bar is None:
                return
            bar.update(1)
            try:
                li = trainer.loss_items
                bar.set_postfix(box=f"{float(li[0]):.3f}",
                                cls=f"{float(li[1]):.3f}", refresh=False)
            except Exception:
                pass

        def on_fit_epoch_end(trainer):
            if state["batch_bar"] is not None:
                state["batch_bar"].close()
                state["batch_bar"] = None
            m = trainer.metrics or {}
            map50 = float(m.get("metrics/mAP50(B)", 0.0))
            map5095 = float(m.get("metrics/mAP50-95(B)", 0.0))
            if map50 > state["best"]:
                state["best"] = map50
                state["best_ep"] = trainer.epoch + 1
            epoch_bar.set_postfix(
                mAP50=f"{map50:.3f}",
                best=f"{state['best']:.3f}@{state['best_ep']}",
                mAP95=f"{map5095:.3f}",
            )
            epoch_bar.update(1)

        model.add_callback("on_train_epoch_start", on_train_epoch_start)
        model.add_callback("on_train_batch_end", on_train_batch_end)
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        ultra_logger.setLevel(logging.WARNING)
        state["ok"] = True
    except Exception as exc:  # noqa: BLE001 — fall back to native logs
        logger.warning("Progress bars unavailable (%s) — using native logs.", exc)
    return state


def _run_variant(
    base_weights: str | tuple[str, str],
    train_kwargs: dict[str, Any],
    out_dir: Path,
    model_name: str,
    variant_name: str,
) -> tuple[float, float]:
    """Train one run and return (mAP50, mAP50-95). Returns (nan, nan) on error."""
    from ultralytics import YOLO

    if isinstance(base_weights, tuple):
        arch_yaml, pretrained = base_weights
        arch_path = Path(arch_yaml)
        # Repo-local yamls resolve against _ROOT; bare names (e.g. a yaml
        # shipped with Ultralytics itself) must pass through untouched so
        # Ultralytics finds them in its own cfg/models registry.
        if not arch_path.is_absolute() and (_ROOT / arch_path).exists():
            arch_path = _ROOT / arch_path
        logger.info("Building custom architecture %s + pretrained %s",
                    arch_path.name, pretrained)
        model = YOLO(str(arch_path)).load(pretrained)
    else:
        logger.info("Loading base weights: %s", base_weights)
        model = YOLO(base_weights)

    logger.info("Training freeze=%d epochs=%d batch=%d imgsz=%d ...",
                train_kwargs["freeze"], train_kwargs["epochs"],
                train_kwargs["batch"], train_kwargs["imgsz"])
    run_label = f"{model_name}/{variant_name.split('/')[-1]}"
    progress = _attach_progress(model, run_label, int(train_kwargs["epochs"]))
    try:
        results = model.train(**train_kwargs)
    finally:
        if progress.get("batch_bar") is not None:
            progress["batch_bar"].close()
        if progress.get("epoch_bar") is not None:
            bar = progress["epoch_bar"]
            if progress["ok"] and bar.n < bar.total:
                bar.total = bar.n  # early stop — snap the bar shut
            bar.close()
    if progress["ok"] and progress["best"]:
        n_done = progress["epoch_bar"].n if progress.get("epoch_bar") else 0
        if n_done < int(train_kwargs["epochs"]):
            logger.info("Early stop after %d epochs (patience=%s).",
                        n_done, train_kwargs.get("patience"))
        logger.info("Best mAP50 %.4f @ epoch %d", progress["best"], progress["best_ep"])

    # Locate best.pt — Ultralytics saves under {project}/{name}/weights/
    weights_dir = out_dir / "weights"
    best_src = weights_dir / "best.pt"
    last_src = weights_dir / "last.pt"

    if not best_src.exists() and hasattr(model, "trainer") and model.trainer is not None:
        best_src = model.trainer.best
        last_src = model.trainer.last

    best_dest = out_dir / "best.pt"
    last_dest = out_dir / "last.pt"

    if best_src.exists() and best_src.resolve() != best_dest.resolve():
        shutil.copy2(str(best_src), str(best_dest))
        logger.info("Best checkpoint → %s", best_dest)

    if last_src.exists() and last_src.resolve() != last_dest.resolve():
        shutil.copy2(str(last_src), str(last_dest))

    map50, map5095 = float("nan"), float("nan")
    if results is not None:
        try:
            metrics = results.results_dict
            map50   = metrics.get("metrics/mAP50(B)",    float("nan"))
            map5095 = metrics.get("metrics/mAP50-95(B)", float("nan"))
            logger.info("Run %s — mAP50=%.4f  mAP50-95=%.4f", variant_name, map50, map5095)
        except Exception:
            pass

    return map50, map5095


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main_cli()
    else:
        run_survey()

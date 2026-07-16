#!/usr/bin/env python3
"""RF-DETR detection training — interactive survey (separate from YOLO).

Why a separate script (not a Q2 option in train_detector.py)
--------------------------------------------------------------
RF-DETR's training stack is a genuinely different shape from Ultralytics':
PyTorch-Lightning-based (not the Ultralytics ``Trainer``), fixed
per-variant input resolution (no ``imgsz``), its own checkpoint format/
naming (``checkpoint_best_total.pth``), and its own progress-bar +
early-stopping support. It has no freeze-N-layers concept either — Q3's
recipes vary ``lr_vit_layer_decay`` (a discriminative per-layer learning
rate that protects the pretrained DINOv2 backbone more as it's lowered,
RF-DETR's actual analogue to freezing) and RF-DETR's own named
augmentation presets (``rfdetr.datasets.aug_config``) instead of YOLO's
mosaic/mixup/freeze grid — porting that grid verbatim would mean faking
support that doesn't apply to this architecture. Dataset scanning, the
prompt UX, checkpoint discovery, and experiment-log provenance ARE shared
(via ``_survey_common.py``) since none of that is trainer-specific.

Uses its OWN venv, deliberately separate from this project's main one:
    python3.12 -m venv .venv-rfdetr-train
    source .venv-rfdetr-train/bin/activate
    pip install -r requirements-rfdetr-train.txt
    python scripts/detection/training/train_detector_rfdetr.py

Why a separate venv (not just a pip install in the main one)
--------------------------------------------------------------
XL/2XL need ``rfdetr-plus``, which requires ``rfdetr>=1.6.0``, which requires
``transformers>=5.1.0`` — incompatible with this project's main venv, which
pins ``transformers==4.46.3`` for SegFormer. See requirements-rfdetr-train.txt
for the full explanation. This script therefore imports NOTHING from
``src/perception`` (that package tree pulls in cv2 and other main-venv-only
deps) — the RF-DETR variant table below is a deliberately-duplicated copy of
``src/perception/models/instance/rfdetr/model.py:_RFDETR_VARIANTS``. A
trained checkpoint is a plain ``.pt`` file; it gets loaded back into the
main venv for inference same as any other checkpoint, so this isolation
only affects the training run itself.

Known limitation: leaderboard.py / compare_detection_models.py currently
hardcode ``ultralytics.YOLO(ckpt)`` to load checkpoints and cannot yet load
an RF-DETR ``.pth`` — a trained RF-DETR checkpoint is not comparable on the
shared leaderboard until that tooling is extended separately.

Usage
-----
    python scripts/detection/training/train_detector_rfdetr.py
"""
from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[3]

from _survey_common import (  # noqa: E402
    _ask,
    _ask_int,
    _confirm,
    _log_experiment,
    _print_ranking,
    _scan_checkpoints,
    _scan_datasets,
    seed_everything,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_detector_rfdetr")

# Must happen before the first `import rfdetr` anywhere in this process:
# rfdetr's own plus-extra detection (`rfdetr.platform._IS_RFDETR_PLUS_AVAILABLE`,
# computed via `importlib.util.find_spec("rfdetr_plus.models")`) is a circular
# import — resolving that spec needs to import rfdetr_plus, which imports
# rfdetr back, which is still mid-import the first time this runs. Verified
# empirically: importing rfdetr_plus first makes the flag resolve correctly;
# importing rfdetr first makes XL/2XL permanently unavailable in that
# process even though the [plus] extra IS installed. Best-effort/no-op when
# rfdetr-plus isn't installed at all (N/S/M/L don't need it).
try:
    import rfdetr_plus  # noqa: F401
except ImportError:
    pass

# Mirrors src/perception/models/instance/rfdetr/model.py's _RFDETR_VARIANTS —
# duplicated rather than imported so this script has zero dependency on the
# main venv's package tree (see module docstring). Keep both in sync if
# RF-DETR ever adds a variant.
_RFDETR_VARIANTS: dict[str, tuple[str, int]] = {
    "rfdetr-n":   ("RFDETRNano",    384),
    "rfdetr-s":   ("RFDETRSmall",   512),
    "rfdetr-m":   ("RFDETRMedium",  576),
    "rfdetr-l":   ("RFDETRLarge",   704),
    "rfdetr-xl":  ("RFDETRXLarge",  700),
    "rfdetr-2xl": ("RFDETR2XLarge", 880),
}

# Survey order: recommended default first.
_SURVEY_VARIANT_ORDER = ["rfdetr-m", "rfdetr-s", "rfdetr-l", "rfdetr-n",
                         "rfdetr-xl", "rfdetr-2xl"]
_SURVEY_VARIANT_NOTES = {
    "rfdetr-n":   "nano — 384px, fastest, lowest accuracy",
    "rfdetr-s":   "small — 512px",
    "rfdetr-m":   "medium — 576px, good speed/accuracy balance (recommended default)",
    "rfdetr-l":   "large — 704px, highest accuracy of the free variants",
    "rfdetr-xl":  "xlarge — 700px",
    "rfdetr-2xl": "2xlarge — 880px",
}
_PLUS_VARIANTS = {"rfdetr-xl", "rfdetr-2xl"}

# ---------------------------------------------------------------------------
# Training recipes — RF-DETR's actual knobs, not a port of YOLO's freeze/aug
# grid (there's no per-layer freeze concept here). ``lr_vit_layer_decay`` is
# RF-DETR's analogue to freezing: a discriminative per-layer LR multiplier
# that protects the pretrained DINOv2 backbone more as it's lowered, rather
# than a hard freeze/unfreeze split. ``aug`` selects one of RF-DETR's own
# named presets from rfdetr.datasets.aug_config — "conservative" is that
# package's own documented recommendation for datasets under 500 images.
# ---------------------------------------------------------------------------
_RFDETR_RECIPES: dict[str, dict[str, Any]] = {
    "default": {
        "lr_vit_layer_decay": 0.8,
        "aug": "default",
        "description": "RF-DETR's own defaults — mild flip-only augmentation, standard backbone LR decay",
    },
    "conservative_aug": {
        "lr_vit_layer_decay": 0.8,
        "aug": "conservative",
        "description": "RF-DETR's AUG_CONSERVATIVE preset (its own recommendation for <500-image datasets)",
    },
    "no_aug": {
        "lr_vit_layer_decay": 0.8,
        "aug": "none",
        "description": "Augmentation fully disabled (aug_config={})",
    },
    "strong_decay": {
        "lr_vit_layer_decay": 0.5,
        "aug": "default",
        "description": "Stronger backbone-layer LR decay (protects DINOv2 features more), default augmentation",
    },
    "strong_decay_conservative_aug": {
        "lr_vit_layer_decay": 0.5,
        "aug": "conservative",
        "description": "Combines both — closest RF-DETR analogue to YOLO's freeze10_aug_clean sweep winner",
    },
}


def _recommend_rfdetr_recipe(n_train: int) -> tuple[str, str]:
    """Dataset-size-aware recipe recommendation → (recipe_name, reason)."""
    if n_train < 500:
        return ("strong_decay_conservative_aug",
                f"{n_train} train images — RF-DETR's own small-dataset augmentation "
                "preset + stronger backbone LR decay")
    return ("default", f"{n_train} train images — RF-DETR's defaults are already tuned for this scale")


def _resolve_aug_config(tag: str) -> dict | None:
    """None = let RF-DETR use its own default (AUG_CONFIG); {} = fully
    disabled; otherwise one of rfdetr's own named presets."""
    if tag == "default":
        return None
    if tag == "none":
        return {}
    if tag == "conservative":
        from rfdetr.datasets.aug_config import AUG_CONSERVATIVE
        return dict(AUG_CONSERVATIVE)
    raise ValueError(f"Unknown aug tag: {tag!r}")


def _ensure_valid_symlink(dataset_dir: Path) -> None:
    """RF-DETR's YOLO-format loader hardcodes a ``valid/`` split directory
    (``rfdetr/datasets/yolo.py: REQUIRED_SPLIT_DIRS``); this project's
    ``build_dataset.py`` output uses ``val/``. Non-destructive fix: symlink
    ``valid -> val`` when only ``val/`` exists."""
    valid_dir = dataset_dir / "valid"
    val_dir = dataset_dir / "val"
    if valid_dir.exists() or not val_dir.is_dir():
        return
    valid_dir.symlink_to(val_dir.resolve(), target_is_directory=True)
    logger.info("Created symlink for RF-DETR compatibility: %s -> %s", valid_dir, val_dir)


def _check_variant_available(model_name: str) -> None:
    """Abort with a clear install command before training starts (not
    mid-queue) if the picked variant's class isn't in the installed rfdetr.

    Can't use plain ``hasattr()`` here: rfdetr's own ``__getattr__`` raises
    ``ImportError`` (not ``AttributeError``) for XL/2XL when the ``[plus]``
    extra is missing, which ``hasattr()`` does not swallow — it propagates.
    """
    import rfdetr as rfdetr_pkg

    cls_name, _ = _RFDETR_VARIANTS[model_name]
    try:
        available = getattr(rfdetr_pkg, cls_name, None) is not None
    except Exception:
        available = False
    if not available:
        extra = "rfdetr[plus]" if model_name in _PLUS_VARIANTS else "rfdetr"
        logger.error(
            "%s (%s) is not available in the installed rfdetr package. "
            "Install it with: pip install \"%s\"",
            model_name, cls_name, extra,
        )
        sys.exit(1)


def _check_correct_venv() -> None:
    """Fail clearly, before anything else, if this is the main project venv
    rather than the dedicated training one (see module docstring) — the
    alternative is a deep pydantic traceback from a mismatched rfdetr
    version several questions into the survey."""
    try:
        import pytorch_lightning  # noqa: F401
    except ImportError:
        logger.error(
            "pytorch_lightning is not importable — this looks like the main "
            "project venv, not the dedicated RF-DETR training venv. Run:\n"
            "    python3.12 -m venv .venv-rfdetr-train\n"
            "    source .venv-rfdetr-train/bin/activate\n"
            "    pip install -r requirements-rfdetr-train.txt"
        )
        sys.exit(1)


# =========================================================================== #
# Interactive mode                                                            #
# =========================================================================== #

def run_survey() -> None:
    _check_correct_venv()
    print("=" * 70)
    print("RF-DETR training — interactive setup (Enter = recommended default)")
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
    canonical = ["Military Vehicle", "person"]
    ds_default = next((i for i, info in enumerate(datasets)
                       if info.class_names == canonical), 0)
    ds_idx = _ask("1) Which dataset to train on?", ds_options, default_idx=ds_default)[0]
    dataset = datasets[ds_idx]
    _ensure_valid_symlink(dataset.dir)

    # ---- Q2: RF-DETR variant(s) (queue) ------------------------------------
    variant_options = [(v, _SURVEY_VARIANT_NOTES.get(v, "")) for v in _SURVEY_VARIANT_ORDER]
    variant_picks = _ask(
        "2) Which RF-DETR variant(s) to train? (a queue like '1,3' runs sequentially)",
        variant_options, default_idx=0, multi=True,
    )
    variants = [_SURVEY_VARIANT_ORDER[i] for i in variant_picks]
    for v in variants:
        _check_variant_available(v)

    # ---- Q3: recipe (queue) -----------------------------------------------
    rec_name, rec_reason = _recommend_rfdetr_recipe(dataset.n_train)
    recipe_names = list(_RFDETR_RECIPES.keys())
    recipe_options = []
    for name in recipe_names:
        desc = _RFDETR_RECIPES[name]["description"]
        if name == rec_name:
            desc += f"  [{rec_reason}]"
        recipe_options.append((name, desc))
    recipe_picks = _ask(
        "3) Which training recipes? (a queue like '2,4' runs each per variant)",
        recipe_options, default_idx=recipe_names.index(rec_name), multi=True,
    )
    recipes = [recipe_names[i] for i in recipe_picks]

    # ---- Q4: starting weights --------------------------------------------------
    ckpts = [(label, path) for label, path in _scan_checkpoints()
             if label.split("/")[0] in variants]
    init_options = [
        ("COCO-pretrained base", "official RF-DETR pretrained weights for each queued variant"),
    ]
    init_options += [(label, "") for label, _ in ckpts]
    init_idx = _ask("4) Starting weights?", init_options, default_idx=0)[0]
    manual_ckpt: Path | None = ckpts[init_idx - 1][1] if init_idx > 0 else None
    manual_ckpt_model = ckpts[init_idx - 1][0].split("/")[0] if manual_ckpt else None

    def _resolve_init(model_name: str) -> tuple[str | None, bool]:
        """→ (pretrain_weights path or None, is_finetune)."""
        if manual_ckpt is not None:
            if manual_ckpt_model == model_name:
                return str(manual_ckpt), True
            logger.warning("%s: selected checkpoint is a %s — falling back to COCO base.",
                           model_name, manual_ckpt_model)
        return None, False

    # ---- Q5: epochs ---------------------------------------------------------
    epochs = _ask_int(
        "5) How many epochs? (or type any number)",
        [(100, "RF-DETR's own default — early stopping usually ends sooner"),
         (30,  "quick sanity run"),
         (200, "long — for large datasets / slow convergence")],
        default_idx=0,
    )

    # ---- Q6: advanced -----------------------------------------------------------
    adv_idx = _ask("6) Advanced settings?",
                   [("Use defaults", "batch_size=auto, lr=1e-4, lr_encoder=1.5e-4, seed=42"),
                    ("Customize",    "enter each value manually")],
                   default_idx=0)[0]
    batch_size: Any = "auto"
    lr, lr_encoder, seed = 1e-4, 1.5e-4, 42
    if adv_idx == 1:
        def _read(prompt: str, default: Any, cast=float) -> Any:
            try:
                raw = input(f"  {prompt} [{default}]: ").strip()
            except EOFError:
                return default
            return cast(raw) if raw else default
        batch_raw = _read("batch_size (int or 'auto')", batch_size, cast=str)
        batch_size = batch_raw if batch_raw == "auto" else int(batch_raw)
        lr = _read("lr", lr)
        lr_encoder = _read("lr_encoder", lr_encoder)
        seed = _read("seed", seed, cast=int)

    # ---- Summary + confirm --------------------------------------------------
    dataset_slug = dataset.name.lower()
    variant_inits = [(v, *_resolve_init(v)) for v in variants]

    print("\n" + "=" * 70)
    print("RF-DETR training plan")
    print(f"  dataset:   {dataset.name}  ({dataset.n_train} train / {dataset.n_val} val)")
    print(f"  variants:  {', '.join(variants)}")
    print(f"  recipes:   {', '.join(recipes)}")
    for v, weights, is_ft in variant_inits:
        tag = f"fine-tune from {weights}" if is_ft else "COCO-pretrained base"
        print(f"  init:      {v}: {tag}")
    print(f"  epochs:    {epochs}   batch_size: {batch_size}   seed: {seed}")
    print(f"  queue:     {len(variants)} variant(s) x {len(recipes)} recipe(s) "
          f"= {len(variants) * len(recipes)} training run(s)")
    for v, _, is_ft in variant_inits:
        for r in recipes:
            run_name = f"{r}_ft" if is_ft else r
            print(f"  output:    weights/detection/{v}/{dataset_slug}/{run_name}/")
    print("=" * 70)
    if not _confirm("Start training?"):
        print("Aborted — nothing trained.")
        return

    # ---- Train queue ----------------------------------------------------------
    seed_everything(seed)
    results_summary: list[tuple[str, float, float]] = []

    for model_name, weights, is_ft in variant_inits:
        for recipe_name in recipes:
            recipe = _RFDETR_RECIPES[recipe_name]
            run_name = f"{recipe_name}_ft" if is_ft else recipe_name
            out_dir = _ROOT / "weights" / "detection" / model_name / dataset_slug / run_name
            out_dir.mkdir(parents=True, exist_ok=True)

            train_kwargs: dict[str, Any] = {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "lr_encoder": lr_encoder,
                "seed": seed,
                "lr_vit_layer_decay": recipe["lr_vit_layer_decay"],
                "aug_config": _resolve_aug_config(recipe["aug"]),
            }

            logger.info("")
            logger.info("=" * 70)
            logger.info("Training %s on %s  (recipe=%s, epochs=%d%s)",
                        model_name, dataset.name, run_name, epochs,
                        f", fine-tune from {weights}" if is_ft else "")
            logger.info("=" * 70)
            run_label = f"{model_name}/{dataset_slug}/{run_name}"
            try:
                map50, map5095 = _run_rfdetr_variant(model_name, dataset.dir, weights,
                                                     train_kwargs, out_dir)
            except Exception as exc:  # noqa: BLE001 — one failed run must not kill the queue
                logger.error("Run %s FAILED: %s", run_label, exc)
                logger.error("Continuing with the next queued run.")
                results_summary.append((f"{run_label} [FAILED]", float("nan"), float("nan")))
                _log_experiment({
                    "mode": "survey", "arch_family": "rfdetr", "model": model_name,
                    "dataset": dataset.name, "recipe": run_name,
                    "run_dir": str(out_dir.relative_to(_ROOT)), "error": str(exc),
                })
                continue
            results_summary.append((run_label, map50, map5095))
            _log_experiment({
                "mode": "survey", "arch_family": "rfdetr", "model": model_name,
                "dataset": dataset.name, "recipe": run_name,
                "run_dir": str(out_dir.relative_to(_ROOT)),
                "epochs": epochs, "batch_size": batch_size, "seed": seed,
                "init": weights or "coco",
                "train_mAP50": round(map50, 5) if map50 == map50 else None,
                "train_mAP50_95": round(map5095, 5) if map5095 == map5095 else None,
            })

    _print_ranking(results_summary, title=f"QUEUE COMPLETE — {dataset.name}")


# =========================================================================== #
# Training internals                                                          #
# =========================================================================== #

def _run_rfdetr_variant(
    model_name: str,
    dataset_dir: Path,
    weights: str | None,
    train_kwargs: dict[str, Any],
    out_dir: Path,
) -> tuple[float, float]:
    """Train one RF-DETR variant and return (best mAP50, mAP50-95) read back
    from PTL's metrics.csv. NaN if that file is missing or unreadable."""
    import rfdetr as rfdetr_pkg

    cls_name, _ = _RFDETR_VARIANTS[model_name]
    cls = getattr(rfdetr_pkg, cls_name)
    model = cls(pretrain_weights=weights) if weights else cls()

    logger.info("Training epochs=%d batch_size=%s ...",
                train_kwargs["epochs"], train_kwargs["batch_size"])
    # RFDETR.train() trains in place via PyTorch Lightning and returns None —
    # unlike Ultralytics' YOLO.train(), there's no results object here;
    # metrics are read back from its CSVLogger output below instead.
    model.train(
        dataset_dir=str(dataset_dir),
        dataset_file="yolo",
        output_dir=str(out_dir),
        progress_bar="tqdm",
        early_stopping=True,
        **train_kwargs,
    )

    # RF-DETR's canonical "best" checkpoint (winner of regular-vs-EMA,
    # optimizer state stripped) — copy alongside Ultralytics' best.pt
    # convention so _scan_checkpoints() discovers it uniformly.
    best_src = out_dir / "checkpoint_best_total.pth"
    best_dest = out_dir / "best.pt"
    if best_src.exists():
        shutil.copy2(str(best_src), str(best_dest))
        logger.info("Best checkpoint → %s", best_dest)
    else:
        logger.warning("Expected checkpoint not found: %s", best_src)

    map50, map5095 = _read_final_metrics(out_dir)
    if map50 == map50:  # not NaN
        logger.info("Run %s — mAP50=%.4f  mAP50-95=%.4f", model_name, map50, map5095)
    else:
        logger.info("Could not read final metrics from %s/metrics.csv — "
                    "check that file directly.", out_dir)
    return map50, map5095


def _read_final_metrics(out_dir: Path) -> tuple[float, float]:
    """Best (not merely last) val/mAP_50 and val/mAP_50_95 from PyTorch
    Lightning's CSVLogger output. Most rows are train-only (val columns
    blank — validation runs less often than train steps), so this scans
    every row rather than just the last one. NaN, NaN if the file is
    missing or the expected columns aren't present."""
    import csv

    metrics_csv = out_dir / "metrics.csv"
    if not metrics_csv.exists():
        return float("nan"), float("nan")

    best50, best5095 = float("nan"), float("nan")
    with metrics_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            raw = row.get("val/mAP_50", "")
            if not raw:
                continue
            try:
                m50 = float(raw)
                m5095 = float(row.get("val/mAP_50_95", "nan") or "nan")
            except ValueError:
                continue
            if best50 != best50 or m50 > best50:  # first hit, or a new best
                best50, best5095 = m50, m5095
    return best50, best5095


if __name__ == "__main__":
    run_survey()

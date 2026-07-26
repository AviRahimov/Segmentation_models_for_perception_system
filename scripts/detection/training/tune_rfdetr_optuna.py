#!/usr/bin/env python3
"""Optuna (TPE) hyperparameter search for rfdetr-m on Detection_Dataset_hardneg.

Why bypass RFDETR.train()
--------------------------
Pruning a clearly-bad trial mid-training needs a PyTorch Lightning callback
(``PyTorchLightningPruningCallback``) hooked into the trainer. ``RFDETR.train()``
cannot do this -- its own docstring says so explicitly: the legacy ``callbacks``
dict kwarg is deprecated/discarded, and callers are told to "use PTL Callback
objects passed via build_trainer instead". So this script mirrors
``RFDETR.train()``'s own internal sequence (rfdetr/detr.py) one level down:
build a TrainConfig via ``get_train_config()``, align num_classes, construct
``RFDETRModelModule``/``RFDETRDataModule``/``build_trainer`` directly, then
attach the pruning callback before calling ``trainer.fit()``.

One subtlety this depends on: ``build_trainer(train_config, model_config,
**trainer_kwargs)`` builds its own internal callback list (EMA, checkpointing,
early stopping, COCOEvalCallback, ...) and does a plain ``dict.update()`` of
that config with ``trainer_kwargs`` before constructing ``Trainer(**cfg)`` --
so passing ``callbacks=[...]`` as a kwarg to ``build_trainer`` would silently
REPLACE RF-DETR's own callback stack rather than add to it. Instead, this
script calls ``build_trainer()`` with no ``callbacks`` kwarg (letting it
assemble its full normal stack) and appends the pruning callback to the
returned ``Trainer.callbacks`` list afterward -- confirmed to be a plain
mutable list PTL reads from, not a snapshot.

The pruning callback reads ``trainer.callback_metrics["val/mAP_50"]`` in its
own ``on_validation_end`` hook, which PTL always calls after every
callback's ``on_validation_epoch_end`` for that same epoch -- so by the time
it runs, ``COCOEvalCallback`` (registered by ``build_trainer``) has already
written that key for this epoch. Confirmed by reading
``rfdetr/training/callbacks/coco_eval.py`` directly.

Search space (all real TrainConfig fields, confirmed by dumping
``model._train_config_class.model_fields`` on this venv's rfdetr==1.7.1 --
NOT the same field set as the main venv's older rfdetr==1.5.2. Notably there
is no ``focal_alpha``/``bbox_loss_coef``/``giou_loss_coef`` in this version;
classification loss defaults to IA-BCE (``ia_bce_loss=True``), not focal loss):
lr, lr_encoder, lr_vit_layer_decay, warmup_epochs, weight_decay, drop_path,
ema_decay, cls_loss_coef, aug_config (categorical over none/conservative/
aggressive presets from rfdetr.datasets.aug_config).

Per-trial epoch budget defaults to 100 -- matching the production
``conservative_aug`` recipe exactly (see reports/detection/experiments.jsonl)
so Phase 5's later "Optuna alone vs. baseline" comparison is apples-to-apples.
Use --epochs for a fast smoke test before committing to a real sweep.

Usage
-----
    # Smoke test first (~a few minutes):
    python scripts/detection/training/tune_rfdetr_optuna.py --n-trials 2 --epochs 3

    # Real sweep:
    python scripts/detection/training/tune_rfdetr_optuna.py --n-trials 25

Output: reports/detection/optuna_rfdetr_m.md (leaderboard of all trials) +
weights/detection/rfdetr-m/optuna/best_params.json (for Phase 5's retrain).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]

# Must happen before the first `import rfdetr` -- same circular-import
# workaround train_detector_rfdetr.py uses for XL/2XL plus-variant detection.
# Harmless no-op for rfdetr-m, which doesn't need rfdetr_plus.
try:
    import rfdetr_plus  # noqa: F401
except ImportError:
    pass

import optuna  # noqa: E402
import rfdetr as rfdetr_pkg  # noqa: E402
import torch  # noqa: E402
from pytorch_lightning import Callback  # noqa: E402
from rfdetr.datasets.aug_config import AUG_AGGRESSIVE, AUG_CONSERVATIVE  # noqa: E402
from rfdetr.detr import _ensure_model_on_device  # noqa: E402
from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer  # noqa: E402
from rfdetr.training.auto_batch import resolve_auto_batch_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("tune_rfdetr_optuna")
optuna.logging.set_verbosity(optuna.logging.WARNING)  # PTL's own per-epoch tables are noisy enough

_DEFAULT_DATASET = _ROOT / "datasets/Detection_Dataset_hardneg"
_OUT_ROOT = _ROOT / "weights/detection/rfdetr-m/optuna"

# AUG_AERIAL/AUG_INDUSTRIAL omitted -- neither preset fits this off-road
# ground-vehicle/person domain (see aug_config.py's own preset table).
_AUG_PRESETS = {"none": None, "conservative": AUG_CONSERVATIVE, "aggressive": AUG_AGGRESSIVE}


class _PruningCallback(Callback):
    """Minimal single-process Optuna pruning callback for a PTL Trainer.

    optuna_integration.PyTorchLightningPruningCallback imports `lightning.pytorch`
    (the newer unified `lightning` package) rather than the standalone
    `pytorch_lightning` package RF-DETR actually depends on -- confirmed by
    running it here (ImportError: No module named 'lightning'). Installing the
    unified `lightning` package just to get one callback risks pulling in a
    second copy of the PTL stack under a different import path, so this
    reimplements the (DDP-agnostic; this project trains single-GPU) core of
    that callback directly against `pytorch_lightning.Callback` instead. Logic
    mirrors optuna_integration's on_validation_end body 1:1."""

    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:
        super().__init__()
        self._trial = trial
        self._monitor = monitor

    def on_validation_end(self, trainer, pl_module) -> None:  # noqa: ANN001
        if trainer.sanity_checking:
            return
        score = trainer.callback_metrics.get(self._monitor)
        if score is None:
            return
        epoch = pl_module.current_epoch
        self._trial.report(score.item(), step=epoch)
        if self._trial.should_prune():
            raise optuna.TrialPruned(f"Trial was pruned at epoch {epoch}.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    p.add_argument("--n-trials", type=int, default=25)
    p.add_argument("--epochs", type=int, default=100,
                   help="Per-trial epoch budget. Matches production's conservative_aug recipe by "
                        "default so later comparisons are apples-to-apples; lower for a smoke test.")
    p.add_argument("--batch-size", default="auto")
    p.add_argument("--eval-interval", type=int, default=1)
    p.add_argument("--pruner-warmup-epochs", type=int, default=10,
                   help="Trials are not eligible for pruning until this many epochs have completed")
    p.add_argument("--out", type=Path, default=Path("reports/detection/optuna_rfdetr_m.md"))
    p.add_argument("--study-db", type=Path, default=_OUT_ROOT / "study.db",
                   help="SQLite storage -- study survives a crash/Ctrl-C and resumes from the same path")
    return p.parse_args()


def _read_final_metrics(out_dir: Path) -> tuple[float, float]:
    """Best (not merely last) val/mAP_50 and val/mAP_50_95 from PTL's CSVLogger
    output. Duplicated from train_detector_rfdetr.py rather than imported --
    that script is deliberately zero-dependency on other project modules (see
    its own module docstring), and this one follows the same convention."""
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
            if best50 != best50 or m50 > best50:
                best50, best5095 = m50, m5095
    return best50, best5095


def _suggest_hparams(trial: optuna.trial.Trial) -> dict:
    aug_name = trial.suggest_categorical("aug_preset", list(_AUG_PRESETS))
    return {
        "lr": trial.suggest_float("lr", 3e-5, 3e-4, log=True),
        "lr_encoder": trial.suggest_float("lr_encoder", 3e-5, 3e-4, log=True),
        "lr_vit_layer_decay": trial.suggest_float("lr_vit_layer_decay", 0.6, 0.95),
        "warmup_epochs": trial.suggest_float("warmup_epochs", 0.0, 5.0),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "drop_path": trial.suggest_float("drop_path", 0.0, 0.3),
        "ema_decay": trial.suggest_float("ema_decay", 0.985, 0.9995),
        "cls_loss_coef": trial.suggest_float("cls_loss_coef", 0.5, 3.0),
        "aug_config": _AUG_PRESETS[aug_name],
    }


def _objective(trial: optuna.trial.Trial, args: argparse.Namespace) -> float:
    out_dir = _OUT_ROOT / f"trial_{trial.number:03d}"
    if out_dir.exists():
        shutil.rmtree(out_dir)  # a re-run of this trial number must not see a stale metrics.csv

    hp = _suggest_hparams(trial)
    logger.info("Trial %d params: %s", trial.number,
                {k: v for k, v in hp.items() if k != "aug_config"} | {"aug_preset": trial.params["aug_preset"]})

    model = rfdetr_pkg.RFDETRMedium()
    config = model.get_train_config(
        dataset_dir=str(args.dataset),
        dataset_file="yolo",
        output_dir=str(out_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=42,
        progress_bar=None,
        early_stopping=False,  # the Optuna pruner replaces RF-DETR's own patience-based early stop
        eval_interval=args.eval_interval,
        **hp,
    )
    if config.batch_size == "auto":
        # RFDETR.train() resolves "auto" into concrete batch_size/grad_accum_steps
        # via a memory probe before constructing RFDETRModelModule -- this
        # bypass path skips RFDETR.train() entirely, so replicate that step
        # here (rfdetr/detr.py:614-631) or module_data.py's train_dataloader()
        # crashes trying to compare an int against the literal string "auto".
        _ensure_model_on_device(model.model)
        auto_batch = resolve_auto_batch_config(
            model_context=model.model, model_config=model.model_config, train_config=config,
        )
        config.batch_size = auto_batch.safe_micro_batch
        config.grad_accum_steps = auto_batch.recommended_grad_accum_steps
    model._align_num_classes_from_dataset(str(args.dataset))
    module = RFDETRModelModule(model.model_config, config)
    datamodule = RFDETRDataModule(model.model_config, config)
    trainer = build_trainer(config, model.model_config)
    trainer.callbacks.append(_PruningCallback(trial, monitor="val/mAP_50"))

    try:
        trainer.fit(module, datamodule)
    finally:
        del model, module, datamodule, trainer
        torch.cuda.empty_cache()

    map50, map5095 = _read_final_metrics(out_dir)
    if map50 != map50:
        raise optuna.TrialPruned(f"trial {trial.number} produced no metrics.csv rows")
    trial.set_user_attr("map50_95", map5095)
    logger.info("Trial %d done: mAP50=%.4f  mAP50-95=%.4f", trial.number, map50, map5095)
    return map50


def main() -> int:
    args = parse_args()
    args.dataset = args.dataset if args.dataset.is_absolute() else _ROOT / args.dataset
    if isinstance(args.batch_size, str) and args.batch_size.isdigit():
        args.batch_size = int(args.batch_size)
    _OUT_ROOT.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name="rfdetr_m_hardneg",
        storage=f"sqlite:///{args.study_db}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=args.pruner_warmup_epochs),
    )
    study.optimize(lambda t: _objective(t, args), n_trials=args.n_trials)

    logger.info("Best trial #%d: mAP50=%.4f  params=%s",
                study.best_trial.number, study.best_value, study.best_params)

    out_path = args.out if args.out.is_absolute() else _ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_map5095 = study.best_trial.user_attrs.get("map50_95", float("nan"))
    lines = [
        "# Optuna hyperparameter search — rfdetr-m\n\n",
        f"> dataset=`{args.dataset.relative_to(_ROOT)}` epochs/trial={args.epochs} "
        f"n_trials={len(study.trials)} pruner=MedianPruner(warmup={args.pruner_warmup_epochs})\n\n",
        f"Best trial: **#{study.best_trial.number}**  mAP50={study.best_value:.4f}  "
        f"mAP50-95={best_map5095:.4f}\n\n",
        "## Best hyperparameters\n\n```json\n" + json.dumps(study.best_params, indent=2) + "\n```\n\n",
        "## All trials\n\n| # | state | mAP50 | mAP50-95 | params |\n|---|---|---|---|---|\n",
    ]
    for t in study.trials:
        map50_s = f"{t.value:.4f}" if t.value is not None else "—"
        m5095 = t.user_attrs.get("map50_95", float("nan"))
        map5095_s = f"{m5095:.4f}" if m5095 == m5095 else "—"
        lines.append(f"| {t.number} | {t.state.name} | {map50_s} | {map5095_s} | `{t.params}` |\n")
    out_path.write_text("".join(lines))

    best_json = _OUT_ROOT / "best_params.json"
    best_json.write_text(json.dumps({
        "trial": study.best_trial.number,
        "map50": study.best_value,
        "map50_95": best_map5095,
        "params": study.best_params,
    }, indent=2))
    logger.info("Report -> %s", out_path)
    logger.info("Best params -> %s", best_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

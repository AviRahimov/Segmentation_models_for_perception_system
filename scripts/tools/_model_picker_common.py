"""Shared helpers for scripts that run BOTH the instance (detection) and
semantic (segmentation) models together — render_samples.py, annotate_images.py.

Reuses scripts/segmentation/evaluation/_semantic_eval_common.py's checkpoint
scanner/interactive-picker for the semantic side; adds an equivalent scanner
for detection checkpoints (mirrors scripts/detection/training/_survey_common.py's
_scan_checkpoints(), kept self-contained rather than cross-imported, matching
this repo's existing convention of independent per-area _*_common.py helpers).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO / "scripts" / "segmentation" / "evaluation"))

from _semantic_eval_common import (  # noqa: E402
    BASELINE_CHOICES,
    ask_choice,
    parse_model_spec,
    resolve_weights,
    scan_semantic_checkpoints,
)


@dataclass(frozen=True)
class DetectionCheckpointChoice:
    key: str        # factory key to build with (top-level weights/detection/<key>/ dirname)
    weights: str    # local .pt path, or "" for the key's factory default
    label: str      # human-readable recipe description


def scan_detection_checkpoints(weights_root: Path) -> list[DetectionCheckpointChoice]:
    """Walk ``weights_root/**/best.pt`` — unlike segmentation, the top-level
    dirname under weights/detection/ IS the factory key directly (rfdetr-m,
    yolo11m, yolo26m, ...), no name-inference needed."""
    choices: list[DetectionCheckpointChoice] = []
    if not weights_root.is_dir():
        return choices
    for ckpt in sorted(weights_root.glob("**/best.pt")):
        if ckpt.parent.name == "weights":  # skip Ultralytics' inner weights/ copies
            continue
        rel = ckpt.relative_to(weights_root)
        key = rel.parts[0]
        label = "/".join(rel.parts[:-1])
        choices.append(DetectionCheckpointChoice(key=key, weights=str(ckpt), label=label))
    return choices


def resolve_instance_weights(key: str, explicit_weights: str, cfg) -> str:
    """Weights for an instance-model factory key: explicit override > config.yaml's
    active model override (if this key is the currently-configured one) > the
    factory's own default."""
    from perception.models.factory import INSTANCE_DEFAULT_WEIGHTS

    if explicit_weights:
        return explicit_weights
    lk = key.lower().strip()
    if lk == cfg.models.instance.name.lower().strip() and cfg.models.instance.weights:
        return cfg.models.instance.weights
    return INSTANCE_DEFAULT_WEIGHTS.get(lk, "")


def _resolve_against(repo_root: Path, raw: str) -> str:
    """Resolve a (possibly relative, possibly empty) path string against
    repo_root so it can be compared against scanner-produced absolute paths."""
    if not raw:
        return ""
    p = Path(raw)
    return str((repo_root / p).resolve() if not p.is_absolute() else p.resolve())


def _best_default_idx(choices, active_key: str, active_weights_resolved: str) -> int:
    """Prefer an exact (key, weights) match; fall back to the first matching
    key if the active weights aren't one of the scanned checkpoints."""
    key_matches = [i for i, c in enumerate(choices) if c.key.lower().strip() == active_key]
    if active_weights_resolved:
        exact = [i for i in key_matches if str(Path(choices[i].weights).resolve()) == active_weights_resolved]
        if exact:
            return exact[0]
    return key_matches[0] if key_matches else 0


def pick_instance_and_semantic_model_specs(cfg, repo_root: Path) -> tuple[str, str]:
    """Interactively choose one instance-model spec and one semantic-model
    spec (each ``"key"`` or ``"key:weights_path"``), defaulting to whatever
    is currently active in config.yaml."""
    det_choices = scan_detection_checkpoints(repo_root / "weights" / "detection")
    det_options = [
        (f"{c.key}:{c.weights}" if c.weights else c.key, f"{c.key}  [{c.label}]")
        for c in det_choices
    ]
    if not det_options:
        raise SystemExit(f"No detection checkpoints found under {repo_root / 'weights' / 'detection'}.")
    active_instance = cfg.models.instance.name.lower().strip()
    active_instance_weights = _resolve_against(repo_root, cfg.models.instance.weights or "")
    det_default_idx = _best_default_idx(det_choices, active_instance, active_instance_weights)
    instance_spec = ask_choice("Which instance/detection model?", det_options, default_idx=det_default_idx)

    sem_choices = list(scan_semantic_checkpoints(repo_root / "weights" / "segmentation" / "orfd")) + list(BASELINE_CHOICES)
    sem_options = [
        (f"{c.key}:{c.weights}" if c.weights else c.key, f"{c.key}  [{c.label}]")
        for c in sem_choices
    ]
    if not sem_options:
        raise SystemExit("No semantic checkpoints/baselines found.")
    active_semantic = cfg.models.semantic.name.lower().strip()
    active_semantic_weights = _resolve_against(repo_root, cfg.models.semantic.weights or "")
    sem_default_idx = _best_default_idx(sem_choices, active_semantic, active_semantic_weights)
    semantic_spec = ask_choice("Which semantic/segmentation model?", sem_options, default_idx=sem_default_idx)

    return instance_spec, semantic_spec


__all__ = [
    "DetectionCheckpointChoice",
    "scan_detection_checkpoints",
    "pick_instance_and_semantic_model_specs",
    "parse_model_spec",
    "resolve_weights",
    "resolve_instance_weights",
    "scan_semantic_checkpoints",
    "BASELINE_CHOICES",
]

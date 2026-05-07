"""YAML loader and validator for :class:`AppConfig`.

Validation is intentionally strict: bad config fails fast with a clear
message rather than silently producing degraded inference output.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

from .schema import (
    AppConfig,
    ClassDef,
    DatasetsCfg,
    HardwareCfg,
    InstanceModelCfg,
    InstanceSAM2Cfg,
    ModelsCfg,
    PlayerCfg,
    SemanticEMACfg,
    SemanticModelCfg,
    SourceCfg,
    TemporalCfg,
    VALID_DISPLAY_MODES,
    VALID_SOURCE_TYPES,
)


class ConfigError(ValueError):
    """Raised when the YAML config is malformed or contradictory."""


def load_config(path: str | Path) -> AppConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    raw = yaml.safe_load(p.read_text()) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"Top-level YAML must be a mapping, got {type(raw).__name__}")
    return _build_app_config(raw)


def override_source(
    cfg: AppConfig,
    *,
    source_type: str | None = None,
    path: str | None = None,
    camera: int | None = None,
    image_dir_glob: str | None = None,
) -> AppConfig:
    """Return a new :class:`AppConfig` with overridden source fields."""
    src = cfg.source
    new = dataclasses.replace(
        src,
        type=source_type or src.type,
        path=path if path is not None else src.path,
        camera_index=camera if camera is not None else src.camera_index,
        image_dir_glob=image_dir_glob or src.image_dir_glob,
    )
    if new.type not in VALID_SOURCE_TYPES:
        raise ConfigError(f"Invalid source.type override: {new.type!r}")
    return dataclasses.replace(cfg, source=new)


# --------------------------------------------------------------------------- #
# Internal builders                                                            #
# --------------------------------------------------------------------------- #


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"Section {name!r} must be a mapping, got {type(value).__name__}")
    return value


def _build_app_config(raw: dict[str, Any]) -> AppConfig:
    classes = _build_classes(raw.get("classes"))
    return AppConfig(
        models=_build_models(_require_dict(raw.get("models"), "models")),
        classes=classes,
        temporal=_build_temporal(_require_dict(raw.get("temporal"), "temporal")),
        hardware=_build_hardware(_require_dict(raw.get("hardware"), "hardware")),
        player=_build_player(_require_dict(raw.get("player"), "player")),
        source=_build_source(_require_dict(raw.get("source"), "source")),
        datasets=_build_datasets(_require_dict(raw.get("datasets"), "datasets")),
    )


def _build_classes(raw: Any) -> tuple[ClassDef, ...]:
    if not raw:
        raise ConfigError("'classes' must contain at least one entry")
    if not isinstance(raw, list):
        raise ConfigError("'classes' must be a list of class entries")

    seen: set[str] = set()
    out: list[ClassDef] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ConfigError(f"classes[{i}] must be a mapping")
        try:
            name = str(entry["name"])
            text_prompt = str(entry["text_prompt"])
        except KeyError as e:
            raise ConfigError(f"classes[{i}] missing required field: {e.args[0]!r}") from None
        if name in seen:
            raise ConfigError(f"Duplicate class name: {name!r}")
        seen.add(name)

        display_mode = str(entry.get("display_mode", "both"))
        if display_mode not in VALID_DISPLAY_MODES:
            raise ConfigError(
                f"classes[{i}={name!r}].display_mode={display_mode!r} not in {sorted(VALID_DISPLAY_MODES)}"
            )

        color = entry.get("color_rgb", [255, 0, 0])
        if not isinstance(color, (list, tuple)) or len(color) != 3:
            raise ConfigError(f"classes[{i}={name!r}].color_rgb must be a 3-element list, got {color!r}")
        try:
            color_t = tuple(int(c) for c in color)
        except (TypeError, ValueError):
            raise ConfigError(f"classes[{i}={name!r}].color_rgb must contain integers") from None
        if any(not 0 <= c <= 255 for c in color_t):
            raise ConfigError(f"classes[{i}={name!r}].color_rgb values must be in [0, 255]")

        is_semantic = bool(entry.get("is_semantic", False))
        ade = entry.get("ade20k_indices", ())
        if isinstance(ade, int):
            ade = [ade]
        try:
            ade_t = tuple(int(x) for x in ade)
        except (TypeError, ValueError):
            raise ConfigError(f"classes[{i}={name!r}].ade20k_indices must be ints") from None

        if is_semantic and not ade_t:
            raise ConfigError(
                f"Semantic class {name!r} must define ade20k_indices "
                "(SegFormer-B2 is closed-vocabulary on ADE20K)"
            )
        if not is_semantic and ade_t:
            # Soft warning would be nice, but a strict error is safer.
            raise ConfigError(
                f"Instance class {name!r} must NOT define ade20k_indices"
            )

        cls_conf_raw = entry.get("confidence_threshold", None)
        cls_conf: float | None
        if cls_conf_raw is None:
            cls_conf = None
        else:
            try:
                cls_conf = float(cls_conf_raw)
            except (TypeError, ValueError):
                raise ConfigError(
                    f"classes[{i}={name!r}].confidence_threshold must be a number, "
                    f"got {cls_conf_raw!r}"
                ) from None
            if not 0.0 <= cls_conf <= 1.0:
                raise ConfigError(
                    f"classes[{i}={name!r}].confidence_threshold must be in [0, 1], "
                    f"got {cls_conf}"
                )
            if is_semantic:
                raise ConfigError(
                    f"Semantic class {name!r} must NOT define confidence_threshold "
                    "(SegFormer has no per-class score; argmax is unconditional)"
                )

        out.append(
            ClassDef(
                name=name,
                text_prompt=text_prompt,
                display_mode=display_mode,  # type: ignore[arg-type]
                color_rgb=color_t,  # type: ignore[arg-type]
                is_semantic=is_semantic,
                ade20k_indices=ade_t,
                confidence_threshold=cls_conf,
            )
        )
    return tuple(out)


def _build_models(raw: dict[str, Any]) -> ModelsCfg:
    inst_raw = _require_dict(raw.get("instance"), "models.instance")
    sem_raw = _require_dict(raw.get("semantic"), "models.semantic")
    inst = InstanceModelCfg(
        name=str(inst_raw.get("name", "yoloe26l")),
        confidence_threshold=float(inst_raw.get("confidence_threshold", 0.35)),
        weights=inst_raw.get("weights"),
    )
    sem = SemanticModelCfg(
        name=str(sem_raw.get("name", "segformer-b2")),
        weights=str(sem_raw.get("weights", "nvidia/segformer-b2-finetuned-ade-512-512")),
    )
    if not 0.0 <= inst.confidence_threshold <= 1.0:
        raise ConfigError(
            f"models.instance.confidence_threshold must be in [0, 1], got {inst.confidence_threshold}"
        )
    return ModelsCfg(instance=inst, semantic=sem)


def _build_temporal(raw: dict[str, Any]) -> TemporalCfg:
    sem = _require_dict(raw.get("semantic_ema"), "temporal.semantic_ema")
    sam = _require_dict(raw.get("instance_sam2"), "temporal.instance_sam2")
    sem_cfg = SemanticEMACfg(
        alpha=float(sem.get("alpha", 0.35)),
        reset_on_scene_cut=bool(sem.get("reset_on_scene_cut", True)),
        scene_cut_threshold=float(sem.get("scene_cut_threshold", 0.45)),
    )
    if not 0.0 < sem_cfg.alpha <= 1.0:
        raise ConfigError(f"temporal.semantic_ema.alpha must be in (0, 1], got {sem_cfg.alpha}")
    if not 0.0 <= sem_cfg.scene_cut_threshold <= 1.0:
        raise ConfigError(
            f"temporal.semantic_ema.scene_cut_threshold must be in [0, 1], got {sem_cfg.scene_cut_threshold}"
        )
    sam_cfg = InstanceSAM2Cfg(
        enabled=bool(sam.get("enabled", True)),
        reprompt_every_n_frames=int(sam.get("reprompt_every_n_frames", 30)),
        min_track_score=float(sam.get("min_track_score", 0.4)),
        checkpoint=str(sam.get("checkpoint", "") or ""),
        model_config=str(sam.get("model_config", "") or ""),
    )
    if sam_cfg.reprompt_every_n_frames < 1:
        raise ConfigError("temporal.instance_sam2.reprompt_every_n_frames must be >= 1")
    return TemporalCfg(semantic_ema=sem_cfg, instance_sam2=sam_cfg)


def _build_hardware(raw: dict[str, Any]) -> HardwareCfg:
    return HardwareCfg(
        device=str(raw.get("device", "cuda")),
        fp16=bool(raw.get("fp16", True)),
        use_tensorrt=bool(raw.get("use_tensorrt", False)),
        text_embed_cache=bool(raw.get("text_embed_cache", True)),
    )


def _build_player(raw: dict[str, Any]) -> PlayerCfg:
    cfg = PlayerCfg(
        mask_alpha=float(raw.get("mask_alpha", 0.45)),
        show_fps=bool(raw.get("show_fps", True)),
        show_class_legend=bool(raw.get("show_class_legend", True)),
        default_speed=float(raw.get("default_speed", 1.0)),
    )
    if not 0.0 <= cfg.mask_alpha <= 1.0:
        raise ConfigError(f"player.mask_alpha must be in [0, 1], got {cfg.mask_alpha}")
    if cfg.default_speed <= 0.0:
        raise ConfigError(f"player.default_speed must be > 0, got {cfg.default_speed}")
    return cfg


def _build_source(raw: dict[str, Any]) -> SourceCfg:
    t = str(raw.get("type", "video"))
    if t not in VALID_SOURCE_TYPES:
        raise ConfigError(f"source.type must be one of {sorted(VALID_SOURCE_TYPES)}, got {t!r}")
    cfg = SourceCfg(
        type=t,  # type: ignore[arg-type]
        path=raw.get("path"),
        camera_index=int(raw.get("camera_index", 0)),
        image_dir_glob=str(raw.get("image_dir_glob", "*.png")),
        fps_hint=float(raw.get("fps_hint", 30.0)),
    )
    return cfg


def _build_datasets(raw: dict[str, Any]) -> DatasetsCfg:
    return DatasetsCfg(download_dir=str(raw.get("download_dir", "./datasets")))

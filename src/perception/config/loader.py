"""YAML loader and validator for :class:`AppConfig`.

Validation is intentionally strict: bad config fails fast with a clear
message rather than silently producing degraded inference output.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

from ..core.colors_named import _NAMED_COLORS, parse_color
from ..models.semantic._class_catalogues import GOOSE_12_NAMES
from ..models.semantic._class_catalogues import CATALOGUE_SIZES
from .schema import (
    AppConfig,
    ClassDef,
    DatasetsCfg,
    HardwareCfg,
    InstanceModelCfg,
    ModelsCfg,
    OrfdSemanticComparisonCfg,
    OrfdSemanticComparisonGooseCfg,
    OrfdSemanticComparisonInstanceMaskCfg,
    PlayerCfg,
    SemanticEMACfg,
    SemanticModelCfg,
    SourceCfg,
    TemporalCfg,
    VALID_DISPLAY_MODES,
    VALID_NATIVE_CATALOGUES,
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
        orfd_semantic_comparison=_build_orfd_semantic_comparison(
            _require_dict(raw.get("orfd_semantic_comparison"), "orfd_semantic_comparison"),
        ),
    )


def _build_orfd_semantic_comparison(raw: dict[str, Any]) -> OrfdSemanticComparisonCfg:
    grey = raw.get("orfd_trav_gray", 255)
    if not isinstance(grey, int) or grey < 0 or grey > 255:
        raise ConfigError(f"orfd_semantic_comparison.orfd_trav_gray must be int in [0,255], got {grey!r}")

    tau = raw.get("freespace_merged_prob_floor", None)
    if tau is not None:
        if not isinstance(tau, (int, float)) or not (0.0 < float(tau) < 1.0):
            raise ConfigError(
                "orfd_semantic_comparison.freespace_merged_prob_floor must be null "
                "or float in (0, 1), "
                f"got {tau!r}",
            )
        tau_f: float | None = float(tau)
    else:
        tau_f = None

    goose_r = _require_dict(raw.get("goose"), "goose") if raw else {}
    tcats = goose_r.get("traversable_categories", ["terrain", "road"])
    if tcats is None:
        tcats_list: list[str] = []
    elif isinstance(tcats, (list, tuple)):
        tcats_list = []
        for i, item in enumerate(tcats):
            if not isinstance(item, str) or not item.strip():
                raise ConfigError(f"goose.traversable_categories[{i}] must be a non-empty string")
            tcats_list.append(item.strip().lower())
    elif isinstance(tcats, str):
        tcats_list = [x.strip().lower() for x in tcats.split(",") if x.strip()]
    else:
        raise ConfigError(f"goose.traversable_categories must be a list or string, got {type(tcats).__name__}")
    if not tcats_list:
        raise ConfigError("goose.traversable_categories must list at least one GOOSE coarse name")

    allowed = frozenset(GOOSE_12_NAMES)
    for c in tcats_list:
        if c not in allowed:
            raise ConfigError(
                f"goose.traversable_categories: unknown GOOSE-12 category {c!r} "
                f"(allowed: {sorted(allowed)})",
            )

    seen: set[str] = set()
    tcats_unique: list[str] = []
    for c in tcats_list:
        if c not in seen:
            seen.add(c)
            tcats_unique.append(c)

    g_samples = goose_r.get("samples", 0)
    if not isinstance(g_samples, int) or g_samples < 0:
        raise ConfigError(f"goose.samples must be int >= 0, got {g_samples!r}")

    inst_r = _require_dict(raw.get("instance_mask_subtraction"), "instance_mask_subtraction") if raw else {}
    subtract = bool(inst_r.get("subtract_from_traversable", False))
    dilate_px = inst_r.get("dilate_px", 5)
    if not isinstance(dilate_px, int) or dilate_px < 0:
        raise ConfigError(f"instance_mask_subtraction.dilate_px must be int >= 0, got {dilate_px!r}")

    goose_cfg = OrfdSemanticComparisonGooseCfg(
        ex_root=str(goose_r.get(
            "ex_root",
            OrfdSemanticComparisonGooseCfg.ex_root,
        )),
        label_csv=str(goose_r.get(
            "label_csv",
            OrfdSemanticComparisonGooseCfg.label_csv,
        )),
        scenario_dir=str(goose_r.get(
            "scenario_dir",
            OrfdSemanticComparisonGooseCfg.scenario_dir,
        )),
        samples=int(g_samples),
        traversable_categories=tuple(tcats_unique),
    )
    inst_cfg = OrfdSemanticComparisonInstanceMaskCfg(
        subtract_from_traversable=subtract,
        dilate_px=int(dilate_px),
    )

    return OrfdSemanticComparisonCfg(
        orfd_trav_gray=int(grey),
        freespace_merged_prob_floor=tau_f,
        goose=goose_cfg,
        instance_mask_subtraction=inst_cfg,
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

        color_t = _parse_class_color(entry, i, name)

        is_semantic = bool(entry.get("is_semantic", False))
        native_idx = _parse_native_indices(entry, i, name)
        ade_t = native_idx.get("ade20k", ())

        if is_semantic and not any(native_idx.values()):
            raise ConfigError(
                f"Semantic class {name!r} must define at least one non-empty "
                "entry under native_indices (or the legacy ade20k_indices "
                "shorthand) - closed-vocab models cannot be served without it"
            )
        if not is_semantic and (native_idx or "ade20k_indices" in entry or "native_indices" in entry):
            # Soft warning would be nice, but a strict error is safer.
            raise ConfigError(
                f"Instance class {name!r} must NOT define ade20k_indices "
                "or native_indices"
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
                native_indices=native_idx,
                confidence_threshold=cls_conf,
            )
        )
    return tuple(out)


# --------------------------------------------------------------------------- #
# Color and native-indices helpers                                            #
# --------------------------------------------------------------------------- #


def _parse_class_color(entry: dict[str, Any], i: int, name: str) -> tuple[int, int, int]:
    """Parse the ``color`` (or legacy ``color_rgb``) field for one class.

    The on-disk format is ``color: "<name-or-hex>"``. The legacy
    ``color_rgb: [R, G, B]`` array form is rejected with a clear
    deprecation message — silently coercing is too easy to leave wrong.
    """
    if "color_rgb" in entry:
        raise ConfigError(
            f"classes[{i}={name!r}].color_rgb is deprecated; use "
            f'color: "<name>" or color: "#RRGGBB". '
            f"Valid names: {sorted(_NAMED_COLORS)}"
        )
    spec = entry.get("color", "red")
    if not isinstance(spec, str):
        raise ConfigError(
            f"classes[{i}={name!r}].color must be a string "
            f'(named color or "#RRGGBB" hex), got {type(spec).__name__}: {spec!r}'
        )
    try:
        return parse_color(spec)
    except ValueError as e:
        raise ConfigError(f"classes[{i}={name!r}].color: {e}") from None


def _parse_native_indices(
    entry: dict[str, Any], i: int, name: str
) -> dict[str, tuple[int, ...]]:
    """Parse ``native_indices: {ade20k: [...], goose_12: [...]}`` plus the
    legacy ``ade20k_indices: [...]`` shorthand.

    Returns a fresh dict so callers can store it on the (frozen) dataclass
    without aliasing across instances.
    """
    out: dict[str, tuple[int, ...]] = {}

    legacy = entry.get("ade20k_indices", None)
    if legacy is not None:
        if isinstance(legacy, int):
            legacy = [legacy]
        try:
            out["ade20k"] = tuple(int(x) for x in legacy)
        except (TypeError, ValueError):
            raise ConfigError(
                f"classes[{i}={name!r}].ade20k_indices must be ints"
            ) from None
        _validate_catalogue_range("ade20k", out["ade20k"], i, name)

    native = entry.get("native_indices", None)
    if native is not None:
        if not isinstance(native, dict):
            raise ConfigError(
                f"classes[{i}={name!r}].native_indices must be a mapping, "
                f"got {type(native).__name__}"
            )
        for key, raw in native.items():
            k = str(key)
            if k not in VALID_NATIVE_CATALOGUES:
                raise ConfigError(
                    f"unknown native_indices key {k!r}; allowed: "
                    "{'ade20k','goose_12'}"
                )
            if isinstance(raw, int):
                raw = [raw]
            if raw is None:
                raw = []
            try:
                idx = tuple(int(x) for x in raw)
            except (TypeError, ValueError):
                raise ConfigError(
                    f"classes[{i}={name!r}].native_indices[{k!r}] must "
                    "contain ints"
                ) from None
            _validate_catalogue_range(k, idx, i, name)
            if k in out and out[k] != idx:
                raise ConfigError(
                    f"classes[{i}={name!r}] declares both ade20k_indices "
                    f"and native_indices.ade20k with different values"
                )
            out[k] = idx
    return out


def _validate_catalogue_range(
    catalogue: str, idx: tuple[int, ...], i: int, name: str
) -> None:
    n = CATALOGUE_SIZES[catalogue]
    for v in idx:
        if not 0 <= v < n:
            raise ConfigError(
                f"classes[{i}={name!r}].native_indices[{catalogue!r}] "
                f"value {v} out of range [0, {n})"
            )


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
        # Empty string => factory resolves the default weights for `name`.
        weights=str(sem_raw.get("weights", "") or ""),
    )
    if not 0.0 <= inst.confidence_threshold <= 1.0:
        raise ConfigError(
            f"models.instance.confidence_threshold must be in [0, 1], got {inst.confidence_threshold}"
        )
    return ModelsCfg(instance=inst, semantic=sem)


def _build_temporal(raw: dict[str, Any]) -> TemporalCfg:
    sem = _require_dict(raw.get("semantic_ema"), "temporal.semantic_ema")
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
    return TemporalCfg(semantic_ema=sem_cfg)


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
        draw_road_ground_semantic_last=bool(
            raw.get("draw_road_ground_semantic_last", True),
        ),
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

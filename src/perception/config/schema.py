"""Typed configuration dataclasses.

These dataclasses are *frozen* so they can be hashed and shared between
threads safely. To override a field at runtime use :func:`dataclasses.replace`
or the helper :func:`perception.config.loader.override_source`.

The schema deliberately introduces a small but important extension to the
example config provided in the project brief: every ``is_semantic: true``
class carries an ``ade20k_indices`` tuple. SegFormer-B2 is closed-vocabulary
on ADE20K (150 classes), so a free-form text prompt cannot itself create a
new semantic class. The ``ade20k_indices`` field tells the wrapper which
ADE20K classes to merge into a single user class (e.g. road + sidewalk +
earth -> ``road_ground``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

DisplayMode = Literal["both", "bbox_only", "mask_only", "none"]
SourceType = Literal["video", "camera", "image_dir"]

VALID_DISPLAY_MODES: frozenset[str] = frozenset({"both", "bbox_only", "mask_only", "none"})
VALID_SOURCE_TYPES: frozenset[str] = frozenset({"video", "camera", "image_dir"})


@dataclass(frozen=True)
class ClassDef:
    """A single user-facing class.

    Attributes:
        name:           Stable identifier used everywhere (e.g. ``"person"``).
        text_prompt:    Open-vocabulary prompt for the instance model and the
                        legend label for semantic classes.
        display_mode:   One of ``both | bbox_only | mask_only | none``.
        color_rgb:      Display color, RGB 0-255 tuple.
        is_semantic:    ``True`` if this class is produced by the semantic
                        model, ``False`` if produced by the instance model.
        ade20k_indices: For semantic classes only - tuple of ADE20K logit
                        channels to merge into this user class. Ignored for
                        instance classes.
        confidence_threshold:
                        Optional per-class confidence override for instance
                        classes. ``None`` means "fall back to
                        ``models.instance.confidence_threshold``". Must lie in
                        ``[0, 1]`` when set. Rejected for semantic classes
                        (SegFormer has no per-class score - the merged
                        argmax is unconditional).
    """

    name: str
    text_prompt: str
    display_mode: DisplayMode
    color_rgb: tuple[int, int, int]
    is_semantic: bool
    ade20k_indices: tuple[int, ...] = ()
    confidence_threshold: float | None = None


@dataclass(frozen=True)
class InstanceModelCfg:
    name: str = "yoloe26l"
    confidence_threshold: float = 0.35
    weights: str | None = None  # default per-model when None


@dataclass(frozen=True)
class SemanticModelCfg:
    name: str = "segformer-b2"
    weights: str = "nvidia/segformer-b2-finetuned-ade-512-512"


@dataclass(frozen=True)
class ModelsCfg:
    instance: InstanceModelCfg = field(default_factory=InstanceModelCfg)
    semantic: SemanticModelCfg = field(default_factory=SemanticModelCfg)


@dataclass(frozen=True)
class SemanticEMACfg:
    alpha: float = 0.35
    reset_on_scene_cut: bool = True
    scene_cut_threshold: float = 0.45  # Bhattacharyya distance in [0, 1]


@dataclass(frozen=True)
class InstanceSAM2Cfg:
    enabled: bool = True
    reprompt_every_n_frames: int = 30
    min_track_score: float = 0.4
    checkpoint: str = ""     # path to sam2 .pt; empty -> tracker is disabled
    model_config: str = ""   # path to sam2 yaml config


@dataclass(frozen=True)
class TemporalCfg:
    semantic_ema: SemanticEMACfg = field(default_factory=SemanticEMACfg)
    instance_sam2: InstanceSAM2Cfg = field(default_factory=InstanceSAM2Cfg)


@dataclass(frozen=True)
class HardwareCfg:
    device: str = "cuda"
    fp16: bool = True
    use_tensorrt: bool = False
    text_embed_cache: bool = True


@dataclass(frozen=True)
class PlayerCfg:
    mask_alpha: float = 0.45
    show_fps: bool = True
    show_class_legend: bool = True
    default_speed: float = 1.0


@dataclass(frozen=True)
class SourceCfg:
    type: SourceType = "video"
    path: str | None = None
    camera_index: int = 0
    image_dir_glob: str = "*.png"
    fps_hint: float = 30.0  # used for image_dir / camera fallback


@dataclass(frozen=True)
class DatasetsCfg:
    download_dir: str = "./datasets"


@dataclass(frozen=True)
class AppConfig:
    """Root configuration object."""

    models: ModelsCfg
    classes: tuple[ClassDef, ...]
    temporal: TemporalCfg
    hardware: HardwareCfg
    player: PlayerCfg
    source: SourceCfg
    datasets: DatasetsCfg

    @property
    def instance_classes(self) -> tuple[ClassDef, ...]:
        return tuple(c for c in self.classes if not c.is_semantic)

    @property
    def semantic_classes(self) -> tuple[ClassDef, ...]:
        return tuple(c for c in self.classes if c.is_semantic)

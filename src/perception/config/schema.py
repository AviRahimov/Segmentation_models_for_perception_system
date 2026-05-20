"""Typed configuration dataclasses.

These dataclasses are *frozen* so they can be shared between threads
safely. To override a field at runtime use :func:`dataclasses.replace`
or the helper :func:`perception.config.loader.override_source`.

Native catalogues
-----------------

Each ``is_semantic: true`` class carries a ``native_indices`` mapping
keyed by *catalogue name* (``"ade20k"`` or ``"goose_12"``) → tuple of
native channel indices to merge into that user class. Different model
wrappers consume different catalogues:

* :class:`SegFormerSemanticModel` (B2/B4) consumes ``"ade20k"`` (150 ch).
* :class:`DDRNetSemanticModel`   consumes ``"goose_12"`` (12 ch).
* :class:`PPLiteSegSemanticModel` consumes ``"goose_12"`` (12 ch).

A semantic user class must define **at least one** non-empty entry in
``native_indices`` for the loader to accept it. Whether a given wrapper
can actually serve that class is enforced at ``warmup()`` time per
model.

For backward compatibility with older configs the loader still accepts
the top-level ``ade20k_indices: [...]`` shorthand and routes it into
``native_indices["ade20k"]``. The :attr:`ClassDef.ade20k_indices`
property below preserves the old read API so existing tests keep
passing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

DisplayMode = Literal["both", "bbox_only", "mask_only", "none"]
SourceType = Literal["video", "camera", "image_dir"]
InstancePromptMode = Literal["production", "discovery"]
VALID_INSTANCE_PROMPT_MODES: frozenset[str] = frozenset({"production", "discovery"})

VALID_DISPLAY_MODES: frozenset[str] = frozenset({"both", "bbox_only", "mask_only", "none"})
VALID_SOURCE_TYPES: frozenset[str] = frozenset({"video", "camera", "image_dir"})

#: Allowlisted ``native_indices`` keys. ``"goose_64"`` is intentionally
#: omitted in this round (no 64-class checkpoint is wired up yet); add
#: it here when a future round needs it.
VALID_NATIVE_CATALOGUES: frozenset[str] = frozenset({"ade20k", "goose_12"})


@dataclass(frozen=True)
class ClassDef:
    """A single user-facing class.

    Attributes:
        name:           Stable identifier used everywhere (e.g. ``"person"``).
        text_prompt:    Open-vocabulary prompt for the instance model and the
                        legend label for semantic classes.
        display_mode:   One of ``both | bbox_only | mask_only | none``.
        color_rgb:      Display color as ``(R, G, B)`` 0-255. Set internally
                        by the loader from the YAML ``color: <name-or-hex>``
                        field; do NOT set this directly when constructing
                        configs by hand outside the loader.
        is_semantic:    ``True`` if this class is produced by the semantic
                        model, ``False`` if produced by the instance model.
        native_indices: For semantic classes only - mapping from native
                        catalogue name (``"ade20k"`` / ``"goose_12"``) to
                        the tuple of native channel indices to merge into
                        this user class. Must have at least one non-empty
                        entry for semantic classes. Ignored (and forbidden)
                        on instance classes.
        confidence_threshold:
                        Optional per-class confidence override for instance
                        classes. ``None`` means "fall back to
                        ``models.instance.confidence_threshold``". Must lie in
                        ``[0, 1]`` when set. Rejected for semantic classes
                        (closed-vocab models have no per-class score - the
                        merged argmax is unconditional).
    """

    name: str
    text_prompt: str
    display_mode: DisplayMode
    color_rgb: tuple[int, int, int]
    is_semantic: bool
    native_indices: dict[str, tuple[int, ...]] = field(default_factory=dict)
    confidence_threshold: float | None = None

    @property
    def ade20k_indices(self) -> tuple[int, ...]:
        """Backward-compat shim: return ``native_indices["ade20k"]`` or ``()``.

        Kept so that pre-migration call sites and existing tests continue
        to read ``cls.ade20k_indices`` without modification.
        """
        return self.native_indices.get("ade20k", ())


@dataclass(frozen=True)
class InstanceModelCfg:
    name: str = "yoloe26l"
    confidence_threshold: float = 0.35
    weights: str | None = None  # default per-model when None
    #: ``production`` uses ``classes[*].text_prompt`` for YOLOE ``set_classes``.
    #: ``discovery`` loads many prompts from ``discovery_vocabulary_path`` for exploration.
    prompt_mode: InstancePromptMode = "production"
    #: Absolute path after load (when ``prompt_mode == "discovery"``); empty in production.
    discovery_vocabulary_path: str = ""
    discovery_conf_floor: float = 0.05
    discovery_max_det: int | None = None
    #: Ultralytics inference image size (square). Lower = faster at cost of small-object recall.
    #: 640 is the YOLOE-26L default; 512 saves ~25% with negligible quality loss for large objects.
    imgsz: int = 640


@dataclass(frozen=True)
class SemanticModelCfg:
    name: str = "segformer-b2"
    #: Empty string ``""`` means "use the per-name default from
    #: :data:`perception.models.factory.SEMANTIC_DEFAULT_WEIGHTS`". Set
    #: this in YAML only when you want to override the standard checkpoint.
    weights: str = ""
    #: Number of output classes for a fine-tuned model. ``None`` means the
    #: standard closed-vocab mode (ADE20K-150 for SegFormer, GOOSE-12 for
    #: DDRNet) with LUT merging. When set to a positive integer the wrapper
    #: skips the LUT and uses the model's output channels directly; semantic
    #: classes in config serve as the ordered label list.
    num_classes: int | None = None
    #: SegFormer processor input resolution (square, pixels). ``None`` keeps
    #: the model default (512 for segformer-b2/b4). Set to 256 for ~2× speedup
    #: at the cost of slightly coarser class boundaries; 384 is a middle ground.
    #: Change in config.yaml only — no code changes needed to revert.
    processor_size: int | None = None


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
class InstanceTrackerCfg:
    iou_threshold: float = 0.30
    max_hold_frames: int = 2        # frames to re-emit a missed track (0 = disabled)
    hold_score_decay: float = 0.85  # per-missed-frame score multiplier
    bbox_alpha: float = 0.50        # EMA weight for bbox coords (1=raw, 0=frozen)
    score_alpha: float = 0.40       # EMA weight for displayed confidence


@dataclass(frozen=True)
class TemporalCfg:
    semantic_ema: SemanticEMACfg = field(default_factory=SemanticEMACfg)
    instance_tracker: InstanceTrackerCfg = field(default_factory=InstanceTrackerCfg)
    #: Run SegFormer every N frames; reuse the last EMA-smoothed result for skipped frames.
    #: 1 = every frame (no skipping). 2 = half cost, imperceptible on stable terrain.
    #: Scene cuts and seeks always force immediate SegFormer inference regardless of this value.
    semantic_skip_frames: int = 1


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
    draw_road_ground_semantic_last: bool = True


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
class OrfdSemanticComparisonGooseCfg:
    """GOOSE-Ex val extras for ``scripts/orfd_semantic_comparison.py``."""

    ex_root: str = "datasets/goose/gooseEx_2d_val/gooseEx_2d_val"
    label_csv: str = (
        "datasets/goose/gooseEx_2d_val/gooseEx_2d_val/goose_label_mapping.csv"
    )
    scenario_dir: str = "spot_scenario03"
    samples: int = 0
    traversable_categories: tuple[str, ...] = ("terrain", "road")


@dataclass(frozen=True)
class OrfdSemanticComparisonInstanceMaskCfg:
    subtract_from_traversable: bool = False
    dilate_px: int = 5


@dataclass(frozen=True)
class OrfdSemanticComparisonCfg:
    """Strip comparison harness (ORFD + optional GOOSE val frames)."""

    orfd_trav_gray: int = 255
    #: If set, traversable iff merged P(road_ground) >= this (otherwise argmax).
    freespace_merged_prob_floor: float | None = None
    goose: OrfdSemanticComparisonGooseCfg = field(default_factory=OrfdSemanticComparisonGooseCfg)
    instance_mask_subtraction: OrfdSemanticComparisonInstanceMaskCfg = field(
        default_factory=OrfdSemanticComparisonInstanceMaskCfg,
    )


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
    orfd_semantic_comparison: OrfdSemanticComparisonCfg = field(
        default_factory=OrfdSemanticComparisonCfg,
    )

    @property
    def instance_classes(self) -> tuple[ClassDef, ...]:
        return tuple(c for c in self.classes if not c.is_semantic)

    @property
    def semantic_classes(self) -> tuple[ClassDef, ...]:
        return tuple(c for c in self.classes if c.is_semantic)

    @property
    def runs_yoloe_instance_inference(self) -> bool:
        """True when the pipeline should run YOLOE (production classes or discovery vocab)."""
        if self.models.instance.prompt_mode == "discovery":
            return True
        return bool(self.instance_classes)

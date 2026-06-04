"""Factories that build instance and semantic models from config.

Adding a new model = register it here and ship a wrapper class that
implements the corresponding ABC. No other code needs to change.
"""
from __future__ import annotations

from typing import Type

from ..config.schema import HardwareCfg, InstanceModelCfg, InstancePromptMode, SemanticModelCfg
from .backends.base import InferenceBackend
from .instance.base import InstanceModel
from .instance.yoloe import YOLOEInstanceModel
from .semantic.base import SemanticModel
from .semantic.auriganet import AurigaNetSemanticModel
from .semantic.segformer import SegFormerSemanticModel


INSTANCE_REGISTRY: dict[str, Type[InstanceModel]] = {
    "yoloe26l": YOLOEInstanceModel,
    "yoloe-26l": YOLOEInstanceModel,
    "yoloe": YOLOEInstanceModel,
}

SEMANTIC_REGISTRY: dict[str, Type[SemanticModel]] = {
    "segformer-b0": SegFormerSemanticModel,
    "segformer_b0": SegFormerSemanticModel,
    "segformer-b1": SegFormerSemanticModel,
    "segformer_b1": SegFormerSemanticModel,
    "segformer-b2": SegFormerSemanticModel,
    "segformer_b2": SegFormerSemanticModel,
    "segformer": SegFormerSemanticModel,
    "segformer-b4": SegFormerSemanticModel,
    "segformer_b4": SegFormerSemanticModel,
    "auriganet": AurigaNetSemanticModel,
}

#: Per-key default weights — HuggingFace Hub repo IDs or local paths.
#: Override via ``models.semantic.weights`` in config.yaml.
SEMANTIC_DEFAULT_WEIGHTS: dict[str, str] = {
    "segformer-b0": "nvidia/segformer-b0-finetuned-ade-512-512",
    "segformer_b0": "nvidia/segformer-b0-finetuned-ade-512-512",
    "segformer-b1": "nvidia/segformer-b1-finetuned-ade-512-512",
    "segformer_b1": "nvidia/segformer-b1-finetuned-ade-512-512",
    "segformer-b2": "nvidia/segformer-b2-finetuned-ade-512-512",
    "segformer_b2": "nvidia/segformer-b2-finetuned-ade-512-512",
    "segformer":    "nvidia/segformer-b2-finetuned-ade-512-512",
    "segformer-b4": "nvidia/segformer-b4-finetuned-ade-512-512",
    "segformer_b4": "nvidia/segformer-b4-finetuned-ade-512-512",
    "auriganet":    "",  # no pretrained BDD100K weights published; random init until fine-tuned
}


def build_instance_model(
    cfg: InstanceModelCfg,
    hw: HardwareCfg,
    backend: InferenceBackend,
) -> InstanceModel:
    name = cfg.name.lower().strip()
    if name not in INSTANCE_REGISTRY:
        raise ValueError(
            f"Unknown instance model {cfg.name!r}. "
            f"Available: {sorted(INSTANCE_REGISTRY)}"
        )
    cls = INSTANCE_REGISTRY[name]
    # Default matches the project brief (YOLOE-26L); override via
    # `models.instance.weights` in config.yaml to pick a different size.
    weights = cfg.weights or "yoloe-26l-seg.pt"
    return cls(
        weights=weights,
        confidence_threshold=cfg.confidence_threshold,
        backend=backend,
        device=hw.device,
        fp16=hw.fp16,
        prompt_mode=cfg.prompt_mode,
        discovery_vocab_path=cfg.discovery_vocabulary_path,
        discovery_conf_floor=cfg.discovery_conf_floor,
        discovery_max_det=cfg.discovery_max_det,
    )


def build_semantic_model(
    cfg: SemanticModelCfg,
    hw: HardwareCfg,
    backend: InferenceBackend,
) -> SemanticModel:
    name = cfg.name.lower().strip()
    if name not in SEMANTIC_REGISTRY:
        raise ValueError(
            f"Unknown semantic model {cfg.name!r}. "
            f"Available: {sorted(SEMANTIC_REGISTRY)}"
        )
    cls = SEMANTIC_REGISTRY[name]
    # Resolve weights: explicit YAML override wins; otherwise fall back to
    # the per-key default (different checkpoint for each B-variant /
    # GOOSE wrapper).
    weights = cfg.weights or SEMANTIC_DEFAULT_WEIGHTS.get(name, "")
    kwargs: dict = dict(
        weights=weights,
        backend=backend,
        device=hw.device,
        fp16=hw.fp16,
    )
    if cfg.num_classes is not None:
        kwargs["num_classes"] = cfg.num_classes
    if "segformer" in name:
        kwargs["name"] = name   # lets the wrapper find the HF base for local .pth files
    kwargs["tta"] = cfg.tta
    return cls(**kwargs)

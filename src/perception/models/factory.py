"""Factories that build instance and semantic models from config.

Adding a new model = register it here and ship a wrapper class that
implements the corresponding ABC. No other code needs to change.
"""
from __future__ import annotations

from typing import Type

from ..config.schema import HardwareCfg, InstanceModelCfg, SemanticModelCfg
from .backends.base import InferenceBackend
from .instance.base import InstanceModel
from .instance.yoloe import YOLOEInstanceModel
from .semantic.base import SemanticModel
from .semantic.segformer import SegFormerSemanticModel


INSTANCE_REGISTRY: dict[str, Type[InstanceModel]] = {
    "yoloe26l": YOLOEInstanceModel,
    "yoloe-26l": YOLOEInstanceModel,
    "yoloe": YOLOEInstanceModel,
}

SEMANTIC_REGISTRY: dict[str, Type[SemanticModel]] = {
    "segformer-b2": SegFormerSemanticModel,
    "segformer_b2": SegFormerSemanticModel,
    "segformer": SegFormerSemanticModel,
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
    return cls(
        weights=cfg.weights,
        backend=backend,
        device=hw.device,
        fp16=hw.fp16,
    )

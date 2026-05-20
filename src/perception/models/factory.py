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
from .semantic.ddrnet import DDRNetSemanticModel
from .semantic.ppliteseg import PPLiteSegSemanticModel
from .semantic.segformer import SegFormerSemanticModel


INSTANCE_REGISTRY: dict[str, Type[InstanceModel]] = {
    "yoloe26l": YOLOEInstanceModel,
    "yoloe-26l": YOLOEInstanceModel,
    "yoloe": YOLOEInstanceModel,
}

SEMANTIC_REGISTRY: dict[str, Type[SemanticModel]] = {
    # SegFormer-B2 (ADE20K, 150 ch).
    "segformer-b2": SegFormerSemanticModel,
    "segformer_b2": SegFormerSemanticModel,
    "segformer": SegFormerSemanticModel,
    # SegFormer-B4 (ADE20K, 150 ch). Same wrapper, different weights.
    "segformer-b4": SegFormerSemanticModel,
    "segformer_b4": SegFormerSemanticModel,
    # DDRNet-39 with GOOSE-12 head, super_gradients-flavour layer naming.
    # See ``perception.models.semantic._vendored.ddrnet39_goose`` for why
    # we vendor super_gradients' DDRNet variant rather than the more
    # commonly-published chenjun2hao DDRNet-23-slim port.
    "ddrnet": DDRNetSemanticModel,
    "ddrnet-39": DDRNetSemanticModel,
    "ddrnet39": DDRNetSemanticModel,
    # PP-LiteSeg-B2 with GOOSE-12 head -- skeleton kept on disk for a
    # future round; predict_logits() raises NotImplementedError. See
    # ``perception.models.semantic.ppliteseg`` for status.
    "ppliteseg": PPLiteSegSemanticModel,
    "pp-liteseg": PPLiteSegSemanticModel,
    "ppliteseg-b2": PPLiteSegSemanticModel,
    "pp-liteseg-b2": PPLiteSegSemanticModel,
}

#: Per-key default weights resolution. Falls back to ``cfg.weights`` when
#: the user explicitly sets it in YAML. Centralised here so the comparison
#: harness can ask "what would build_semantic_model use for this key?"
#: without instantiating the model.
#:
#: SegFormer values are HuggingFace Hub repo IDs (the wrapper hands them
#: to ``transformers.SegformerForSemanticSegmentation.from_pretrained``).
#: DDRNet / PP-LiteSeg values are **on-disk paths** under ``weights/``;
#: the wrappers ``torch.load`` them directly. The download script
#: ``scripts/download_datasets.py`` populates these paths from the
#: ``goose-dataset.de`` upstream URL.
SEMANTIC_DEFAULT_WEIGHTS: dict[str, str] = {
    "segformer-b2":     "nvidia/segformer-b2-finetuned-ade-512-512",
    "segformer_b2":     "nvidia/segformer-b2-finetuned-ade-512-512",
    "segformer":        "nvidia/segformer-b2-finetuned-ade-512-512",
    "segformer-b4":     "nvidia/segformer-b4-finetuned-ade-512-512",
    "segformer_b4":     "nvidia/segformer-b4-finetuned-ade-512-512",
    "ddrnet":           "weights/ddrnet_category_512.pth",
    "ddrnet-39":        "weights/ddrnet_category_512.pth",
    "ddrnet39":         "weights/ddrnet_category_512.pth",
    # PP-LiteSeg checkpoint intentionally absent from the disk in this
    # round (parent agent dropped it from comparison). The skeleton
    # wrapper still raises NotImplementedError if anyone tries to use it.
    "ppliteseg":        "weights/ppliteseg_category_512.pth",
    "pp-liteseg":       "weights/ppliteseg_category_512.pth",
    "ppliteseg-b2":     "weights/ppliteseg_category_512.pth",
    "pp-liteseg-b2":    "weights/ppliteseg_category_512.pth",
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
        imgsz=cfg.imgsz,
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
        if cfg.processor_size is not None:
            kwargs["processor_size"] = cfg.processor_size
    return cls(**kwargs)

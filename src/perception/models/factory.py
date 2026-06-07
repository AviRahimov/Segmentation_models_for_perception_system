"""Factories that build instance and semantic models from config.

Adding a new model = register it here and ship a wrapper class that
implements the corresponding ABC. No other code needs to change.
"""
from __future__ import annotations

from typing import Type

from ..config.schema import HardwareCfg, InstanceModelCfg, InstancePromptMode, SemanticModelCfg
from .backends.base import InferenceBackend
from .instance.base import InstanceModel
from .instance.rfdetr.model import RFDETRInstanceModel
from .instance.yolo.closed import YOLOClosedInstanceModel
from .instance.yolo.open import YOLOEInstanceModel
from .semantic.auriganet import AurigaNetSemanticModel
from .semantic.base import SemanticModel
from .semantic.segformer import SegFormerSemanticModel


# ---------------------------------------------------------------------------
# Instance model registry
# ---------------------------------------------------------------------------

INSTANCE_REGISTRY: dict[str, Type[InstanceModel]] = {
    # YOLOE — open-vocabulary (existing aliases preserved for backward compat)
    "yoloe26l":  YOLOEInstanceModel,
    "yoloe-26l": YOLOEInstanceModel,
    "yoloe":     YOLOEInstanceModel,
    # YOLO11 — closed-vocabulary (COCO-80, 0-indexed internally)
    "yolo11n": YOLOClosedInstanceModel, "yolo11s": YOLOClosedInstanceModel,
    "yolo11m": YOLOClosedInstanceModel, "yolo11l": YOLOClosedInstanceModel,
    "yolo11x": YOLOClosedInstanceModel,
    # YOLO12 — closed-vocabulary
    "yolo12n": YOLOClosedInstanceModel, "yolo12s": YOLOClosedInstanceModel,
    "yolo12m": YOLOClosedInstanceModel, "yolo12l": YOLOClosedInstanceModel,
    "yolo12x": YOLOClosedInstanceModel,
    # YOLO26 — closed-vocabulary, NMS-free end-to-end head
    "yolo26n": YOLOClosedInstanceModel, "yolo26s": YOLOClosedInstanceModel,
    "yolo26m": YOLOClosedInstanceModel, "yolo26l": YOLOClosedInstanceModel,
    "yolo26x": YOLOClosedInstanceModel,
    # RF-DETR — closed-vocabulary, transformer-based (1-indexed COCO IDs)
    "rfdetr-n":   RFDETRInstanceModel, "rfdetr-s":   RFDETRInstanceModel,
    "rfdetr-m":   RFDETRInstanceModel, "rfdetr-l":   RFDETRInstanceModel,
    "rfdetr-xl":  RFDETRInstanceModel, "rfdetr-2xl": RFDETRInstanceModel,
}

#: Default weight filename per model name.  Override via models.instance.weights in config.yaml.
INSTANCE_DEFAULT_WEIGHTS: dict[str, str] = {
    "yoloe26l":  "yoloe-26l-seg.pt",
    "yoloe-26l": "yoloe-26l-seg.pt",
    "yoloe":     "yoloe-26l-seg.pt",
    "yolo11n": "yolo11n.pt", "yolo11s": "yolo11s.pt", "yolo11m": "yolo11m.pt",
    "yolo11l": "yolo11l.pt", "yolo11x": "yolo11x.pt",
    "yolo12n": "yolo12n.pt", "yolo12s": "yolo12s.pt", "yolo12m": "yolo12m.pt",
    "yolo12l": "yolo12l.pt", "yolo12x": "yolo12x.pt",
    "yolo26n": "yolo26n.pt", "yolo26s": "yolo26s.pt", "yolo26m": "yolo26m.pt",
    "yolo26l": "yolo26l.pt", "yolo26x": "yolo26x.pt",
    "rfdetr-n":   "rf-detr-nano.pth",    "rfdetr-s":   "rf-detr-small.pth",
    "rfdetr-m":   "rf-detr-medium.pth",  "rfdetr-l":   "rf-detr-large.pth",
    "rfdetr-xl":  "rf-detr-xlarge.pth",  "rfdetr-2xl": "rf-detr-2xlarge.pth",
}


# ---------------------------------------------------------------------------
# Semantic model registry
# ---------------------------------------------------------------------------

SEMANTIC_REGISTRY: dict[str, Type[SemanticModel]] = {
    "segformer-b0": SegFormerSemanticModel,
    "segformer_b0": SegFormerSemanticModel,
    "segformer-b1": SegFormerSemanticModel,
    "segformer_b1": SegFormerSemanticModel,
    "segformer-b2": SegFormerSemanticModel,
    "segformer_b2": SegFormerSemanticModel,
    "segformer":    SegFormerSemanticModel,
    "segformer-b4": SegFormerSemanticModel,
    "segformer_b4": SegFormerSemanticModel,
    "auriganet":    AurigaNetSemanticModel,
}

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
    "auriganet":    "",
}


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------

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
    weights = cfg.weights or INSTANCE_DEFAULT_WEIGHTS.get(name, "")
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
        model_name=name,
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
        kwargs["name"] = name
        if cfg.processor_size is not None:
            kwargs["processor_size"] = cfg.processor_size
        if cfg.trt_engine_path:
            kwargs["trt_engine_path"] = cfg.trt_engine_path
    return cls(**kwargs)

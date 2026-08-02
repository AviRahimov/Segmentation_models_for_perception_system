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
from .semantic.dinov2 import DINOv2SemanticModel
from .semantic.mask2former import Mask2FormerSemanticModel
from .semantic.segformer import SegFormerSemanticModel
from .semantic.upernet import UPerNetSemanticModel


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
#: All base/pretrained checkpoints live in weights/base_checkpoints/ (auto-downloaded
#: there by Ultralytics/rfdetr if not already present).
_BASE_CKPT_DIR = "weights/base_checkpoints/"
INSTANCE_DEFAULT_WEIGHTS: dict[str, str] = {
    "yoloe26l":  _BASE_CKPT_DIR + "yoloe-26l-seg.pt",
    "yoloe-26l": _BASE_CKPT_DIR + "yoloe-26l-seg.pt",
    "yoloe":     _BASE_CKPT_DIR + "yoloe-26l-seg.pt",
    "yolo11n": _BASE_CKPT_DIR + "yolo11n.pt", "yolo11s": _BASE_CKPT_DIR + "yolo11s.pt",
    "yolo11m": _BASE_CKPT_DIR + "yolo11m.pt",
    "yolo11l": _BASE_CKPT_DIR + "yolo11l.pt", "yolo11x": _BASE_CKPT_DIR + "yolo11x.pt",
    "yolo12n": _BASE_CKPT_DIR + "yolo12n.pt", "yolo12s": _BASE_CKPT_DIR + "yolo12s.pt",
    "yolo12m": _BASE_CKPT_DIR + "yolo12m.pt",
    "yolo12l": _BASE_CKPT_DIR + "yolo12l.pt", "yolo12x": _BASE_CKPT_DIR + "yolo12x.pt",
    "yolo26n": _BASE_CKPT_DIR + "yolo26n.pt", "yolo26s": _BASE_CKPT_DIR + "yolo26s.pt",
    "yolo26m": _BASE_CKPT_DIR + "yolo26m.pt",
    "yolo26l": _BASE_CKPT_DIR + "yolo26l.pt", "yolo26x": _BASE_CKPT_DIR + "yolo26x.pt",
    "rfdetr-n":   _BASE_CKPT_DIR + "rf-detr-nano.pth",    "rfdetr-s":   _BASE_CKPT_DIR + "rf-detr-small.pth",
    "rfdetr-m":   _BASE_CKPT_DIR + "rf-detr-medium.pth",  "rfdetr-l":   _BASE_CKPT_DIR + "rf-detr-large.pth",
    "rfdetr-xl":  _BASE_CKPT_DIR + "rf-detr-xlarge.pth",  "rfdetr-2xl": _BASE_CKPT_DIR + "rf-detr-2xlarge.pth",
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
    "mask2former":       Mask2FormerSemanticModel,
    "mask2former-base":  Mask2FormerSemanticModel,
    "mask2former-large": Mask2FormerSemanticModel,
    "upernet":      UPerNetSemanticModel,
    "dinov2":       DINOv2SemanticModel,
    "dinov2-base":  DINOv2SemanticModel,
    "dinov2-large": DINOv2SemanticModel,
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
    "auriganet":    "weights/segmentation/orfd/auriganet/best.pth",
    "mask2former":       "weights/segmentation/orfd/mask2former/best.pth",
    "mask2former-base":  "weights/segmentation/orfd/mask2former/best.pth",
    "mask2former-large": "weights/segmentation/orfd/mask2former-large/best.pth",
    "upernet":      "weights/segmentation/orfd/upernet/best.pth",
    "dinov2":       "weights/segmentation/orfd/dinov2/best.pth",
    "dinov2-base":  "weights/segmentation/orfd/dinov2/best.pth",
    "dinov2-large": "weights/segmentation/orfd/dinov2-large/best.pth",
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
        recovery_conf_floor=(
            cfg.low_conf_recovery.recovery_conf_floor
            if cfg.low_conf_recovery.enabled else None
        ),
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
    if "segformer" in name or "mask2former" in name or "dinov2" in name:
        kwargs["name"] = name
        if cfg.processor_size is not None:
            kwargs["processor_size"] = cfg.processor_size
        if cfg.trt_engine_path:
            kwargs["trt_engine_path"] = cfg.trt_engine_path
    return cls(**kwargs)

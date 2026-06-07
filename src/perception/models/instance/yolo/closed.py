"""Closed-vocabulary YOLO detector wrapper (YOLO11 / YOLO12 / YOLO26).

Maps standard COCO IDs (1-indexed, as stored in config.yaml) to user-defined
class names via ``ClassDef.coco_classes``.  The YOLO runtime uses 0-indexed
class IDs, so each COCO ID is decremented by 1 internally.

Classes with an empty ``coco_classes`` list are silently skipped — they will
not appear in detections from this model (no COCO equivalent, e.g. "obstacle").
Switch to YOLOE or a fine-tuned model to detect those classes.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from ....config.schema import ClassDef
from ....core.types import Detection
from ..._weights import resolve_instance_weights
from ...backends.base import InferenceBackend
from ..base import InstanceModel
from ._base import _load_ultralytics_model

logger = logging.getLogger(__name__)


class YOLOClosedInstanceModel(InstanceModel):
    """Wraps any Ultralytics YOLO11/YOLO12/YOLO26 checkpoint for fixed-class detection.

    Text-prompt and discovery kwargs are accepted but ignored so the factory
    can pass a uniform constructor signature regardless of model type.
    """

    def __init__(
        self,
        weights: str = "yolo26l.pt",
        confidence_threshold: float = 0.35,
        backend: InferenceBackend | None = None,
        device: str = "cuda",
        fp16: bool = True,
        *,
        imgsz: int = 640,
        # ignored kwargs (factory passes these universally across all model types)
        prompt_mode: Any = None,
        discovery_vocab_path: Any = None,
        discovery_conf_floor: Any = None,
        discovery_max_det: Any = None,
        model_name: Any = None,
    ) -> None:
        # Ultralytics handles auto-download for bare .pt names (e.g. "yolo26l.pt").
        # Only run our resolver for explicit local paths (.engine, .onnx, or existing file).
        from pathlib import Path as _Path
        if weights and (_Path(weights).suffix in (".engine", ".onnx") or _Path(weights).exists()):
            self._weights = str(resolve_instance_weights(weights))
        else:
            self._weights = weights
        self._device = device
        self._fp16 = fp16
        self._confidence_threshold = float(confidence_threshold)
        self._imgsz = int(imgsz)
        self._model = _load_ultralytics_model(self._weights)
        # Filled at warmup(): {yolo_0indexed_id: (class_name, threshold)}
        self._coco_to_user: dict[int, tuple[str, float]] = {}
        self._predict_conf_floor: float = self._confidence_threshold
        self._ready: bool = False

    # ------------------------------------------------------------------ #

    def warmup(self, classes: Sequence[ClassDef]) -> None:
        self._coco_to_user = {}
        thresholds: list[float] = []

        for cls in classes:
            if cls.is_semantic or not cls.coco_classes:
                continue
            threshold = (
                float(cls.confidence_threshold)
                if cls.confidence_threshold is not None
                else self._confidence_threshold
            )
            thresholds.append(threshold)
            for coco_id in cls.coco_classes:
                # config uses 1-indexed COCO standard IDs; YOLO uses 0-indexed
                self._coco_to_user[coco_id - 1] = (cls.name, threshold)

        if not self._coco_to_user:
            logger.warning(
                "YOLOClosed (%s): no coco_classes defined for any instance class — "
                "predict() will return no detections. Add coco_classes to instance "
                "classes in config.yaml or switch to yoloe for open-vocab detection.",
                self._weights,
            )
            self._ready = True
            return

        self._predict_conf_floor = min(thresholds) if thresholds else self._confidence_threshold
        active_classes = sorted({name for name, _ in self._coco_to_user.values()})
        logger.info(
            "YOLOClosed warmed up: %d COCO→user mappings for classes %s; "
            "predict conf floor=%.2f (device=%s, fp16=%s)",
            len(self._coco_to_user),
            active_classes,
            self._predict_conf_floor,
            self._device,
            self._fp16,
        )
        self._ready = True

    # ------------------------------------------------------------------ #

    def predict(self, frame_bgr: np.ndarray) -> list[Detection]:
        if not self._ready or not self._coco_to_user:
            return []

        results = self._model.predict(
            frame_bgr,
            imgsz=self._imgsz,
            conf=self._predict_conf_floor,
            verbose=False,
            device=self._device,
            half=self._fp16,
        )
        if not results:
            return []

        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        out: list[Detection] = []
        for box in boxes:
            yolo_cls = int(box.cls.item())
            if yolo_cls not in self._coco_to_user:
                continue
            class_name, threshold = self._coco_to_user[yolo_cls]
            score = float(box.conf.item())
            if score < threshold:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            out.append(
                Detection(
                    class_name=class_name,
                    score=score,
                    bbox_xyxy=(x1, y1, x2, y2),
                    mask=None,
                )
            )
        return out

"""RF-DETR detector wrapper — N / S / M / L / XL / 2XL variants.

Uses the ``rfdetr`` Python package (Roboflow).  Maps standard COCO IDs
(1-indexed, as in config.yaml ``coco_classes``) to user class names.
RF-DETR preserves official COCO category IDs (1-indexed) so no offset
adjustment is needed — unlike the YOLO wrappers which subtract 1.

Input size is FIXED per variant; the ``imgsz`` config field is ignored
(a warning is logged).  Supported fixed sizes:
  N=384  S=512  M=576  L=704  XL=700  2XL=880  (square pixels)

FP16 is supported via ``model.optimize_for_inference(dtype=torch.float16)``.
Be aware that FP16 may degrade detection quality on Jetson (log warning).

XL and 2XL require the ``rfdetr[plus]`` extra (PML 1.0 license):
  pip install "rfdetr[plus]"
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import cv2
import numpy as np

from ....config.schema import ClassDef
from ....core.types import Detection
from ...backends.base import InferenceBackend
from ..base import InstanceModel

logger = logging.getLogger(__name__)

# Maps config name → (rfdetr class name, fixed input size in pixels)
_RFDETR_VARIANTS: dict[str, tuple[str, int]] = {
    "rfdetr-n":   ("RFDETRNano",    384),
    "rfdetr-s":   ("RFDETRSmall",   512),
    "rfdetr-m":   ("RFDETRMedium",  576),
    "rfdetr-l":   ("RFDETRLarge",   704),
    "rfdetr-xl":  ("RFDETRXLarge",  700),   # requires rfdetr[plus]
    "rfdetr-2xl": ("RFDETR2XLarge", 880),   # requires rfdetr[plus]
}


def _load_rfdetr(model_name: str, weights: str | None) -> Any:
    try:
        import rfdetr as _rfdetr_pkg  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "The 'rfdetr' package is not installed. "
            "Install it with: pip install rfdetr  "
            "(XL/2XL variants also need: pip install 'rfdetr[plus]')"
        ) from e

    cls_name, _ = _RFDETR_VARIANTS[model_name]
    cls = getattr(_rfdetr_pkg, cls_name, None)
    if cls is None:
        raise RuntimeError(
            f"rfdetr.{cls_name} not found in the installed version. "
            "For XL/2XL variants install: pip install 'rfdetr[plus]'"
        )
    if weights:
        return cls(pretrained=False, checkpoint=weights)
    return cls(pretrained=True)


class RFDETRInstanceModel(InstanceModel):
    """Wraps an RF-DETR variant for closed-vocabulary COCO detection.

    Text-prompt and discovery kwargs are accepted but ignored so the factory
    can pass a uniform constructor signature regardless of model type.
    """

    def __init__(
        self,
        weights: str | None = None,
        confidence_threshold: float = 0.35,
        backend: InferenceBackend | None = None,
        device: str = "cuda",
        fp16: bool = True,
        *,
        model_name: str = "rfdetr-m",
        imgsz: int = 640,
        # ignored open-vocab / discovery kwargs
        prompt_mode: Any = None,
        discovery_vocab_path: Any = None,
        discovery_conf_floor: Any = None,
        discovery_max_det: Any = None,
    ) -> None:
        self._model_name = model_name
        self._weights = weights
        self._device = device
        self._fp16 = fp16
        self._confidence_threshold = float(confidence_threshold)
        _, self._fixed_imgsz = _RFDETR_VARIANTS.get(model_name, ("", imgsz))
        if imgsz != 640:  # non-default was explicitly set
            logger.warning(
                "RFDETRInstanceModel: imgsz=%d is ignored — RF-DETR uses a fixed "
                "input size of %d for the '%s' variant.",
                imgsz, self._fixed_imgsz, model_name,
            )
        self._model = _load_rfdetr(model_name, weights)
        self._coco_to_user: dict[int, tuple[str, float]] = {}
        self._predict_conf_floor: float = self._confidence_threshold
        self._ready: bool = False

    # ------------------------------------------------------------------ #

    def warmup(self, classes: Sequence[ClassDef]) -> None:
        import torch

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
                # RF-DETR uses 1-indexed COCO IDs — no offset needed
                self._coco_to_user[coco_id] = (cls.name, threshold)

        if not self._coco_to_user:
            logger.warning(
                "RFDETRInstanceModel: no coco_classes defined — predict() will return "
                "no detections."
            )

        self._predict_conf_floor = min(thresholds) if thresholds else self._confidence_threshold

        if self._fp16:
            logger.warning(
                "RFDETRInstanceModel: FP16 requested. RF-DETR FP16 may degrade detection "
                "quality on Jetson — validate results before deploying."
            )
            self._model.optimize_for_inference(compile=False, dtype=torch.float16)
        else:
            self._model.optimize_for_inference(compile=False, dtype=torch.float32)

        active_classes = sorted({name for name, _ in self._coco_to_user.values()})
        logger.info(
            "RFDETRInstanceModel ('%s') warmed up: %d COCO→user mappings for %s; "
            "conf floor=%.2f (device=%s, fp16=%s, fixed_imgsz=%d)",
            self._model_name,
            len(self._coco_to_user),
            active_classes,
            self._predict_conf_floor,
            self._device,
            self._fp16,
            self._fixed_imgsz,
        )
        self._ready = True

    # ------------------------------------------------------------------ #

    def predict(self, frame_bgr: np.ndarray) -> list[Detection]:
        if not self._ready or not self._coco_to_user:
            return []

        import PIL.Image  # lazy: only when rfdetr is active

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PIL.Image.fromarray(rgb)
        sv_det = self._model.predict(pil_img, threshold=self._predict_conf_floor)

        if sv_det is None or len(sv_det) == 0:
            return []

        out: list[Detection] = []
        for i in range(len(sv_det)):
            coco_id = int(sv_det.class_id[i])
            if coco_id not in self._coco_to_user:
                continue
            class_name, threshold = self._coco_to_user[coco_id]
            score = float(sv_det.confidence[i])
            if score < threshold:
                continue
            x1, y1, x2, y2 = (int(v) for v in sv_det.xyxy[i])
            out.append(
                Detection(
                    class_name=class_name,
                    score=score,
                    bbox_xyxy=(x1, y1, x2, y2),
                    mask=None,
                )
            )
        return out

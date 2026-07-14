"""YOLOE wrapper — open-vocabulary instance segmentation (Ultralytics).

Text embeddings for the configured prompts are computed exactly once during
:meth:`warmup` (via ``get_text_pe`` -> ``set_classes``) and cached on the
underlying model.  Subsequent :meth:`predict` calls run pure visual forward
only — the text encoder is never re-invoked per frame.

Discovery mode swaps in a newline-delimited vocabulary file (still via
``set_classes``); per-class thresholds are bypassed in favour of
``discovery_conf_floor``.

This file is the canonical home of ``YOLOEInstanceModel``.
``perception.models.instance.yoloe`` re-exports it for backward compatibility.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from ....config.schema import ClassDef, InstancePromptMode
from ....core.types import Detection
from .._threshold_gate import gate_confidence
from ..._weights import resolve_instance_weights
from ...backends.base import InferenceBackend
from ..base import InstanceModel
from ..discovery_vocab import load_discovery_prompts
from ._base import _load_ultralytics_model

logger = logging.getLogger(__name__)


def _resolve_method(model: Any, name: str) -> Callable[..., Any] | None:
    """Find *name* on model, model.model, or model.predictor (YOLOE API variation)."""
    seen: set[int] = set()
    for cand in (model, getattr(model, "model", None), getattr(model, "predictor", None)):
        if cand is None or id(cand) in seen:
            continue
        seen.add(id(cand))
        m = getattr(cand, name, None)
        if callable(m):
            return m
    return None


class YOLOEInstanceModel(InstanceModel):
    """Wraps Ultralytics' YOLOE for open-vocabulary instance segmentation."""

    def __init__(
        self,
        weights: str = "yoloe-26l-seg.pt",
        confidence_threshold: float = 0.35,
        backend: InferenceBackend | None = None,
        device: str = "cuda",
        fp16: bool = True,
        *,
        prompt_mode: InstancePromptMode = "production",
        discovery_vocab_path: str = "",
        discovery_conf_floor: float = 0.05,
        discovery_max_det: int | None = None,
        imgsz: int = 640,
        recovery_conf_floor: float | None = None,
        model_name: Any = None,  # ignored; factory passes it universally
    ) -> None:
        local_weights = resolve_instance_weights(weights)
        self._weights = str(local_weights)
        self._device = device
        self._fp16 = fp16
        self._conf = float(confidence_threshold)
        self._recovery_floor = (
            float(recovery_conf_floor) if recovery_conf_floor is not None else None
        )
        self._backend = backend
        self._model = _load_ultralytics_model(self._weights)
        self._instance_classes: list[ClassDef] = []
        self._cls_idx_to_name: dict[int, str] = {}
        self._cls_idx_to_threshold: dict[int, float] = {}
        self._predict_conf_floor: float = self._conf
        self._discovery_mode = prompt_mode == "discovery"
        self._discovery_vocab_path = discovery_vocab_path
        self._discovery_conf_floor = float(discovery_conf_floor)
        self._discovery_max_det = discovery_max_det
        self._imgsz = int(imgsz)
        self._yoloe_ready: bool = False

    # ------------------------------------------------------------------ #

    def warmup(self, classes: Sequence[ClassDef]) -> None:
        self._yoloe_ready = False
        if self._discovery_mode:
            self._instance_classes = []
            if not self._discovery_vocab_path:
                raise RuntimeError(
                    "YOLOE discovery mode requires discovery_vocabulary_path (set via config).",
                )
            try:
                prompts = load_discovery_prompts(self._discovery_vocab_path)
            except (OSError, ValueError) as e:
                raise RuntimeError(f"Failed to load discovery vocabulary: {e}") from e
            self._cls_idx_to_name = {i: p for i, p in enumerate(prompts)}
            self._cls_idx_to_threshold = {i: 0.0 for i in range(len(prompts))}
            self._predict_conf_floor = self._discovery_conf_floor
            self._apply_set_classes(prompts)
            self._yoloe_ready = True
            logger.info(
                "YOLOE discovery mode: %d vocabulary prompts; predict conf floor=%.4f "
                "(device=%s, fp16=%s)",
                len(prompts),
                self._predict_conf_floor,
                self._device,
                self._fp16,
            )
            return

        self._instance_classes = [c for c in classes if not c.is_semantic]
        if not self._instance_classes:
            logger.info("YOLOE: no instance classes configured; predict() will return [].")
            return
        prompts = [c.text_prompt for c in self._instance_classes]
        self._cls_idx_to_name = {i: c.name for i, c in enumerate(self._instance_classes)}
        self._cls_idx_to_threshold = {
            i: float(c.confidence_threshold)
            if c.confidence_threshold is not None
            else self._conf
            for i, c in enumerate(self._instance_classes)
        }
        base_floor = min([self._conf, *self._cls_idx_to_threshold.values()])
        self._predict_conf_floor = (
            base_floor if self._recovery_floor is None
            else min(base_floor, self._recovery_floor)
        )
        per_class_msgs = [
            f"{c.name}={self._cls_idx_to_threshold[i]:.2f}"
            + ("" if c.confidence_threshold is not None else " (default)")
            for i, c in enumerate(self._instance_classes)
        ]
        logger.info(
            "YOLOE per-class confidence thresholds: %s; predict floor=%.2f%s",
            ", ".join(per_class_msgs), self._predict_conf_floor,
            f" (recovery floor={self._recovery_floor:.2f})" if self._recovery_floor is not None else "",
        )
        self._apply_set_classes(prompts)
        self._yoloe_ready = True
        logger.info(
            "YOLOE warmed up with %d open-vocab prompts (device=%s, fp16=%s)",
            len(prompts), self._device, self._fp16,
        )

    def _apply_set_classes(self, prompts: list[str]) -> None:
        set_classes = _resolve_method(self._model, "set_classes")
        if set_classes is None:
            raise RuntimeError(
                f"The loaded checkpoint {self._weights!r} does not expose a "
                "'set_classes' method. Use a YOLOE-seg checkpoint "
                "(e.g. 'yoloe-26l-seg.pt') with a recent ultralytics version."
            )
        # Head nc BEFORE set_classes: PE-fine-tuned checkpoints (YOLOEPETrainer,
        # e.g. round1) carry their training classes FUSED into the head.
        # set_classes() optimistically rewrites head.nc afterwards, so the
        # pre-call value is the only honest one.
        try:
            nc_before = int(self._model.model.model[-1].nc)
        except (AttributeError, IndexError, TypeError):
            nc_before = -1

        get_text_pe = _resolve_method(self._model, "get_text_pe")
        if get_text_pe is not None:
            text_pe = get_text_pe(prompts)
            set_classes(prompts, text_pe)
        else:
            logger.info(
                "Loaded model has no get_text_pe(); calling set_classes(prompts) only "
                "(YOLOWorld-style API)."
            )
            set_classes(prompts)

        # Smoke forward: a PE-fused head silently accepts a wrong prompt count
        # and then crashes mid-video with a cryptic tensor-size error. Fail at
        # warmup with an actionable message instead.
        try:
            import numpy as _np
            self._model.predict(
                _np.zeros((160, 160, 3), dtype=_np.uint8),
                imgsz=160, conf=0.99, verbose=False,
                device=self._device, half=self._fp16,
            )
        except RuntimeError as exc:
            if nc_before <= 0 or nc_before == len(prompts):
                raise  # not the fused-head mismatch — surface the real error
            raise RuntimeError(
                f"Checkpoint {self._weights!r} appears PE fine-tuned with a fixed "
                f"{nc_before}-class head, but the active profile defines "
                f"{len(prompts)} classes — this combination cannot run. Use a "
                f"profile with exactly {nc_before} classes for this checkpoint, "
                "or the pretrained open-vocabulary checkpoint "
                "(weights: \"\" → yoloe-26l-seg.pt) for this profile."
            ) from exc

    # ------------------------------------------------------------------ #

    def predict(self, frame_bgr: np.ndarray) -> list[Detection]:
        if not self._yoloe_ready:
            return []

        pred_kw: dict[str, Any] = {
            "conf": self._predict_conf_floor,
            "verbose": False,
            "device": self._device,
            "half": self._fp16,
            "imgsz": self._imgsz,
        }
        if self._discovery_max_det is not None:
            pred_kw["max_det"] = int(self._discovery_max_det)

        results = self._model.predict(frame_bgr, **pred_kw)
        out: list[Detection] = []
        if not results:
            return out

        r = results[0]
        boxes = getattr(r, "boxes", None)
        masks = getattr(r, "masks", None)
        if boxes is None or len(boxes) == 0:
            return out

        xyxy = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()
        cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
        h, w = frame_bgr.shape[:2]

        mask_arrs: np.ndarray | None = None
        if masks is not None and getattr(masks, "data", None) is not None:
            mask_arrs = masks.data.detach().cpu().numpy()

        for i, (box, sc, ci) in enumerate(zip(xyxy, scores, cls_ids, strict=True)):
            ci_int = int(ci)
            cls_thr: float | None = None
            if not self._discovery_mode:
                cls_thr = self._cls_idx_to_threshold.get(ci_int, self._conf)
                if not gate_confidence(float(sc), cls_thr, self._recovery_floor):
                    continue
            cname = self._cls_idx_to_name.get(ci_int, "unknown")
            mask: np.ndarray | None = None
            if mask_arrs is not None and i < mask_arrs.shape[0]:
                m = mask_arrs[i]
                if m.shape != (h, w):
                    m = cv2.resize(
                        m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                    )
                else:
                    m = m.astype(np.uint8)
                mask = (m > 0).astype(np.uint8)
            x1, y1, x2, y2 = (int(v) for v in box)
            out.append(
                Detection(
                    class_name=cname,
                    score=float(sc),
                    bbox_xyxy=(x1, y1, x2, y2),
                    mask=mask,
                    display_threshold=cls_thr,
                )
            )
        return out

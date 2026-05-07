"""SAM 2.1 streaming tracker with graceful fallback.

If ``sam2`` is not installed, :func:`is_sam2_available` returns ``False``
and the perception pipeline transparently uses :class:`IoUInstanceTracker`
instead. When SAM2 is present, this class:

* primes the streaming predictor with detector bboxes on the first frame,
* propagates SAM2 memory on every subsequent frame,
* re-prompts whenever ``frame_count`` % ``reprompt_every_n_frames == 0`` or
  when the detector score drops below ``min_track_score``,
* matches incoming detections to existing tracks by class-conditioned IoU.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

from ..core.geometry import iou_xyxy
from ..core.types import Detection
from .base import InstanceTracker

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Best-effort import - many environments will not have sam2 installed.
# --------------------------------------------------------------------------- #
_SAM2_AVAILABLE: bool = False
_SAM2_IMPORT_ERROR: str = ""
_build_sam2_camera_predictor: Any = None
try:
    from sam2.build_sam import build_sam2_camera_predictor as _build  # type: ignore
    _build_sam2_camera_predictor = _build
    _SAM2_AVAILABLE = True
except Exception as e:  # noqa: BLE001
    _SAM2_IMPORT_ERROR = repr(e)


def is_sam2_available() -> bool:
    return _SAM2_AVAILABLE


@dataclass
class _Track:
    obj_id: int
    class_name: str
    last_score: float
    last_bbox: tuple[int, int, int, int]


class SAM2InstanceTracker(InstanceTracker):
    def __init__(
        self,
        ckpt: str,
        model_cfg: str,
        reprompt_every_n_frames: int = 30,
        min_track_score: float = 0.4,
        iou_match_threshold: float = 0.3,
        device: str = "cuda",
    ) -> None:
        if not _SAM2_AVAILABLE:
            raise RuntimeError(
                "SAM2 is not installed. Use IoUInstanceTracker instead. "
                f"Underlying ImportError: {_SAM2_IMPORT_ERROR}"
            )
        self._predictor = _build_sam2_camera_predictor(model_cfg, ckpt, device=device)
        self._reprompt_n = max(1, int(reprompt_every_n_frames))
        self._min_track_score = float(min_track_score)
        self._iou_match = float(iou_match_threshold)
        self._next_obj_id = 1
        self._frame_count = 0
        self._tracks: dict[int, _Track] = {}
        self._initialised = False

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        try:
            if hasattr(self._predictor, "reset_state"):
                self._predictor.reset_state()
        except Exception as e:  # noqa: BLE001
            logger.debug("SAM2 reset_state failed (continuing): %s", e)
        self._tracks.clear()
        self._frame_count = 0
        self._initialised = False
        # _next_obj_id stays monotonic so logs are unambiguous across resets.

    # ------------------------------------------------------------------ #
    def update(
        self,
        frame_bgr: np.ndarray,
        detections: Sequence[Detection],
    ) -> list[Detection]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            if not self._initialised:
                return self._initial_frame(rgb, detections)
            return self._track_frame(rgb, detections)
        except Exception as e:  # noqa: BLE001
            logger.warning("SAM2 update failed (%s); passing detections through", e)
            self._frame_count += 1
            return [
                Detection(
                    class_name=d.class_name,
                    score=d.score,
                    bbox_xyxy=d.bbox_xyxy,
                    mask=d.mask,
                    track_id=None,
                )
                for d in detections
            ]

    # ------------------------------------------------------------------ #
    def _initial_frame(
        self, rgb: np.ndarray, detections: Sequence[Detection]
    ) -> list[Detection]:
        self._predictor.load_first_frame(rgb)
        self._initialised = True
        out: list[Detection] = []
        for det in detections:
            obj_id = self._next_obj_id
            self._next_obj_id += 1
            self._predictor.add_new_prompt(
                frame_idx=0,
                obj_id=obj_id,
                bbox=np.asarray(det.bbox_xyxy, dtype=np.float32),
            )
            self._tracks[obj_id] = _Track(
                obj_id=obj_id,
                class_name=det.class_name,
                last_score=det.score,
                last_bbox=det.bbox_xyxy,
            )
            out.append(
                Detection(
                    class_name=det.class_name,
                    score=det.score,
                    bbox_xyxy=det.bbox_xyxy,
                    mask=det.mask,
                    track_id=obj_id,
                )
            )
        self._frame_count += 1
        return out

    # ------------------------------------------------------------------ #
    def _track_frame(
        self, rgb: np.ndarray, detections: Sequence[Detection]
    ) -> list[Detection]:
        # 1. propagate memory and collect masks per existing object id
        propagated: dict[int, np.ndarray] = self._propagate(rgb)

        # 2. greedy IoU match new detections to existing tracks
        matched: set[int] = set()
        out: list[Detection] = []
        for det in detections:
            best_oid, best_iou = -1, 0.0
            for oid, track in self._tracks.items():
                if oid in matched or track.class_name != det.class_name:
                    continue
                iou = iou_xyxy(det.bbox_xyxy, track.last_bbox)
                if iou > best_iou:
                    best_iou, best_oid = iou, oid

            if best_oid >= 0 and best_iou >= self._iou_match:
                matched.add(best_oid)
                track = self._tracks[best_oid]
                track.last_score = det.score
                track.last_bbox = det.bbox_xyxy
                # Re-prompt periodically or on confidence drop.
                if (
                    self._frame_count % self._reprompt_n == 0
                    or det.score < self._min_track_score
                ):
                    self._safe_reprompt(best_oid, det.bbox_xyxy)
                mask = propagated.get(best_oid, det.mask)
                out.append(
                    Detection(
                        class_name=det.class_name,
                        score=det.score,
                        bbox_xyxy=det.bbox_xyxy,
                        mask=mask,
                        track_id=best_oid,
                    )
                )
            else:
                # New track: assign fresh id and register with SAM2.
                new_id = self._next_obj_id
                self._next_obj_id += 1
                self._safe_reprompt(new_id, det.bbox_xyxy)
                self._tracks[new_id] = _Track(
                    obj_id=new_id,
                    class_name=det.class_name,
                    last_score=det.score,
                    last_bbox=det.bbox_xyxy,
                )
                out.append(
                    Detection(
                        class_name=det.class_name,
                        score=det.score,
                        bbox_xyxy=det.bbox_xyxy,
                        mask=det.mask,
                        track_id=new_id,
                    )
                )
        self._frame_count += 1
        return out

    # ------------------------------------------------------------------ #
    def _propagate(self, rgb: np.ndarray) -> dict[int, np.ndarray]:
        propagated: dict[int, np.ndarray] = {}
        try:
            obj_ids, mask_logits = self._predictor.track(rgb)
        except Exception as e:  # noqa: BLE001
            logger.debug("SAM2 track() failed (%s); skipping propagation", e)
            return propagated
        if obj_ids is None or mask_logits is None:
            return propagated
        for oid, ml in zip(obj_ids, mask_logits):
            arr = ml.detach().cpu().numpy() if hasattr(ml, "detach") else np.asarray(ml)
            if arr.ndim == 3:
                arr = arr[0]
            propagated[int(oid)] = (arr > 0.0).astype(np.uint8)
        return propagated

    # ------------------------------------------------------------------ #
    def _safe_reprompt(self, obj_id: int, bbox: tuple[int, int, int, int]) -> None:
        try:
            self._predictor.add_new_prompt(
                frame_idx=self._frame_count,
                obj_id=obj_id,
                bbox=np.asarray(bbox, dtype=np.float32),
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("SAM2 add_new_prompt(obj_id=%d) failed: %s", obj_id, e)

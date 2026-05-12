"""Per-class display-mode-aware renderer.

The renderer is a pure function of (FrameResult, ClassConfig, RenderConfig)
to a BGR image. It depends on neither models nor Qt.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from ..config.schema import ClassDef, InstancePromptMode, PlayerCfg
from ..core.color import make_bgr_palette
from ..core.types import FrameResult
from . import overlay

_DISCOVERY_BGR = (255, 200, 0)
_LABEL_TRUNC = 64


class Renderer:
    def __init__(
        self,
        classes: Iterable[ClassDef],
        player_cfg: PlayerCfg,
        *,
        yoloe_prompt_mode: InstancePromptMode = "production",
    ) -> None:
        self._classes: dict[str, ClassDef] = {c.name: c for c in classes}
        self._cfg = player_cfg
        self._palette_bgr = make_bgr_palette(self._classes.values())
        self._discovery = yoloe_prompt_mode == "discovery"

    # ------------------------------------------------------------------ #
    def render(self, result: FrameResult, fps: float = 0.0) -> np.ndarray:
        img = result.frame_bgr.copy()

        # Semantic first so instance overlays draw on top.
        if result.semantic is not None:
            img = self._render_semantic(img, result)

        # Instance detections.
        for det in result.detections:
            if self._discovery:
                label = (
                    det.class_name
                    if len(det.class_name) <= _LABEL_TRUNC
                    else det.class_name[: _LABEL_TRUNC - 3] + "..."
                )
                lab = label if det.track_id is None else f"{label}#{det.track_id}"
                color = _DISCOVERY_BGR
                if det.mask is not None:
                    img = overlay.blend_mask(img, det.mask, color, self._cfg.mask_alpha)
                img = overlay.draw_bbox(img, det.bbox_xyxy, color, lab, det.score)
                continue
            cls = self._classes.get(det.class_name)
            if cls is None or cls.display_mode == "none":
                continue
            color = self._palette_bgr.get(det.class_name, (0, 255, 0))
            if cls.display_mode in ("mask_only", "both") and det.mask is not None:
                img = overlay.blend_mask(img, det.mask, color, self._cfg.mask_alpha)
            if cls.display_mode in ("bbox_only", "both"):
                label = cls.name if det.track_id is None else f"{cls.name}#{det.track_id}"
                img = overlay.draw_bbox(img, det.bbox_xyxy, color, label, det.score)

        # HUD: legend + FPS.
        if self._cfg.show_class_legend:
            note_y = 10
            if self._discovery:
                img = overlay.draw_yoloe_discovery_note(img, y_start=10)
                note_y = 36
                sem_only = [
                    c for c in self._classes.values()
                    if c.is_semantic and c.display_mode != "none"
                ]
                img = overlay.draw_legend(img, sem_only, origin=(10, note_y))
            else:
                visible = [c for c in self._classes.values() if c.display_mode != "none"]
                img = overlay.draw_legend(img, visible)
        if self._cfg.show_fps:
            img = overlay.draw_fps(img, fps)
        return img

    # ------------------------------------------------------------------ #
    def _render_semantic(self, img: np.ndarray, result: FrameResult) -> np.ndarray:
        sem = result.semantic
        assert sem is not None
        if sem.logits.numel() == 0 or len(sem.class_names) == 0:
            return img
        idx = sem.logits.argmax(dim=0).detach().cpu().numpy().astype(np.int32)

        rg = "road_ground"
        road_idx = sem.class_names.index(rg) if rg in sem.class_names else None
        order = list(range(len(sem.class_names)))
        cls_rg = self._classes.get(rg)
        draw_rg_last = (
            road_idx is not None
            and self._cfg.draw_road_ground_semantic_last
            and cls_rg is not None
            and cls_rg.display_mode in ("mask_only", "both")
        )
        if draw_rg_last:
            order = [i for i in order if i != road_idx]
            order.append(int(road_idx))

        for c_idx in order:
            name = sem.class_names[c_idx]
            cls = self._classes.get(name)
            if cls is None or cls.display_mode == "none":
                continue
            if cls.display_mode in ("mask_only", "both"):
                mask = (idx == c_idx).astype(np.uint8)
                if mask.any():
                    color = self._palette_bgr.get(name, (0, 0, 255))
                    img = overlay.blend_mask(img, mask, color, self._cfg.mask_alpha)
        return img

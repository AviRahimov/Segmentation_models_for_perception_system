"""Drawing primitives. All inputs/outputs are BGR images."""
from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_bbox(
    img: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
    label: str = "",
    score: Optional[float] = None,
    thickness: int = 2,
) -> np.ndarray:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        text = label if score is None else f"{label} {score:.2f}"
        scale = 0.5
        (tw, th), baseline = cv2.getTextSize(text, _FONT, scale, 1)
        y_top = max(0, y1 - th - baseline - 4)
        cv2.rectangle(img, (x1, y_top), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            img, text, (x1 + 2, y1 - baseline - 2),
            _FONT, scale, (0, 0, 0), 1, cv2.LINE_AA,
        )
    return img


def blend_mask(
    img: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.5,
) -> np.ndarray:
    if mask is None:
        return img
    h, w = img.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(
            mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST,
        )
    sel = mask.astype(bool)
    if not sel.any():
        return img
    overlay = np.zeros_like(img)
    overlay[sel] = color
    out = img.copy()
    out[sel] = (img[sel].astype(np.float32) * (1.0 - alpha)
                + overlay[sel].astype(np.float32) * alpha).astype(np.uint8)
    return out


def draw_legend(
    img: np.ndarray,
    classes: Iterable[object],
    *,
    origin: tuple[int, int] = (10, 10),
    row_height: int = 22,
) -> np.ndarray:
    x0, y0 = origin
    for c in classes:
        rgb = getattr(c, "color_rgb", (255, 0, 0))
        name = getattr(c, "name", "?")
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        cv2.rectangle(img, (x0, y0), (x0 + 18, y0 + row_height - 6), bgr, -1)
        cv2.rectangle(img, (x0, y0), (x0 + 18, y0 + row_height - 6), (0, 0, 0), 1)
        cv2.putText(
            img, str(name), (x0 + 26, y0 + row_height - 8),
            _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
        y0 += row_height
    return img


def draw_fps(img: np.ndarray, fps: float) -> np.ndarray:
    text = f"{fps:5.1f} FPS"
    h, w = img.shape[:2]
    (tw, _), _ = cv2.getTextSize(text, _FONT, 0.7, 2)
    cv2.putText(
        img, text, (w - tw - 12, 28),
        _FONT, 0.7, (0, 255, 255), 2, cv2.LINE_AA,
    )
    return img


def draw_yoloe_discovery_note(img: np.ndarray, *, y_start: int = 10) -> np.ndarray:
    """Small HUD line for YOLOE discovery overlays (BGR)."""
    line = "YOLOE discovery — prompt text on boxes; semantic legend unchanged"
    x, y = 10, max(26, int(y_start))
    cv2.putText(img, line, (x, y), _FONT, 0.5, (0, 240, 255), 2, cv2.LINE_AA)
    return img

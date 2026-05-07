"""Color helpers (RGB <-> BGR for OpenCV interop)."""
from __future__ import annotations

from typing import Iterable, Mapping


def rgb_to_bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


def bgr_for(class_name: str, palette: Mapping[str, tuple[int, int, int]]) -> tuple[int, int, int]:
    """Return the BGR color for ``class_name`` (palette is RGB-keyed)."""
    rgb = palette.get(class_name, (255, 0, 0))
    return rgb_to_bgr(rgb)


def make_bgr_palette(
    classes: Iterable[object],
) -> dict[str, tuple[int, int, int]]:
    """Build a ``{class_name: BGR}`` palette from any iterable of objects with
    ``name`` and ``color_rgb`` attributes (typically :class:`ClassDef`).
    """
    palette: dict[str, tuple[int, int, int]] = {}
    for c in classes:
        rgb = getattr(c, "color_rgb", (255, 0, 0))
        name = getattr(c, "name", None)
        if name:
            palette[str(name)] = rgb_to_bgr(tuple(rgb))  # type: ignore[arg-type]
    return palette

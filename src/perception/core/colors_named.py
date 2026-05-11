"""Named-color and hex-string color parser used by the YAML config loader.

The on-disk YAML uses ``color: "<name-or-hex>"`` (e.g. ``"green"`` or
``"#00C850"``); :func:`parse_color` converts that into an internal
``tuple[int, int, int]`` stored on :class:`~perception.config.schema.ClassDef`
as ``color_rgb`` so the renderer continues to consume the same RGB form.

The 30 named colors below cover a deliberately broad palette of visually
distinct hues (close enough to common CSS values that a human picking a
color from memory will land on something usable, but not constrained to
exact CSS-level RGBs).
"""
from __future__ import annotations

_NAMED_COLORS: dict[str, tuple[int, int, int]] = {
    "red":        (255,   0,   0),
    "orange":     (255, 140,   0),
    "yellow":     (255, 220,   0),
    "green":      (  0, 160,  60),
    "lime":       (  0, 255,   0),
    "blue":       ( 30,  90, 255),
    "navy":       (  0,   0, 128),
    "teal":       (  0, 128, 128),
    "cyan":       (  0, 220, 220),
    "purple":     (128,   0, 128),
    "magenta":    (255,   0, 255),
    "pink":       (255, 105, 180),
    "brown":      (139,  69,  19),
    "tan":        (210, 180, 140),
    "olive":      (128, 128,   0),
    "beige":      (235, 225, 195),
    "gold":       (255, 200,   0),
    "silver":     (200, 200, 200),
    "gray":       (128, 128, 128),
    "black":      (  0,   0,   0),
    "white":      (255, 255, 255),
    "maroon":     (128,   0,   0),
    "crimson":    (220,  20,  60),
    "indigo":     ( 75,   0, 130),
    "violet":     (148,   0, 211),
    "chartreuse": (127, 255,   0),
    "turquoise":  ( 64, 224, 208),
    "salmon":     (250, 128, 114),
    "coral":      (255, 127,  80),
    "mint":       (152, 255, 152),
}

_HEX_DIGITS = frozenset("0123456789abcdef")


def parse_color(spec: str) -> tuple[int, int, int]:
    """Parse a color spec into an RGB tuple.

    Accepts:
      * any of the 30 named colors in :data:`_NAMED_COLORS` (case-insensitive,
        whitespace-stripped);
      * a 6-digit hex string ``"#RRGGBB"`` or ``"RRGGBB"`` (with or
        without the leading ``#``).

    Raises:
        ValueError: on any other input — unknown name, non-string,
            empty string, or hex of length ``3 / 4 / 8``.
    """
    if not isinstance(spec, str):
        raise ValueError(
            f"color must be a string, got {type(spec).__name__}: {spec!r}"
        )
    s = spec.strip().lower()
    if not s:
        raise ValueError("color string must be non-empty")

    if s.startswith("#"):
        return _parse_hex(s[1:], spec)
    if s in _NAMED_COLORS:
        return _NAMED_COLORS[s]
    # Bare hex (no '#') — only treat as hex if every char is a hex digit AND
    # it is not also a valid named color (already handled above).
    if all(ch in _HEX_DIGITS for ch in s):
        return _parse_hex(s, spec)
    raise ValueError(
        f"unknown color name {spec!r}; valid names: {sorted(_NAMED_COLORS)}"
    )


def _parse_hex(body: str, original: str) -> tuple[int, int, int]:
    if any(ch not in _HEX_DIGITS for ch in body):
        raise ValueError(f"invalid hex color {original!r}")
    if len(body) == 6:
        return (int(body[0:2], 16), int(body[2:4], 16), int(body[4:6], 16))
    if len(body) in (3, 4, 8):
        raise ValueError(
            f"hex color {original!r}: only 6-digit '#RRGGBB' is supported "
            f"(3/4/8-digit hex is rejected to avoid ambiguity)."
        )
    raise ValueError(f"invalid hex color {original!r}")

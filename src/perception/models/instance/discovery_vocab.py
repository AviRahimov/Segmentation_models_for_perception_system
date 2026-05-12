"""Load newline-delimited YOLOE discovery prompt lists."""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DISCOVERY_WARN_LINES = 500
_DISCOVERY_MAX_LINES = 10_000


def load_discovery_prompts(path: str | Path) -> list[str]:
    """Return unique non-empty prompts (first occurrence preserved).

    Lines starting with ``#`` after strip are skipped. Warns above
    ``_DISCOVERY_WARN_LINES``; rejects above ``_DISCOVERY_MAX_LINES``.
    """
    p = Path(path)
    raw = p.read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    out: list[str] = []
    for ln in raw:
        s = str(ln).strip()
        if not s or s.startswith("#"):
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) > _DISCOVERY_MAX_LINES:
            raise ValueError(
                f"Discovery vocabulary exceeds {_DISCOVERY_MAX_LINES} unique lines ({p}). "
                "Split or trim the list."
            )
    if not out:
        raise ValueError(f"Discovery vocabulary is empty after parsing ({p}).")
    if len(out) > _DISCOVERY_WARN_LINES:
        logger.warning(
            "Discovery vocabulary has %d prompts (>%d)—warmup/inference cost may spike.",
            len(out),
            _DISCOVERY_WARN_LINES,
        )
    return out

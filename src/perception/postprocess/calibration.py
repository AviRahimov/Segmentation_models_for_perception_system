"""Post-hoc confidence calibration (temperature scaling).

Detector confidence scores are not calibrated probabilities out of the box —
a box scored 0.8 doesn't mean "80% chance this is real." Temperature scaling
fixes this cheaply: divide the score's logit by a learned scalar ``T`` before
re-applying sigmoid, then re-normalize. It cannot change score *ordering*
(monotonic in T), so it never affects which detections pass a fixed
threshold — only how trustworthy that threshold's neighbourhood actually is,
and how well scores compare *across* classes with different T.

Fitting (``fit_temperature``) happens offline against a held-out labeled
benchmark (see ``scripts/detection/evaluation/fit_calibration.py``), which
already has the IoU-based TP/FP matching machinery (``_ap_utils.py``). This
module stays a pure, dependency-free function of (scores, labels) so it has
no knowledge of ``Pred``/``GT`` or how a "TP" was decided — that decision is
the eval script's job, not this module's.

Applying the fitted temperature at inference time is a single sigmoid per
detection — free relative to the model forward pass, and gated behind
``postprocess.calibration.enabled`` so it's an easy on/off A/B switch.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..core.types import Detection

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def apply_temperature(score: float, temperature: float) -> float:
    """Rescale one score by a fitted temperature. ``temperature == 1.0`` is a no-op."""
    if temperature == 1.0:
        return score
    return float(_sigmoid(_logit(np.array([score]))[0] / temperature))


def fit_temperature(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    bounds: tuple[float, float] = (0.05, 10.0),
) -> float:
    """Fit a single temperature minimizing NLL of ``labels`` given ``scores``.

    ``labels`` must be 1 (true positive) / 0 (false positive) — the caller
    decides that via IoU matching against ground truth; this function only
    sees the resulting binary outcomes. Returns 1.0 (no-op) if ``scores`` is
    empty or contains only one class of label (NLL has no informative
    minimum in that degenerate case).
    """
    from scipy.optimize import minimize_scalar

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if len(scores) == 0 or len(np.unique(labels)) < 2:
        return 1.0

    logits = _logit(scores)

    def nll(t: float) -> float:
        p = _sigmoid(logits / t)
        p = np.clip(p, _EPS, 1.0 - _EPS)
        return float(-np.mean(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p)))

    result = minimize_scalar(nll, bounds=bounds, method="bounded")
    return float(result.x)


def fit_temperatures(
    per_class: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    bounds: tuple[float, float] = (0.05, 10.0),
) -> dict[str, float]:
    """``fit_temperature`` applied per class. ``per_class`` maps class_name -> (scores, labels)."""
    return {cls: fit_temperature(scores, labels, bounds=bounds)
            for cls, (scores, labels) in per_class.items()}


def load_temperatures(path: str | Path) -> dict[str, float]:
    """Load a ``{class_name: temperature}`` JSON file written by
    ``scripts/detection/evaluation/fit_calibration.py``."""
    p = Path(path)
    data = json.loads(p.read_text())
    return {str(k): float(v) for k, v in data.items()}


def save_temperatures(temperatures: dict[str, float], path: str | Path) -> None:
    Path(path).write_text(json.dumps(temperatures, indent=2, sort_keys=True))


def apply_calibration(
    detections: list[Detection],
    temperatures: dict[str, float],
    default_temperature: float = 1.0,
) -> list[Detection]:
    """Rescale each detection's ``score`` by its class's fitted temperature.

    Classes absent from ``temperatures`` fall back to ``default_temperature``
    (1.0 = unchanged). ``display_threshold`` is left untouched — calibration
    changes what the score *means*, not the operating point compared against it.
    """
    if not detections:
        return detections
    out: list[Detection] = []
    for d in detections:
        t = temperatures.get(d.class_name, default_temperature)
        if t == 1.0:
            out.append(d)
        else:
            out.append(replace(d, score=apply_temperature(d.score, t)))
    return out

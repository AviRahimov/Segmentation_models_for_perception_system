"""Causal exponential moving average over logit tensors."""
from __future__ import annotations

import torch

from .base import LogitsSmoother


class LogitsEMA(LogitsSmoother):
    """``y_t = alpha * x_t + (1 - alpha) * y_{t-1}``.

    Strictly causal: only the current frame and past EMA state are used.
    The first frame after construction or :meth:`reset` is passed through
    unchanged so the buffer initialises with real data, not zeros.
    """

    def __init__(self, alpha: float) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha: float = float(alpha)
        self._state: torch.Tensor | None = None

    @property
    def state(self) -> torch.Tensor | None:
        return self._state

    def update(self, logits: torch.Tensor) -> torch.Tensor:
        if self._state is None or self._state.shape != logits.shape:
            # First frame (or shape change after seek): initialise.
            self._state = logits.detach().clone()
            return self._state
        # Promote alpha to the same dtype/device as logits.
        a = torch.as_tensor(self.alpha, dtype=logits.dtype, device=logits.device)
        self._state = a * logits + (1.0 - a) * self._state
        return self._state

    def reset(self) -> None:
        self._state = None

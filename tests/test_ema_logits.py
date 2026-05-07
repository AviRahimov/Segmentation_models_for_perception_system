"""Tests for the causal logits EMA."""
import pytest
import torch

from perception.temporal.ema_logits import LogitsEMA


def test_first_frame_passthrough():
    ema = LogitsEMA(alpha=0.5)
    x = torch.randn(3, 8, 8)
    y = ema.update(x)
    assert torch.equal(x, y)


def test_two_frame_blend():
    ema = LogitsEMA(alpha=0.25)
    x0 = torch.zeros(2, 4, 4)
    x1 = torch.ones(2, 4, 4) * 4.0
    ema.update(x0)
    y = ema.update(x1)
    # 0.25 * 4 + 0.75 * 0 = 1.0 everywhere
    assert torch.allclose(y, torch.ones_like(y))


def test_alpha_validation():
    with pytest.raises(ValueError):
        LogitsEMA(alpha=0.0)
    with pytest.raises(ValueError):
        LogitsEMA(alpha=1.5)


def test_reset_drops_state():
    ema = LogitsEMA(alpha=0.5)
    x0 = torch.full((1, 2, 2), 10.0)
    ema.update(x0)
    assert ema.state is not None
    ema.reset()
    assert ema.state is None
    # After reset, the next update should pass through (just like frame 0).
    x1 = torch.full((1, 2, 2), -1.0)
    y = ema.update(x1)
    assert torch.equal(y, x1)


def test_shape_change_re_initializes():
    ema = LogitsEMA(alpha=0.5)
    ema.update(torch.zeros(1, 4, 4))
    y = ema.update(torch.ones(1, 5, 5) * 7.0)
    assert y.shape == (1, 5, 5)
    assert torch.allclose(y, torch.full_like(y, 7.0))


def test_causality_no_future_leakage():
    """Past EMA output must depend only on past inputs."""
    ema = LogitsEMA(alpha=0.5)
    x0 = torch.full((1, 2, 2), 1.0)
    x1 = torch.full((1, 2, 2), -100.0)  # huge future value
    y0 = ema.update(x0).clone()
    # Now feed a wildly different future value; y0 must be unchanged.
    ema.update(x1)
    assert torch.equal(y0, torch.full_like(y0, 1.0))

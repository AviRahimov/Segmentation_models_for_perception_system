"""Tests for the DDRNet-39 / GOOSE-12 wrapper.

These tests split into two layers:

1. Architecture-level (always run): build the vendored
   ``ddrnet_39_goose`` directly, run a forward pass, and check the
   shape/key-count invariants we rely on. No checkpoint required.

2. Wrapper-level with real weights (skipped when the ~250 MB checkpoint
   file is absent): construct :class:`DDRNetSemanticModel`, warm it up
   on the user-class set, and verify that ``predict_logits`` returns
   normalised probabilities of the expected shape over a random BGR
   frame. This is the smoke test that catches state-dict regressions
   in CI environments where the checkpoint is provisioned.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from perception.config.schema import ClassDef
from perception.models.semantic._vendored.ddrnet39_goose import (
    BasicResNetBlock,
    Bottleneck,
    DDRNet,
    ddrnet_39_goose,
)


# --------------------------------------------------------------------------- #
# Architecture invariants (no checkpoint needed)                               #
# --------------------------------------------------------------------------- #


def test_ddrnet_39_goose_state_dict_size():
    """The architecture must expose exactly 501 state-dict keys for 12 classes.

    This is the precise count of the ``ddrnet_category_512.pth``
    checkpoint published by the GOOSE benchmark; if a future code
    change drops or adds tensors we want the test to break loudly so
    we don't silently lose strict-load capability.
    """
    m = ddrnet_39_goose(num_classes=12, use_aux_heads=False)
    sd = m.state_dict()
    assert len(sd) == 501, f"expected 501 keys, got {len(sd)}"
    # Spot-check a few distinctive super_gradients-flavour names.
    expected_keys = [
        "_backbone.stem.0.0.weight",
        "_backbone.layer3.1.2.conv2.weight",  # second sub-stage of layer3 (layer3_repeats=2)
        # Branch 4 has stride=0 (AdaptiveAvgPool, no params) so the BN at
        # index 1 is the first weighted tensor.
        "spp.branches.4.down_scale.1.weight",
        "compression3.1.0.weight",             # ModuleList index 1 -> matches layer3_repeats=2
        "layer3_skip.0.0.conv1.weight",
        "layer4_skip.0.conv1.weight",
        "layer5_skip.0.conv1.weight",
        "final_layer.conv2.weight",
    ]
    for k in expected_keys:
        assert k in sd, f"missing expected key: {k}"
    # The final classifier conv must have num_classes output channels.
    assert sd["final_layer.conv2.weight"].shape[0] == 12


def test_ddrnet_goose_perm_is_bijection():
    from perception.models.semantic.ddrnet_goose_perm import DOC_SLOT_TO_RAW_CHANNEL

    assert len(DOC_SLOT_TO_RAW_CHANNEL) == 12
    assert set(DOC_SLOT_TO_RAW_CHANNEL) == set(range(12))


def test_ddrnet_39_goose_forward_shape_and_param_count():
    """Forward must give 12-channel logits at the *full* input resolution.

    The model's internal stride is 1/8, but its ``final_layer`` upscales
    by 8, so the network output is at the input H, W. Param count is
    asserted to lock in the vendored architecture identity (32.4M).
    """
    m = ddrnet_39_goose(num_classes=12).eval()
    n_learnable = sum(p.numel() for p in m.parameters())
    # 32,358,476 learnable params; the 32,406,943 figure quoted elsewhere
    # is the *state-dict* total which also counts BN running stats.
    assert n_learnable == 32_358_476, f"unexpected param count {n_learnable:,}"
    n_state = sum(t.numel() for t in m.state_dict().values())
    assert n_state == 32_406_943, f"unexpected state-dict total {n_state:,}"

    with torch.inference_mode():
        x = torch.zeros(1, 3, 256, 256)
        y = m(x)
    assert y.shape == (1, 12, 256, 256)


def test_basic_blocks_state_dict_layout():
    """Sanity-check the residual-block state-dict layout vendored from
    super_gradients (no DropPath, ``shortcut`` -> 0/1 indexed Sequential)."""
    blk = BasicResNetBlock(in_planes=32, planes=64, stride=2)
    sd = blk.state_dict()
    assert "conv1.weight" in sd
    assert "bn1.weight" in sd
    assert "shortcut.0.weight" in sd  # Conv2d in shortcut
    assert "shortcut.1.weight" in sd  # BatchNorm2d in shortcut
    # No drop_path tensors (we replaced super_gradients' DropPath with Identity).
    assert not any(k.startswith("drop_path") for k in sd)

    bot = Bottleneck(in_planes=128, planes=128, expansion=2)
    bsd = bot.state_dict()
    assert {"conv1.weight", "bn1.weight", "conv2.weight", "bn2.weight",
            "conv3.weight", "bn3.weight"}.issubset(bsd)


# --------------------------------------------------------------------------- #
# Wrapper-level smoke test (real weights — skipped if absent)                  #
# --------------------------------------------------------------------------- #

_CKPT = Path(__file__).resolve().parents[1] / "weights" / "ddrnet_category_512.pth"
_HAS_CKPT = _CKPT.exists()
# Skip the smoke test on CPU-only CI as well; DDRNet is a non-trivial
# (32M-param) net and CPU+fp32 inference would slow the suite down for
# no extra coverage.
_HAS_CUDA = torch.cuda.is_available()
_RUN_SMOKE = _HAS_CKPT and _HAS_CUDA


@pytest.mark.skipif(
    not _RUN_SMOKE,
    reason=f"requires CUDA + {_CKPT.name} on disk",
)
def test_ddrnet_wrapper_predict_logits_shape_and_normalisation():
    """End-to-end smoke test: real weights, real warmup, real frame."""
    from perception.models.semantic.ddrnet import DDRNetSemanticModel

    classes = (
        ClassDef(name="road_ground", text_prompt="", display_mode="mask_only",
                 color_rgb=(0, 0, 255), is_semantic=True,
                 native_indices={"goose_12": (5,)}),
        ClassDef(name="grass", text_prompt="", display_mode="mask_only",
                 color_rgb=(0, 200, 0), is_semantic=True,
                 native_indices={"goose_12": (0,)}),
        ClassDef(name="sky", text_prompt="", display_mode="mask_only",
                 color_rgb=(200, 200, 200), is_semantic=True,
                 native_indices={"goose_12": (8,)}),
    )

    model = DDRNetSemanticModel(weights=str(_CKPT), device="cuda", fp16=True)
    model.warmup(classes)
    assert model.class_names == ("road_ground", "grass", "sky")

    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    out = model.predict_logits(frame)

    assert out.shape == (3, 240, 320)
    # Probabilities along channels should be in [0, 1] (these are
    # *merged* probabilities; the unmerged 12-class softmax sums to 1
    # but the merged 3-class slice must sum to <=1).
    assert torch.isfinite(out).all().item()
    assert (out >= 0).all().item()
    assert (out.sum(dim=0) <= 1.0 + 1e-3).all().item()


@pytest.mark.skipif(
    not _RUN_SMOKE,
    reason=f"requires CUDA + {_CKPT.name} on disk",
)
def test_ddrnet_wrapper_rejects_classes_without_goose12_indices():
    """A user class without ``goose_12`` indices must produce a clear error."""
    from perception.models.semantic.ddrnet import DDRNetSemanticModel

    bad = (
        ClassDef(name="ade-only", text_prompt="", display_mode="mask_only",
                 color_rgb=(0, 0, 0), is_semantic=True,
                 native_indices={"ade20k": (6,)}),  # no goose_12
    )
    model = DDRNetSemanticModel(weights=str(_CKPT), device="cuda", fp16=True)
    with pytest.raises(ValueError, match="goose_12"):
        model.warmup(bad)

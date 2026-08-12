"""Training-time fine-grained -> ORFD-3-class softmax-then-merge.

Generalizes ``src/perception/models/semantic/segformer.py``'s inference-only
LUT/einsum pattern (``torch.einsum("cu,chw->uhw", lut, softmax(logits))``,
single unbatched frame) to a batched training-time version. Deliberately new
code rather than an import: that module's LUT builder is hardcoded to the
``"ade20k"`` catalogue key and is gated off entirely once a checkpoint is
fine-tuned (``self._fine_tuned``) -- it isn't reachable from the plain
``SegformerForSemanticSegmentation`` models the training scripts build.

Used by:
  * the coarse-auxiliary loss (Stage 4b) -- a cheap regularizer teaching the
    fine-grained head the coarse structure we actually deploy.
  * SAM3-teacher distillation (Stage 4b) -- collapsing SAM3's fine-grained
    soft-probability maps into (B, 3, H, W) before ``train_distill.py``'s
    already-generic ``kd_loss()``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _class_catalogues_loader import COARSE_CLASSES, FINE_TO_COARSE, SAM3_FINEGRAINED_NAMES  # noqa: E402

__all__ = ["COARSE_CLASSES", "FINE_TO_COARSE", "SAM3_FINEGRAINED_NAMES",
           "build_fine_to_coarse_lut", "merge_fine_logits_to_coarse_probs"]


def build_fine_to_coarse_lut(
    fine_names: tuple[str, ...] = SAM3_FINEGRAINED_NAMES,
    coarse_names: tuple[str, ...] = COARSE_CLASSES,
) -> torch.Tensor:
    """Binary (C_fine, C_coarse) LUT: lut[i, j] = 1 iff fine_names[i] maps to coarse_names[j]."""
    lut = torch.zeros(len(fine_names), len(coarse_names), dtype=torch.float32)
    for i, fine_name in enumerate(fine_names):
        coarse_name = FINE_TO_COARSE[fine_name]
        j = coarse_names.index(coarse_name)
        lut[i, j] = 1.0
    return lut


def merge_fine_logits_to_coarse_probs(fine_logits: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """Softmax-then-merge: (B, C_fine, H, W) logits -> (B, C_coarse, H, W) probabilities.

    Softmax over the native fine channels first (proper posterior), then sum
    via the LUT -- same rationale as segformer.py's inference-time merge:
    per-channel logit biases differ, so summing probabilities (not logits)
    is the correct merge.
    """
    probs = torch.softmax(fine_logits.float(), dim=1).to(fine_logits.dtype)
    return torch.einsum("cu,bchw->buhw", lut.to(fine_logits.device, fine_logits.dtype), probs)

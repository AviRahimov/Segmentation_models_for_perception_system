"""Shared SegFormer checkpoint-loading helpers, used across
scripts/segmentation/{training,evaluation,optimization}/.

Every script in this tree was independently repeating the same
torch.load -> get("net", ckpt) -> _remap_segformer_keys() sequence before
either building a fresh HF model or overlaying weights onto one already
built via train_orfd.build_segformer(). One copy (benchmark_orfd.py) had
drifted and skipped _remap_segformer_keys() entirely -- per segformer.py's
own docstring, some transformers builds (e.g. on Jetson aarch64) use older
key naming, so that copy could strict=True-fail loading a checkpoint every
other script here loads fine.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_remapped_state_dict(checkpoint: str | Path) -> dict[str, Any]:
    """torch.load a training checkpoint and return its (key-remapped) model
    state_dict -- the exact sequence every caller in this directory needs,
    whether it goes on to build a fresh model or overlay an existing one."""
    from perception.models.semantic.segformer import _remap_segformer_keys

    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    state_dict = ckpt.get("net", ckpt) if isinstance(ckpt, dict) else ckpt
    return _remap_segformer_keys(state_dict)


def build_segformer_from_checkpoint(
    checkpoint: str | Path,
    device: str,
    resolution: int | None = None,
    hf_base: str = "nvidia/segformer-b2-finetuned-ade-512-512",
    fp16: bool = False,
):
    """Build a SegFormer model from scratch (HF from_pretrained) and load a
    fine-tuned checkpoint's weights into it. Returns (model, processor,
    n_labels) -- n_labels is auto-detected from the checkpoint so this works
    for both 2-class and 3-class (sky) checkpoints.

    For callers that already have a model built via train_orfd's own
    build_segformer() (train_qat.py, train_sparse.py) and just need to
    overlay a checkpoint's weights, use load_remapped_state_dict() directly
    instead -- this function always constructs a brand new model.
    """
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

    ckpt_path = Path(checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = load_remapped_state_dict(ckpt_path)
    n_labels = state_dict["decode_head.classifier.weight"].shape[0]

    processor = SegformerImageProcessor.from_pretrained(hf_base)
    if resolution is not None:
        processor.size = {"height": resolution, "width": resolution}

    model = SegformerForSemanticSegmentation.from_pretrained(
        hf_base, num_labels=n_labels, ignore_mismatched_sizes=True
    )
    model.load_state_dict(state_dict, strict=True)
    model = model.eval().to(device)
    if fp16:
        model = model.half()

    return model, processor, n_labels

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

_remap_segformer_keys below is an intentional, self-contained duplicate of
src/perception/models/semantic/segformer.py's function of the same name
(kept logic-identical -- update both if the key-remap rules ever change).
Importing the real one would pull in perception.models.semantic.base and
perception.config.schema, neither of which exist in the minimal on-device
Jetson checkout (see JETSON.md's "segformer_repo" tree: only
src/perception/datasets/ is transferred, deliberately no models/ package) --
this module needs to run standalone there via benchmark_jetson.py's
--pytorch-ref, same rationale as the RF-DETR Jetson scripts being
deliberately dependency-minimal.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _remap_segformer_keys(sd: dict) -> dict:
    """Remap checkpoint keys from old transformers format to current format.

    Handles the key renaming between transformers versions:
      segformer.encoder.block/patch_embeddings/layer_norm → segformer.stages.*
      attention.self.{query,key,value} → attention.{q,k,v}_proj
      attention.output.dense → attention.o_proj
      attention.self.sr → attention.sequence_reduction.sequence_reduction
      layer_norm_{1,2} → layernorm_{before,after}
      mlp.dense{1,2} → mlp.fc{1,2}
      decode_head.linear_c.{i} → decode_head.linear_projections.{i}

    NOTE: some builds of transformers 4.46 (e.g. Jetson aarch64) still use the
    old 'segformer.encoder.*' key naming.  We detect this via the presence of
    the 'SegformerStage' class, which only exists in the new API.  If the
    installed transformers uses old naming, the checkpoint already matches and
    no remapping is needed.
    """
    if not any(k.startswith("segformer.encoder.") for k in sd):
        return sd  # already in current (stages) format

    # Detect whether this transformers build uses new 'stages' or old 'encoder' naming.
    try:
        import transformers.models.segformer.modeling_segformer as _mseg
        _uses_stages_api = hasattr(_mseg, "SegformerStage")
    except Exception:
        _uses_stages_api = True  # assume modern if detection fails

    if not _uses_stages_api:
        # This build uses 'segformer.encoder.*' naming — checkpoint already matches.
        return sd

    logger.info("SegFormer checkpoint uses old key format — remapping to current transformers API.")
    out: dict = {}
    for k, v in sd.items():
        nk = k

        # decode_head.linear_c.{i}.* → decode_head.linear_projections.{i}.*
        nk = re.sub(r"^decode_head\.linear_c\.(\d+)\.",
                    r"decode_head.linear_projections.\1.", nk)

        # segformer.encoder.patch_embeddings.{i}.* → segformer.stages.{i}.patch_embeddings.*
        nk = re.sub(r"^segformer\.encoder\.patch_embeddings\.(\d+)\.",
                    r"segformer.stages.\1.patch_embeddings.", nk)

        # segformer.encoder.layer_norm.{i}.* → segformer.stages.{i}.layer_norm.*
        nk = re.sub(r"^segformer\.encoder\.layer_norm\.(\d+)\.",
                    r"segformer.stages.\1.layer_norm.", nk)

        # segformer.encoder.block.{i}.{j}.* → segformer.stages.{i}.blocks.{j}.*
        m = re.match(r"^segformer\.encoder\.block\.(\d+)\.(\d+)\.(.+)$", nk)
        if m:
            si, bj, rest = m.group(1), m.group(2), m.group(3)
            rest = re.sub(r"^attention\.self\.query\.",  "attention.q_proj.", rest)
            rest = re.sub(r"^attention\.self\.key\.",    "attention.k_proj.", rest)
            rest = re.sub(r"^attention\.self\.value\.",  "attention.v_proj.", rest)
            rest = re.sub(r"^attention\.output\.dense\.", "attention.o_proj.", rest)
            rest = re.sub(r"^attention\.self\.sr\.",
                          "attention.sequence_reduction.sequence_reduction.", rest)
            rest = re.sub(r"^attention\.self\.layer_norm\.",
                          "attention.sequence_reduction.layer_norm.", rest)
            rest = re.sub(r"^layer_norm_1\.", "layernorm_before.", rest)
            rest = re.sub(r"^layer_norm_2\.", "layernorm_after.", rest)
            rest = re.sub(r"^mlp\.dense1\.", "mlp.fc1.", rest)
            rest = re.sub(r"^mlp\.dense2\.", "mlp.fc2.", rest)
            nk = f"segformer.stages.{si}.blocks.{bj}.{rest}"

        out[nk] = v
    return out


def load_remapped_state_dict(checkpoint: str | Path) -> dict[str, Any]:
    """torch.load a training checkpoint and return its (key-remapped) model
    state_dict -- the exact sequence every caller in this directory needs,
    whether it goes on to build a fresh model or overlay an existing one."""
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

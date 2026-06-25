"""SegFormer semantic segmentation wrapper (B2 / B4 / fine-tuned).

Standard (ADE20K) mode
----------------------
The model is closed-vocabulary on ADE20K (150 channels).  The configured
``ade20k_indices`` define a merge LUT (shape ``[150, C_user]``) that
combines native predictions into the user's class set.

Critically, we softmax over the 150 native channels **before** applying
the LUT, so the merged tensor is a sum of *posterior probabilities* per
user class, not raw logits.  Summing logits would be wrong: ADE20K's
per-channel logit biases differ by several units, so a class with one
high-bias channel (e.g. ``grass`` = #9) can systematically outscore a
class with multiple lower-bias channels (e.g. ``sand_gravel`` =
#46+#91), regardless of what the model actually "sees".  Probabilities,
being normalised to a simplex, are comparable across channels, so the
sum gives the model's actual posterior over the user-defined class set.

Fine-tuned mode
---------------
When ``num_classes`` is passed (or when the loaded checkpoint has
``config.num_labels != 150``), the LUT step is skipped entirely.  The
model already outputs ``num_classes`` channels that map directly to the
user's class set; ``warmup()`` derives ``class_names`` from the semantic
classes listed in the config (in order: config class 0 → model class 0,
etc.).

The merged / direct tensor (still per-pixel scores) is what the EMA
smoother operates on and what the renderer argmaxes downstream.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

from ...config.schema import ClassDef
from ..backends.base import InferenceBackend
from .base import SemanticModel

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

# Maps config model name → canonical HuggingFace model ID.
# Used when weights is a local .pth file: load the processor and architecture
# from HuggingFace, then overlay the local state dict.
_HF_BASES: dict[str, str] = {
    "segformer-b0":  "nvidia/segformer-b0-finetuned-ade-512-512",
    "segformer_b0":  "nvidia/segformer-b0-finetuned-ade-512-512",
    "segformer-b1":  "nvidia/segformer-b1-finetuned-ade-512-512",
    "segformer_b1":  "nvidia/segformer-b1-finetuned-ade-512-512",
    "segformer-b2":  "nvidia/segformer-b2-finetuned-ade-512-512",
    "segformer_b2":  "nvidia/segformer-b2-finetuned-ade-512-512",
    "segformer-b4":  "nvidia/segformer-b4-finetuned-ade-512-512",
    "segformer_b4":  "nvidia/segformer-b4-finetuned-ade-512-512",
}


class SegFormerSemanticModel(SemanticModel):
    def __init__(
        self,
        weights: str = "nvidia/segformer-b2-finetuned-ade-512-512",
        name: str = "segformer-b2",
        backend: InferenceBackend | None = None,
        device: str = "cuda",
        fp16: bool = True,
        num_classes: int | None = None,
        processor_size: int | None = None,
        trt_engine_path: str = "",
    ) -> None:
        # Lazy import for environments without transformers (e.g. unit tests).
        from transformers import (  # type: ignore
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )

        self._device = device
        self._fp16 = bool(fp16) and isinstance(device, str) and device.startswith("cuda")

        # When weights is a local .pth file (e.g. fine-tuned ORFD checkpoint),
        # from_pretrained cannot read it as a JSON config. Instead: load the
        # processor and architecture from the canonical HF model, then strict-load
        # the local state dict on top.
        _is_local = Path(weights).suffix == ".pth"
        hf_base = _HF_BASES.get(name, _HF_BASES["segformer-b2"])
        processor_source = hf_base if _is_local else weights
        self._processor = SegformerImageProcessor.from_pretrained(processor_source)
        if processor_size is not None:
            self._processor.size = {"height": processor_size, "width": processor_size}
            logger.info("SegFormer processor input size overridden to %dx%d", processor_size, processor_size)

        if _is_local:
            ckpt = torch.load(weights, map_location="cpu", weights_only=True)
            state_dict = ckpt.get("net", ckpt) if isinstance(ckpt, dict) else ckpt
            state_dict = _remap_segformer_keys(state_dict)
            # Auto-detect from checkpoint so old 2-class and new 3-class checkpoints
            # both load correctly regardless of what num_classes the caller passes.
            n_labels = state_dict["decode_head.classifier.weight"].shape[0]
            if num_classes is not None and num_classes != n_labels:
                logger.warning(
                    "num_classes=%d passed but checkpoint has %d output classes; "
                    "using checkpoint value.",
                    num_classes, n_labels,
                )
            self._model = SegformerForSemanticSegmentation.from_pretrained(
                hf_base,
                num_labels=n_labels,
                ignore_mismatched_sizes=True,
            )
            self._model.load_state_dict(state_dict, strict=True)
            logger.info("SegFormer loaded from local checkpoint %s (%d classes, strict)",
                        weights, n_labels)
        elif num_classes is not None and num_classes != 150:
            # Fine-tuned checkpoint at a HuggingFace path / directory.
            self._model = SegformerForSemanticSegmentation.from_pretrained(
                weights,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            self._model = SegformerForSemanticSegmentation.from_pretrained(weights)

        self._model.eval()

        # Detect whether the loaded model is in fine-tuned (direct) mode.
        # A freshly loaded fine-tuned checkpoint knows its own num_labels.
        loaded_num_labels = int(getattr(self._model.config, "num_labels", 150))
        self._fine_tuned: bool = loaded_num_labels != 150
        # Cache before backend.prepare() may replace self._model with a TRT wrapper.
        self._num_labels: int = loaded_num_labels

        if backend is not None:
            self._model = backend.prepare(
                self._model,
                device=self._device,
                fp16=self._fp16,
                engine_path=trt_engine_path,
            )
        else:
            self._model = self._model.to(self._device)
            if self._fp16:
                self._model = self._model.half()

        self._semantic_classes: list[ClassDef] = []
        self._lut: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    def warmup(self, classes: Sequence[ClassDef]) -> None:
        sem = [c for c in classes if c.is_semantic]
        self._semantic_classes = sem

        if not sem:
            self._lut = None
            logger.info("SegFormer: no semantic classes configured.")
            return

        if self._fine_tuned:
            # Fine-tuned mode: model outputs directly in user-class space, no LUT needed.
            n_model = int(
                getattr(getattr(self._model, "config", None), "num_labels",
                        getattr(self, "_num_labels", len(sem)))
            )
            if len(sem) < n_model:
                raise ValueError(
                    f"SegFormerSemanticModel (fine-tuned): config has {len(sem)} "
                    f"semantic classes but the checkpoint has {n_model} output "
                    f"channels. Add the missing classes to the config."
                )
            if len(sem) > n_model:
                # Config has more classes than model outputs (e.g. 3-class config with
                # a 2-class checkpoint). Extra classes beyond index n_model-1 will
                # never be argmaxed and therefore never rendered — that is fine.
                logger.warning(
                    "SegFormer (fine-tuned): config has %d semantic classes but "
                    "checkpoint has %d output channels; classes beyond index %d "
                    "will never be predicted.",
                    len(sem), n_model, n_model - 1,
                )
            self._lut = None
            logger.info(
                "SegFormer (fine-tuned) warmed up: %d model channels, %d config classes.",
                n_model, len(sem),
            )
            return

        # Standard ADE20K + LUT merge path.
        n_ade = int(getattr(self._model.config, "num_labels", 150))
        lut = torch.zeros(n_ade, len(sem), dtype=torch.float32)
        for j, c in enumerate(sem):
            for i in c.ade20k_indices:
                if not 0 <= int(i) < n_ade:
                    raise ValueError(
                        f"Class {c.name!r}: ade20k index {i} out of range [0, {n_ade})"
                    )
                lut[int(i), j] = 1.0

        param_dtype = next(self._model.parameters()).dtype
        self._lut = lut.to(self._device).to(param_dtype)
        logger.info(
            "SegFormer warmed up: %d user classes derived from %d ADE20K channels",
            len(sem), int(self._lut.sum().item()),
        )

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self._semantic_classes)

    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def predict_logits(self, frame_bgr: np.ndarray) -> torch.Tensor:
        if not self._semantic_classes:
            raise RuntimeError(
                "SegFormerSemanticModel.predict_logits called before warmup() "
                "(or with no semantic classes configured)."
            )
        if not self._fine_tuned and self._lut is None:
            raise RuntimeError(
                "SegFormerSemanticModel.predict_logits: no LUT built. "
                "Configure semantic classes with ade20k_indices in config.yaml, "
                "or set models.semantic.num_classes for a fine-tuned model."
            )

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=rgb, return_tensors="pt")
        pixel_values: torch.Tensor = inputs["pixel_values"].to(self._device)
        if self._fp16:
            pixel_values = pixel_values.half()

        outputs = self._model(pixel_values=pixel_values)
        logits = outputs.logits  # (1, C, H/4, W/4)

        logits = torch.nn.functional.interpolate(
            logits, size=(h // 2, w // 2), mode="bilinear", align_corners=False
        )[0]  # (C, H//2, W//2)

        if self._fine_tuned:
            # Direct mode: softmax over the model's native output classes.
            probs = torch.softmax(logits.float(), dim=0).to(logits.dtype)
            return probs  # (C_user, H, W)

        # Standard ADE20K LUT merge:
        # Softmax over 150 native channels first, then sum via LUT.
        # Compute softmax in fp32 for numerical stability under fp16 inputs.
        probs = torch.softmax(logits.float(), dim=0).to(logits.dtype)
        merged = torch.einsum("cu,chw->uhw", self._lut, probs)  # (C_user, H, W)
        return merged

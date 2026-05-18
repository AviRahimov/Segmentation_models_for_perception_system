"""DDRNet-39 semantic segmentation wrapper (GOOSE-12 head).

Architecture
------------

The GOOSE benchmark publishes two image-segmentation checkpoints:

* ``ddrnet_category_512.pth``  — 12-class "category" head (the one we
  use for cross-model comparison; that's all the user's class set
  exposes via ``native_indices.goose_12``).
* ``ddrnet_category_1024.pth`` — same architecture, trained at 1024×768
  inputs.

Both share the same architecture: **DDRNet-39** as built by Deci-AI's
``super_gradients`` package (``Models.DDRNET_39``,
``BasicResNetBlock`` skip blocks, ``Bottleneck`` (expansion=2) layer-5
blocks, ``layer3_repeats=2``, ``planes=64``, ``highres_planes=128``,
``head_planes=256``, DAPPM with kernel-sizes ``[1,5,9,17,0]``). The
self-contained re-implementation lives in
``perception.models.semantic._vendored.ddrnet39_goose``; see its module
docstring for the upstream pointer and the local edits.

The state-dict from the official ``.pth`` strict-loads
(``load_state_dict(strict=True)``) — every one of the 501 entries
matches by name and shape. There is no graceful-degradation
``strict=False`` fallback in this wrapper: if the checkpoint structure
ever changes, fail loud rather than silently masking weight loss.

Inference contract
------------------

Mirrors :class:`SegFormerSemanticModel`:

* Input  : BGR ``np.ndarray`` of shape ``(H, W, 3)``, ``uint8``.
* Forward: convert BGR -> RGB, normalise with ImageNet mean/std,
  resize to 512×512 (the checkpoint's training resolution), run the
  model, bilinear upsample logits to ``(H, W)``, permute the 12 logits
  into **canonical** GOOSE-12 order (see :mod:`ddrnet_goose_perm`),
  softmax over those channels in fp32, then merge into the user-class
  set via the ``goose_12`` LUT built at warmup time.
* Output : torch tensor ``(C_user, H, W)`` in the model's parameter
  dtype, parallel to ``self.class_names``.

The softmax happens *before* the LUT for the same numerical reason as
SegFormer's wrapper: summing posteriors gives the model's actual
posterior over the user-class set; summing raw logits would let
high-bias native channels (e.g. GOOSE's ``vegetation`` index 0) drown
out classes that have to be summed across multiple lower-bias channels.
"""
from __future__ import annotations

import logging
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ...config.schema import ClassDef
from ..backends.base import InferenceBackend
from ._class_catalogues import GOOSE_12_NAMES
from ._vendored.ddrnet39_goose import ddrnet_39_goose
from .base import SemanticModel
from .ddrnet_goose_perm import DOC_SLOT_TO_RAW_CHANNEL

logger = logging.getLogger(__name__)


# ImageNet mean/std the GOOSE training scripts use (see
# ``goose_dataset/image_processing/goosetools/data.py``).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Inference resolution: matches the ``ddrnet_category_512.pth`` filename.
_INFERENCE_HW = (512, 512)


class DDRNetSemanticModel(SemanticModel):
    """DDRNet-39 wrapper — GOOSE-12 head (default) or fine-tuned N-class head.

    When ``num_classes`` is ``None`` (default), the model loads the standard
    12-class GOOSE checkpoint and merges predictions via a ``goose_12`` LUT
    (same pattern as :class:`SegFormerSemanticModel`).

    When ``num_classes`` is set to a positive integer other than 12, the model
    is built with that many output channels, the checkpoint is loaded with
    ``strict=True`` (the checkpoint must already have the matching head), and
    ``warmup()`` derives ``class_names`` from the config semantic classes in
    order without building a LUT.
    """

    NATIVE_CATALOGUE: str = "goose_12"
    NUM_NATIVE_CLASSES: int = len(GOOSE_12_NAMES)

    def __init__(
        self,
        weights: str = "weights/ddrnet_category_512.pth",
        backend: InferenceBackend | None = None,
        device: str = "cuda",
        fp16: bool = True,
        num_classes: int | None = None,
    ) -> None:
        self._weights = weights
        self._device = device
        self._fp16 = bool(fp16) and isinstance(device, str) and device.startswith("cuda")

        # Load checkpoint first so we can auto-detect the head size.
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "net" in ckpt:
            state_dict = ckpt["net"]
            self._ckpt_acc = float(ckpt.get("acc", float("nan")))
            self._ckpt_epoch = int(ckpt.get("epoch", -1))
        else:
            state_dict = ckpt
            self._ckpt_acc = float("nan")
            self._ckpt_epoch = -1

        # Auto-detect head size from the final_layer weights so that old 2-class
        # and new 3-class fine-tuned checkpoints both load correctly regardless of
        # what num_classes the caller passes.
        _fl_keys = sorted(
            k for k in state_dict if k.startswith("final_layer.") and k.endswith(".weight")
        )
        if _fl_keys:
            detected = state_dict[_fl_keys[-1]].shape[0]
            if num_classes is not None and num_classes != detected:
                logger.warning(
                    "num_classes=%d passed but checkpoint final_layer has %d output "
                    "channels; using checkpoint value.",
                    num_classes, detected,
                )
            head_classes = detected
        else:
            head_classes = num_classes if num_classes is not None else self.NUM_NATIVE_CLASSES

        self._fine_tuned: bool = head_classes != self.NUM_NATIVE_CLASSES

        # Build the architecture and strict-load the checkpoint.
        self._model = ddrnet_39_goose(num_classes=head_classes, use_aux_heads=False)
        self._model.load_state_dict(state_dict, strict=True)
        self._model.eval()

        if backend is not None:
            self._model = backend.prepare(self._model, device=self._device, fp16=self._fp16)
        else:
            self._model = self._model.to(self._device)
            if self._fp16:
                self._model = self._model.half()

        # Precomputed normalisation tensors, on the same device + dtype
        # as the model parameters. Keeping these as module attributes
        # means the per-frame fast path is allocation-free.
        param_dtype = next(self._model.parameters()).dtype
        self._mean = torch.tensor(_IMAGENET_MEAN, device=self._device, dtype=param_dtype).view(1, 3, 1, 1)
        self._std = torch.tensor(_IMAGENET_STD, device=self._device, dtype=param_dtype).view(1, 3, 1, 1)

        self._semantic_classes: list[ClassDef] = []
        self._lut: torch.Tensor | None = None
        logger.info(
            "DDRNet-39 loaded from %s (acc=%.4f at epoch=%d, strict-load OK).",
            weights, self._ckpt_acc, self._ckpt_epoch,
        )

    # ------------------------------------------------------------------ #
    def warmup(self, classes: Sequence[ClassDef]) -> None:
        """Build the GOOSE-12 -> user-class merge LUT (standard mode),
        or validate class count (fine-tuned mode).
        """
        sem = [c for c in classes if c.is_semantic]
        self._semantic_classes = sem
        if not sem:
            self._lut = None
            logger.info("DDRNet: no semantic classes configured.")
            return

        if self._fine_tuned:
            n_model = self._model.num_classes
            if len(sem) < n_model:
                raise ValueError(
                    f"DDRNetSemanticModel (fine-tuned): config has {len(sem)} "
                    f"semantic classes but the checkpoint has {n_model} output "
                    f"channels. Add the missing classes to the config."
                )
            if len(sem) > n_model:
                logger.warning(
                    "DDRNet (fine-tuned): config has %d semantic classes but "
                    "checkpoint has %d output channels; classes beyond index %d "
                    "will never be predicted.",
                    len(sem), n_model, n_model - 1,
                )
            self._lut = None
            logger.info(
                "DDRNet (fine-tuned) warmed up: %d model channels, %d config classes.",
                n_model, len(sem),
            )
            return

        n_native = self.NUM_NATIVE_CLASSES
        lut = torch.zeros(n_native, len(sem), dtype=torch.float32)
        for j, c in enumerate(sem):
            idx = c.native_indices.get(self.NATIVE_CATALOGUE, ())
            if not idx:
                raise ValueError(
                    f"DDRNetSemanticModel: class {c.name!r} does not define "
                    f"native_indices.{self.NATIVE_CATALOGUE}; this wrapper "
                    f"cannot serve a class without GOOSE-12 indices."
                )
            for i in idx:
                if not 0 <= int(i) < n_native:
                    raise ValueError(
                        f"DDRNetSemanticModel: class {c.name!r} goose_12 "
                        f"index {i} out of range [0, {n_native})"
                    )
                lut[int(i), j] = 1.0

        param_dtype = next(self._model.parameters()).dtype
        self._lut = lut.to(self._device).to(param_dtype)
        logger.info(
            "DDRNet warmed up: %d user classes derived from %d GOOSE-12 channels",
            len(sem), int(lut.sum().item()),
        )

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self._semantic_classes)

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def raw_logits_hw(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """12 raw head channels at native (H,W); order is checkpoint-specific."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, _INFERENCE_HW, interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(resized).to(self._device).permute(2, 0, 1).unsqueeze(0)
        x = x.to(self._mean.dtype) / 255.0
        x = (x - self._mean) / self._std
        logits = self._model(x)
        return F.interpolate(
            logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0]

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def predict_logits(self, frame_bgr: np.ndarray) -> torch.Tensor:
        if not self._semantic_classes:
            raise RuntimeError(
                "DDRNetSemanticModel.predict_logits called before warmup() "
                "(or with no semantic classes configured)."
            )

        logits_raw = self.raw_logits_hw(frame_bgr)

        if self._fine_tuned:
            # Fine-tuned mode: model already outputs user-class channels.
            probs = torch.softmax(logits_raw.float(), dim=0).to(logits_raw.dtype)
            return probs  # (C_user, H, W)

        if self._lut is None:
            raise RuntimeError(
                "DDRNetSemanticModel.predict_logits: no GOOSE-12 LUT built. "
                "Configure semantic classes with native_indices.goose_12 in config.yaml."
            )

        # Standard GOOSE-12 LUT merge path.
        perm = torch.tensor(
            DOC_SLOT_TO_RAW_CHANNEL,
            device=logits_raw.device,
            dtype=torch.long,
        )
        logits = logits_raw.index_select(0, perm)

        # Softmax across 12 GOOSE channels in fp32 for numerical
        # stability under fp16 input, then merge into user classes via
        # the LUT (same pattern as the SegFormer wrapper).
        probs = torch.softmax(logits.float(), dim=0).to(logits.dtype)
        merged = torch.einsum("cu,chw->uhw", self._lut, probs)  # (C_user, H, W)
        return merged

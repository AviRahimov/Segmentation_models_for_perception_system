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
    """DDRNet-39 wrapper with a 12-class GOOSE head."""

    NATIVE_CATALOGUE: str = "goose_12"
    NUM_NATIVE_CLASSES: int = len(GOOSE_12_NAMES)

    def __init__(
        self,
        weights: str = "weights/ddrnet_category_512.pth",
        backend: InferenceBackend | None = None,
        device: str = "cuda",
        fp16: bool = True,
    ) -> None:
        self._weights = weights
        self._device = device
        self._fp16 = bool(fp16) and isinstance(device, str) and device.startswith("cuda")

        # Build the architecture and strict-load the published checkpoint.
        # The constructor is cheap (a few MB on CPU); strict=True is the
        # whole point of vendoring the super_gradients layout.
        self._model = ddrnet_39_goose(num_classes=self.NUM_NATIVE_CLASSES, use_aux_heads=False)
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        # GOOSE checkpoints wrap the state-dict under "net". Be tolerant
        # of the rare bare-state-dict export (e.g. for deployment).
        if isinstance(ckpt, dict) and "net" in ckpt:
            state_dict = ckpt["net"]
            self._ckpt_acc = float(ckpt.get("acc", float("nan")))
            self._ckpt_epoch = int(ckpt.get("epoch", -1))
        else:
            state_dict = ckpt
            self._ckpt_acc = float("nan")
            self._ckpt_epoch = -1
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
        """Build the GOOSE-12 -> user-class merge LUT.

        Each semantic user class must declare ``native_indices.goose_12``
        in ``config.yaml``. Classes that only declare ``ade20k`` indices
        cannot be served by this wrapper and trigger a clear runtime
        error.
        """
        sem = [c for c in classes if c.is_semantic]
        self._semantic_classes = sem
        if not sem:
            self._lut = None
            logger.info("DDRNet: no semantic classes configured.")
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
        if self._lut is None or not self._semantic_classes:
            raise RuntimeError(
                "DDRNetSemanticModel.predict_logits called before warmup() "
                "(or with no semantic classes). Configure semantic classes "
                "with native_indices.goose_12 in config.yaml."
            )

        logits_raw = self.raw_logits_hw(frame_bgr)
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

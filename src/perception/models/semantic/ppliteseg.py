"""PP-LiteSeg-B2 semantic segmentation wrapper (GOOSE-12 head).

Stage 1 status
==============

This is a **skeleton** wrapper. It implements the public surface area
required by :class:`SemanticModel` (constructor signature, ``warmup`` LUT
build, ``class_names`` property) but :meth:`predict_logits` raises
:class:`NotImplementedError` — the actual model build is deferred to
Stage 2.

Target architecture (hypothesis to verify in Stage 2 by inspecting the
checkpoint state-dict)
----------------------------------------------------------------------

* PP-LiteSeg with the **B2** (STDCNet-2) backbone (Peng et al. 2022)
  fine-tuned on GOOSE-12 ("category"-level) with a 12-class head. The
  GOOSE-published "category" checkpoint should expose a final conv with
  ``out_channels=12``; we will confirm in Stage 2 by inspecting the
  ``.pth`` ``state_dict``.

Checkpoint URL
--------------
``https://goose-dataset.de/models/ppliteseg_category_512.pth`` (the
GOOSE benchmark team's category-level release of PP-LiteSeg).

Loader package
--------------
We plan to depend on **mmsegmentation** for PP-LiteSeg: it ships a
maintained PyTorch port of the original PaddleSeg model and accepts a
plain ``.pth`` state-dict via ``init_cfg`` once the layer name mapping
is resolved. The Stage-2 patch will:

1. Add ``mmsegmentation`` (and the matching ``mmcv``/``mmengine``
   versions) as a pip dep.
2. Build a 12-class PP-LiteSeg-B2 config and load the GOOSE checkpoint
   (renaming layers if PaddleSeg<->mmseg name skew turns out to differ).
3. Mirror the SegFormer wrapper's softmax-then-LUT merge in
   :meth:`predict_logits`.

Until then this wrapper exists so the factory + comparison harness can
register the model, build the LUT, and emit clean "Stage 2 needed" rows
when invoked.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import torch

from ...config.schema import ClassDef
from ..backends.base import InferenceBackend
from ._class_catalogues import GOOSE_12_NAMES
from .base import SemanticModel

logger = logging.getLogger(__name__)


class PPLiteSegSemanticModel(SemanticModel):
    """Skeleton wrapper for PP-LiteSeg-B2 with a GOOSE-12 head.

    The constructor signature mirrors :class:`SegFormerSemanticModel`
    so ``build_semantic_model`` can dispatch to either class
    interchangeably. No checkpoint is loaded in Stage 1.
    """

    NATIVE_CATALOGUE: str = "goose_12"
    NUM_NATIVE_CLASSES: int = len(GOOSE_12_NAMES)

    def __init__(
        self,
        weights: str = "https://goose-dataset.de/models/ppliteseg_category_512.pth",
        backend: InferenceBackend | None = None,
        device: str = "cuda",
        fp16: bool = True,
    ) -> None:
        self._weights = weights
        self._backend = backend
        self._device = device
        self._fp16 = bool(fp16) and isinstance(device, str) and device.startswith("cuda")
        self._semantic_classes: list[ClassDef] = []
        self._lut: torch.Tensor | None = None
        logger.info(
            "PPLiteSegSemanticModel constructed (skeleton; weights=%r). "
            "predict_logits() will raise until Stage 2 lands the real loader.",
            weights,
        )

    # ------------------------------------------------------------------ #
    def warmup(self, classes: Sequence[ClassDef]) -> None:
        """Build the GOOSE-12 -> user-class merge LUT.

        Real (non-skeleton) implementation: each semantic user class must
        declare ``native_indices.goose_12`` in ``config.yaml``; classes
        that only declare ``ade20k`` indices cannot be served by this
        wrapper and trigger a clear runtime error.
        """
        sem = [c for c in classes if c.is_semantic]
        self._semantic_classes = sem
        if not sem:
            self._lut = None
            logger.info("PP-LiteSeg: no semantic classes configured.")
            return

        n_native = self.NUM_NATIVE_CLASSES
        lut = torch.zeros(n_native, len(sem), dtype=torch.float32)
        for j, c in enumerate(sem):
            idx = c.native_indices.get(self.NATIVE_CATALOGUE, ())
            if not idx:
                raise ValueError(
                    f"PPLiteSegSemanticModel: class {c.name!r} does not define "
                    f"native_indices.{self.NATIVE_CATALOGUE}; this wrapper "
                    f"cannot serve a class without GOOSE-12 indices."
                )
            for i in idx:
                if not 0 <= int(i) < n_native:
                    raise ValueError(
                        f"PPLiteSegSemanticModel: class {c.name!r} goose_12 "
                        f"index {i} out of range [0, {n_native})"
                    )
                lut[int(i), j] = 1.0

        self._lut = lut
        logger.info(
            "PP-LiteSeg warmed up: %d user classes derived from %d GOOSE-12 channels",
            len(sem), int(lut.sum().item()),
        )

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self._semantic_classes)

    # ------------------------------------------------------------------ #
    def predict_logits(self, frame_bgr: np.ndarray) -> torch.Tensor:
        raise NotImplementedError(
            "PP-LiteSeg model build deferred to Stage 2; see module "
            "docstring for the planned mmsegmentation loader"
        )

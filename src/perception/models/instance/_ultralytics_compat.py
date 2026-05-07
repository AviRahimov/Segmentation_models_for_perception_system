"""Targeted, idempotent compatibility patches for Ultralytics.

Each helper here patches a single upstream symbol, documents the exact
bug being worked around, and is safe to call multiple times. Patches are
applied lazily from :mod:`perception.models.instance.yoloe` on first
model construction so they cost nothing in environments that don't use
the Ultralytics-backed wrapper.

The intent is for every entry to be removable once Ultralytics ships
its own fix - hence the precise version + line-number references.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_APPLIED: bool = False


def apply_patches() -> None:
    """Apply all known Ultralytics compat patches. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return
    _patch_process_mask_fp16()
    _APPLIED = True


# --------------------------------------------------------------------------- #
def _patch_process_mask_fp16() -> None:
    """Make ``ultralytics.utils.ops.process_mask`` fp16-safe.

    Bug (verified in ultralytics==8.4.47, ``utils/ops.py`` line 502)::

        masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)

    ``protos`` is upcast to fp32 but ``masks_in`` stays in the model's
    inference dtype. When ``model.predict(half=True)`` is used with
    YOLOE-26L's segmentation head, ``masks_in`` is fp16 and the matmul
    raises::

        RuntimeError: expected mat1 and mat2 to have the same dtype,
                      but got: c10::Half != float

    Fix: cast ``masks_in`` to fp32 before delegating. ``_orig`` already
    upcasts ``protos`` to fp32 internally via ``protos.float()``; we mirror
    that on ``masks_in`` so the matmul has matching dtypes. This is the
    same numerical pathway upstream intends; we only align the dtype.

    The patch is applied to the module attribute (which is what
    ``ultralytics/models/yolo/segment/predict.py`` resolves at call time
    via ``from ultralytics.utils import ops; ops.process_mask(...)``).
    """
    try:
        from ultralytics.utils import ops  # type: ignore
    except ImportError as e:
        logger.debug("ultralytics.utils.ops unavailable; skipping patch: %s", e)
        return

    target = getattr(ops, "process_mask", None)
    if target is None:
        logger.debug("ultralytics.utils.ops.process_mask missing; skipping patch.")
        return
    if getattr(target, "_perception_patched", False):
        return

    import torch  # local import: this helper is only invoked when ultralytics is present

    _orig = target

    def process_mask(protos, masks_in, bboxes, shape, upsample=False):
        if hasattr(masks_in, "dtype") and masks_in.dtype != torch.float32:
            masks_in = masks_in.float()
        return _orig(protos, masks_in, bboxes, shape, upsample)

    process_mask._perception_patched = True  # type: ignore[attr-defined]
    ops.process_mask = process_mask
    logger.info("Patched ultralytics.utils.ops.process_mask for fp16 compatibility.")

"""Vendored AurigaNet multi-task panoptic architecture.

Source: https://github.com/KiaRational/AurigaNet (Feb 2026)

Modifications vs upstream:
- SegHead: Area_FE_Head now outputs ``num_seg_classes`` channels (was hardcoded 1).
  Lane branch is kept as a module for weight-loading compatibility but its output
  is not concatenated into the returned seg logits.
- ObjDetect: hardcoded ``device="cuda"`` replaced with ``register_buffer`` so
  the model works on any device including CPU.
- All imports made self-contained (no sys.path manipulation).
- ``with_detection=False`` (default) skips the detection head so the model can
  run without needing CUDA or a YOLO anchor setup at import time.

Public API::

    from perception.models.semantic._vendored.auriganet import AurigaNetArch

    model = AurigaNetArch(num_seg_classes=3, with_detection=False)
    seg_logits, seg_embed, det_out = model(x)  # det_out is None when with_detection=False
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ._backbone import BackBone
from ._neck import Neck
from ._seg_head import SegHead
from ._obj_head import Object

_ANCHORS = [
    [[10, 13], [16, 30], [33, 23]],   # P3/8
    [[30, 61], [62, 45], [59, 119]],  # P4/16
    [[116, 90], [156, 198], [373, 326]],  # P5/32
]

_OUT_CHANNELS = [64, 128, 256, 512]
_W, _R, _D = 4, 2, 3


class AurigaNetArch(nn.Module):
    """Full AurigaNet with shared backbone/neck + seg head + optional det head.

    Args:
        num_seg_classes: Number of output segmentation classes (3 for ORFD).
        with_detection:  Instantiate the object detection head. Default False
                         because ORFD training doesn't need it and it avoids
                         any CUDA-grid initialisation at import time on CPU.
    """

    def __init__(self, num_seg_classes: int = 3, with_detection: bool = False):
        super().__init__()
        self.BackBone = BackBone(
            in_channels=3, out_channels_list=_OUT_CHANNELS, w=_W, d=_D, r=_R,
        )
        self.Neck = Neck(_OUT_CHANNELS, w=_W, r=_R, d=_D)
        self.Seg = SegHead(_OUT_CHANNELS, w=_W, r=_R, d=_D, num_seg_classes=num_seg_classes)

        self.with_detection = with_detection
        if with_detection:
            self.Obj = Object(_ANCHORS, out_channels_list=_OUT_CHANNELS, w=_W, r=_R, d=_D)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, object]:
        """Forward pass.

        Returns:
            seg_logits:  ``(B, num_seg_classes, H/4, W/4)`` — raw class logits.
            seg_embed:   ``(B, 8, H/8, W/8)`` — discriminative embedding features.
            det_out:     Detection head output, or ``None`` if ``with_detection=False``.
        """
        Half, Quarter, Octant, One_sixteenth = self.BackBone(x)
        Octant_out, One_sixteenth_out, One_thirty_second_out = self.Neck(Octant, One_sixteenth)
        seg_logits, seg_embed = self.Seg(Half, Quarter, Octant_out, One_sixteenth_out)

        det_out = None
        if self.with_detection:
            det_out = self.Obj(Octant_out, One_sixteenth_out, One_thirty_second_out)

        return seg_logits, seg_embed, det_out

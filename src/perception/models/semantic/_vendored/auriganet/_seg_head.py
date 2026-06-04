"""AurigaNet Segmentation Head.

Vendored from https://github.com/KiaRational/AurigaNet (Feb 2026).
Key modification: Area_FE_Head now takes ``num_seg_classes`` instead of
outputting a hardcoded 1-channel binary map.  SegHead no longer concatenates
the lane branch into the final logits — it returns the multi-class area
output directly.  The LaneSegHead is kept as a module so BDD100K pretrained
weights can be partially loaded (strict=False) without errors.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ._utils import C3, DeformableConv2d

_FEATURE_SIZE = 8  # embedding channels for the discriminative branch


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, n_filters, k_size, padding, stride, bias=False):
        super().__init__(
            nn.Conv2d(in_channels, n_filters, k_size, padding=padding, stride=stride, bias=bias),
            nn.BatchNorm2d(n_filters, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.SiLU(inplace=True),
        )


class DownsampleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, downsample_ratio=2):
        super().__init__()
        steps = int(torch.log2(torch.tensor(float(downsample_ratio))).item())
        for i in range(steps):
            ch_in = in_channels if i == 0 else out_channels
            self.add_module(f"conv_{i}", ConvBNReLU(ch_in, out_channels, 3, 1, 2))
        self.add_module("conv_1x1", ConvBNReLU(out_channels, out_channels, 1, 0, 1))


class UpsampleConvTranspose(nn.Sequential):
    def __init__(self, in_channels, out_channels, upsample_ratio=2):
        super().__init__()
        steps = int(torch.log2(torch.tensor(float(upsample_ratio))).item())
        for i in range(steps):
            ch_in = in_channels if i == 0 else out_channels
            self.add_module(f"deconv_{i}", nn.ConvTranspose2d(ch_in, out_channels, 3, 2, 1, 1))
        self.add_module("conv_1x1", ConvBNReLU(out_channels, out_channels, 1, 0, 1))


class MixedUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_ratio=2):
        super().__init__()
        self.upsampleconv = UpsampleConvTranspose(in_channels, out_channels, upsample_ratio)

    def forward(self, x):
        return self.upsampleconv(x)


class Output(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            ConvBNReLU(in_channels, in_channels // 4, 3, 1, 1),
            ConvBNReLU(in_channels // 4, in_channels // 8, 3, 1, 1),
            nn.Conv2d(in_channels // 8, out_channels, 1, padding=0, stride=1, bias=False),
        )


class LaneSegHead(nn.Module):
    """Binary lane-detection decoder (kept for weight-loading compatibility)."""

    def __init__(self, out_channels_list, w, r, d):
        super().__init__()
        mid = (out_channels_list[2] + out_channels_list[1]) // w
        self.MixUpsample_1 = MixedUpsample(out_channels_list[2] // w, out_channels_list[2] // w)
        self.c3_1 = C3(mid, mid, depth=3)
        self.MixUpsample_2 = MixedUpsample(mid, mid)
        self.c3_2 = C3(mid, out_channels_list[2] // w, depth=3)
        self.out = Output(out_channels_list[2] // w, 1)

    def forward(self, Quarter, Octant):
        x = self.c3_1(torch.cat((self.MixUpsample_1(Octant), Quarter), 1))
        x = self.c3_2(x)
        return self.out(x)


class Area_FE_Head(nn.Module):
    """Drivable-area + feature-embedding decoder.

    Modified to output ``num_seg_classes`` channels instead of 1.
    """

    def __init__(self, out_channels_list, w, r, d, out_fe, num_seg_classes=1):
        super().__init__()
        mid = (out_channels_list[2] + out_channels_list[1]) // w
        self.MixUpsample_1 = MixedUpsample(out_channels_list[2] // w, out_channels_list[2] // w)
        self.c3_1_d = C3(mid, mid, depth=3)
        self.MixUpsample_2 = MixedUpsample(mid, mid)   # defined for weight compat, unused in fwd
        self.c3_2_d = C3(mid, mid, depth=3)
        self.MixUpsample_3 = MixedUpsample(mid, mid)   # defined for weight compat, unused in fwd
        self.def_conv_1_f = DeformableConv2d(mid, mid)
        self.DownsampleConv_1 = DownsampleConv(mid, mid, downsample_ratio=2)
        self.def_conv_3_f = DeformableConv2d(mid, mid)
        # KEY change: output num_seg_classes channels (original was 1)
        self.out_d = Output(mid, num_seg_classes)
        self.def_conv_4_1 = DeformableConv2d(mid, mid)
        self.out_f = DeformableConv2d(mid, out_fe)

    def forward(self, Quarter, Octant):
        out_d = self.c3_1_d(torch.cat((self.MixUpsample_1(Octant), Quarter), 1))
        out_d = self.c3_2_d(out_d)
        out_f = self.def_conv_1_f(self.def_conv_3_f(self.DownsampleConv_1(out_d)))
        out_d = self.out_d(out_d)
        out_f = self.out_f(self.def_conv_4_1(out_f))
        return out_d, out_f


class SegHead(nn.Module):
    def __init__(self, out_channels_list, w, r, d, num_seg_classes=3):
        super().__init__()
        self.lane = LaneSegHead(out_channels_list, w, r, d)
        self.area_fe = Area_FE_Head(
            out_channels_list, w, r, d,
            out_fe=_FEATURE_SIZE, num_seg_classes=num_seg_classes,
        )

    def forward(self, Half, Quarter, Octant, One_sixteenth):
        # Half and One_sixteenth are received for API compatibility but unused.
        seg_logits, seg_embed = self.area_fe(Quarter, Octant)
        return seg_logits, seg_embed

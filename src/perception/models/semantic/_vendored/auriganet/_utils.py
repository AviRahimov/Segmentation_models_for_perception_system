"""Shared building blocks used across the AurigaNet model.

Vendored from https://github.com/KiaRational/AurigaNet (Feb 2026).
No functional changes; import paths updated to be self-contained.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


class ConvBNSiLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=padding, stride=stride, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class ObjBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, width_multiple=1):
        super().__init__()
        c_ = int(width_multiple * in_channels)
        self.c1 = ConvBNSiLU(in_channels, c_, 1, 1, 0)
        self.c2 = ConvBNSiLU(c_, out_channels, 3, 1, 1)

    def forward(self, x):
        return self.c2(self.c1(x)) + x


class C3(nn.Module):
    def __init__(self, in_channels, out_channels, width_multiple=1, depth=1, backbone=True):
        super().__init__()
        c_ = int(width_multiple * in_channels)
        self.c1 = ConvBNSiLU(in_channels, c_, 1, 1, 0)
        self.c_skipped = ConvBNSiLU(in_channels, c_, 1, 1, 0)
        if backbone:
            self.seq = nn.Sequential(
                *[ObjBottleneck(c_, c_, width_multiple=1) for _ in range(depth)]
            )
        else:
            self.seq = nn.Sequential(
                *[nn.Sequential(
                    ConvBNSiLU(c_, c_, 1, 1, 0),
                    ConvBNSiLU(c_, c_, 3, 1, 1),
                ) for _ in range(depth)]
            )
        self.c_out = ConvBNSiLU(c_ * 2, out_channels, 1, 1, 0)

    def forward(self, x):
        x = torch.cat([self.seq(self.c1(x)), self.c_skipped(x)], dim=1)
        return self.c_out(x)


class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.padding = padding
        self.offset_conv = nn.Conv2d(
            in_channels, 2 * kernel_size * kernel_size,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=True,
        )
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)
        self.modulator_conv = nn.Conv2d(
            in_channels, 1 * kernel_size * kernel_size,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=True,
        )
        nn.init.constant_(self.modulator_conv.weight, 0.0)
        nn.init.constant_(self.modulator_conv.bias, 0.0)
        self.regular_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=bias,
        )

    def forward(self, x):
        h, w = x.shape[2:]
        max_offset = max(h, w) / 4.0
        offset = self.offset_conv(x).clamp(-max_offset, max_offset)
        modulator = 2.0 * torch.sigmoid(self.modulator_conv(x))
        return torchvision.ops.deform_conv2d(
            input=x, offset=offset,
            weight=self.regular_conv.weight, bias=self.regular_conv.bias,
            padding=self.padding, mask=modulator,
        )

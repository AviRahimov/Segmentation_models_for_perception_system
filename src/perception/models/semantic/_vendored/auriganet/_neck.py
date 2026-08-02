"""AurigaNet Feature Pyramid Neck.

Vendored from https://github.com/KiaRational/AurigaNet (Feb 2026).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ._utils import C3, ConvBNSiLU


class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        c_ = in_channels // 2
        self.c1 = ConvBNSiLU(in_channels, c_, 1, 1, 0)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.c_out = ConvBNSiLU(c_ * 4, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.c1(x)
        p1, p2, p3 = self.pool(x), self.pool(self.pool(x)), self.pool(self.pool(self.pool(x)))
        return self.c_out(torch.cat([x, p1, p2, p3], dim=1))


class Neck(nn.Module):
    def __init__(self, in_channels, w=4, r=2, d=3):
        super().__init__()
        self.c1 = nn.Sequential(
            ConvBNSiLU(in_channels[3] // w, in_channels[3] * r // w, 3, 2, 1),
            C3(in_channels[3] * r // w, in_channels[3] * r // w, width_multiple=1, depth=1),
            SPPF(in_channels[3] * r // w, in_channels[3] * r // w),
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.c3_1 = C3(in_channels[3] * (r + 1) // w, in_channels[3] // w, depth=d // 3)
        self.cbl_1 = ConvBNSiLU(in_channels[3] // w, in_channels[3] // w, 1, 1, 1)
        self.c3_2 = C3((in_channels[3] + in_channels[2]) // w, in_channels[2] // w, depth=d // 3)

    def forward(self, Octant, One_sixteenth):
        One_thirty_second_out = self.c1(One_sixteenth)
        One_sixteenth_up = self.up(One_thirty_second_out)
        One_sixteenth_out = self.c3_1(torch.cat((One_sixteenth, One_sixteenth_up), dim=1))
        Octant_up = self.up(One_sixteenth_out)
        Octant_out = self.c3_2(torch.cat((Octant, Octant_up), dim=1))
        return Octant_out, One_sixteenth_out, One_thirty_second_out

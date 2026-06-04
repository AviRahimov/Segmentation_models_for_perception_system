"""AurigaNet Backbone encoder.

Vendored from https://github.com/KiaRational/AurigaNet (Feb 2026).
"""
from __future__ import annotations

import torch.nn as nn

from ._utils import C3, ConvBNSiLU


class BackBone(nn.Module):
    def __init__(self, in_channels, out_channels_list, w, d, r):
        super().__init__()
        self.conv1 = ConvBNSiLU(in_channels, out_channels_list[0] // w, 3, 2, 1)
        self.conv2 = ConvBNSiLU(out_channels_list[0] // w, out_channels_list[1] // w, 3, 2, 1)
        self.c3_1 = C3(out_channels_list[1] // w, out_channels_list[1] // w, depth=3 // d)
        self.conv3 = ConvBNSiLU(out_channels_list[1] // w, out_channels_list[2] // w, 3, 2, 1)
        self.c3_2 = C3(out_channels_list[2] // w, out_channels_list[2] // w, depth=6 // d)
        self.conv4 = ConvBNSiLU(out_channels_list[2] // w, out_channels_list[3] // w, 3, 2, 1)
        self.c3_3 = C3(out_channels_list[3] // w, out_channels_list[3] // w, depth=6 // d)

    def forward(self, x):
        x = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.c3_1(out2)
        x = self.conv3(out3)
        out5 = self.c3_2(x)
        x = self.conv4(out5)
        out7 = self.c3_3(x)
        return out2, out3, out5, out7

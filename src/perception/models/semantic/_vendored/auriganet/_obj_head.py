"""AurigaNet Object Detection Head.

Vendored from https://github.com/KiaRational/AurigaNet (Feb 2026).
Fix: removed hardcoded ``device="cuda"`` — anchor buffers are now registered
via ``register_buffer`` so they move with the model.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ._utils import C3, ConvBNSiLU

_BDD_CLASSES = ["car"]


class ObjHead(nn.Module):
    def __init__(self, out_channels_list, w, r, d):
        super().__init__()
        self.cbl_1 = ConvBNSiLU(out_channels_list[2] // w, out_channels_list[2] // w, 3, 2, 1)
        self.c3_1 = C3((out_channels_list[2] + out_channels_list[3]) // w,
                       out_channels_list[3] // w, depth=3 // d)
        self.cbl_2 = ConvBNSiLU(out_channels_list[3] // w, out_channels_list[3] // w, 3, 2, 1)
        self.c3_2 = C3(out_channels_list[3] * (1 + r) // w,
                       out_channels_list[3] * r // w, depth=3 // d)

    def forward(self, Octant_in, One_sixteenth_in, One_thirty_second_in):
        One_sixteenth_Head = self.c3_1(
            torch.cat((One_sixteenth_in, self.cbl_1(Octant_in)), dim=1)
        )
        One_thirty_second_Head = self.c3_2(
            torch.cat((self.cbl_2(One_sixteenth_Head), One_thirty_second_in), dim=1)
        )
        return Octant_in, One_sixteenth_Head, One_thirty_second_Head


class ObjDetect(nn.Module):
    def __init__(self, nc, anchors, ch):
        super().__init__()
        self.nc = nc
        self.nl = len(anchors)
        self.na = len(anchors[0])
        self.stride = [8, 16, 32]
        self.dynamic = False

        # Device-agnostic: use register_buffer so .to(device) propagates.
        anchors_t = torch.tensor(anchors, dtype=torch.float32).view(self.nl, -1, 2)
        stride_t = torch.tensor(self.stride, dtype=torch.float32)
        anchors_t = anchors_t / stride_t.view(-1, 1, 1)
        self.register_buffer("anchors", anchors_t)

        self.grid = [torch.empty(0) for _ in range(self.nl)]
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]

        self.out_convs = nn.ModuleList(
            nn.Conv2d(in_channels=c, out_channels=(5 + self.nc) * self.na, kernel_size=1)
            for c in ch
        )

    def forward(self, x):
        batch_boxes = []
        for i, conv in enumerate(self.out_convs):
            x[i] = conv(x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.nc + 5, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

            pred = x[i].sigmoid()
            obj = pred[..., 4:5]
            xy = (2 * pred[..., :2] + self.grid[i]) * self.stride[i]
            wh = (2 * pred[..., 2:4]) ** 2 * self.anchor_grid[i]
            best_class = torch.argmax(pred[..., 5:], dim=-1).unsqueeze(-1)
            batch_boxes.append(
                torch.cat((best_class, obj, xy, wh), dim=-1).view(bs, -1, 5 + self.nc)
            )
        return [x, torch.cat(batch_boxes, dim=1)]

    def _make_grid(self, nx, ny, i):
        d, t = self.anchors[i].device, self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2
        yv, xv = torch.meshgrid(
            torch.arange(ny, device=d, dtype=t),
            torch.arange(nx, device=d, dtype=t),
            indexing="ij",
        )
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view(1, self.na, 1, 1, 2).expand(shape)
        return grid, anchor_grid


class Object(nn.Module):
    def __init__(self, anchors, out_channels_list, w, r, d):
        super().__init__()
        self.ObjHead = ObjHead(out_channels_list, w=w, r=r, d=d)
        self.ObjDetect = ObjDetect(
            nc=len(_BDD_CLASSES),
            anchors=anchors,
            ch=(out_channels_list[2] // w,
                out_channels_list[3] // w,
                out_channels_list[3] * r // w),
        )

    def forward(self, Octant_in, One_sixteenth_in, One_thirty_second_in):
        Octant_out, One_sixteenth_out, One_thirty_second_out = self.ObjHead(
            Octant_in, One_sixteenth_in, One_thirty_second_in,
        )
        return self.ObjDetect([Octant_out, One_sixteenth_out, One_thirty_second_out])

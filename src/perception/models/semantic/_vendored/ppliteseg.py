"""PP-LiteSeg architecture (vendored).

Source: https://github.com/midasklr/PPLiteSeg.pytorch (pp_liteseg.py)
        @888892f047a6a02b4cd88ba2e4924df1693464e5
License: Apache-2.0 (PaddlePaddle Authors, 2022). The midasklr repo is a
         direct PyTorch port of PaddleSeg's PP-LiteSeg
         (https://github.com/PaddlePaddle/PaddleSeg/tree/develop/configs/pp_liteseg);
         the original source-file header is preserved below.

Local edits relative to upstream
================================

* Fixed two pure-typo bugs that would prevent the module from importing
  cleanly under modern PyTorch:

  - ``AddBottleneck`` referenced ``nn.BatchNorm2D`` (Paddle spelling)
    and ``bias_attr=False`` (Paddle kwarg). Corrected to
    ``nn.BatchNorm2d`` / ``bias=False``.
  - ``avg_max_reduce_channel_helper`` returned ``torch.at(...)`` (typo)
    in the ``use_concat=True`` branch; corrected to ``torch.cat(...)``.

  These code paths are dead in the default PP-LiteSeg-B configuration
  (which uses ``CatBottleneck`` and routes through
  ``avg_max_reduce_channel`` with ``use_concat=False``), but the typos
  would break ``import`` of the module under any tool that
  type-checks defaults — so they are fixed here for safety.

* Removed commented-out ``init_weight`` and ``get_seg_model`` blocks at
  EOF. Their original purpose (loading pretrained weights) is handled
  by our wrapper class via ``load_state_dict``.

* Removed stray ``print()`` calls inside ``PPLiteSeg.__init__`` /
  ``SegHead.__init__`` (debug breadcrumbs from the upstream port) so
  imports are silent.

* Added a small :func:`ppliteseg_b` builder that mirrors PP-LiteSeg-B
  (STDC2 backbone) with our 12-class head.

Reference
---------

Peng et al., "PP-LiteSeg: A Superior Real-Time Semantic Segmentation
Model" (2022), https://arxiv.org/abs/2204.02681
"""
# Original upstream header (preserved verbatim from PaddlePaddle):
#
# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: skip-file
# ruff: noqa
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, bias=False, **kwargs):
        super().__init__()
        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride,
            padding=kernel_size // 2 if padding else 0,
            bias=bias, **kwargs,
        )
        self._batch_norm = nn.BatchNorm2d(out_channels, momentum=0.1)

    def forward(self, x):
        return self._batch_norm(self._conv(x))


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, bias=False, **kwargs):
        super().__init__()
        self._conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride,
            padding=kernel_size // 2 if padding else 0,
            bias=bias, **kwargs,
        )
        self._batch_norm = nn.BatchNorm2d(out_channels, momentum=0.1)
        self._relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self._relu(self._batch_norm(self._conv(x)))


class ConvBNRelu(nn.Module):
    """Variant with attribute names ``conv``/``bn``/``relu`` (Paddle naming).

    The STDC backbone uses this spelling internally. Functionally
    identical to :class:`ConvBNReLU`; we keep both because the upstream
    state-dict references both attribute layouts.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, bias=False, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride,
            padding=kernel_size // 2 if padding else 0,
            bias=bias, **kwargs,
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# --------------------------------------------------------------------------- #
# Channel-reduction helpers used by UAFM_SpAtten                              #
# --------------------------------------------------------------------------- #


def avg_max_reduce_channel_helper(x, use_concat=True):
    """Reduce HW by per-channel mean and max.

    With ``use_concat=True`` returns a single ``(N, 2, H, W)`` tensor;
    with ``use_concat=False`` returns ``[mean, max]`` so the caller can
    interleave reductions across multiple inputs.
    """
    assert not isinstance(x, (list, tuple))
    mean_value = torch.mean(x, dim=1, keepdim=True)
    max_value = torch.max(x, dim=1, keepdim=True)[0]
    if use_concat:
        return torch.cat([mean_value, max_value], dim=1)
    return [mean_value, max_value]


def avg_max_reduce_channel(x):
    if not isinstance(x, (list, tuple)):
        return avg_max_reduce_channel_helper(x)
    if len(x) == 1:
        return avg_max_reduce_channel_helper(x[0])
    res = []
    for xi in x:
        res.extend(avg_max_reduce_channel_helper(xi, False))
    return torch.cat(res, dim=1)


# --------------------------------------------------------------------------- #
# Unified Attention Fusion Module (UAFM)                                      #
# --------------------------------------------------------------------------- #


class UAFM(nn.Module):
    """Base UAFM module (no attention)."""

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='nearest'):
        super().__init__()
        self.conv_x = ConvBNReLU(x_ch, y_ch, kernel_size=ksize,
                                 padding=ksize // 2, bias=False)
        self.conv_out = ConvBNReLU(y_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.resize_mode = resize_mode

    def check(self, x, y):
        assert x.ndim == 4 and y.ndim == 4
        x_h, x_w = x.shape[2:]
        y_h, y_w = y.shape[2:]
        assert x_h >= y_h and x_w >= y_w

    def prepare(self, x, y):
        x = self.prepare_x(x, y)
        y = self.prepare_y(x, y)
        return x, y

    def prepare_x(self, x, y):
        return self.conv_x(x)

    def prepare_y(self, x, y):
        return F.interpolate(y, x.shape[2:], mode=self.resize_mode)

    def fuse(self, x, y):
        return self.conv_out(x + y)

    def forward(self, x, y):
        self.check(x, y)
        x, y = self.prepare(x, y)
        return self.fuse(x, y)


class UAFM_SpAtten(UAFM):
    """UAFM with spatial attention based on per-channel mean/max."""

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='nearest'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)
        self.conv_xy_atten = nn.Sequential(
            ConvBNReLU(4, 2, kernel_size=3, padding=1, bias=False),
            ConvBN(2, 1, kernel_size=3, padding=1, bias=False),
        )

    def fuse(self, x, y):
        atten = avg_max_reduce_channel([x, y])
        atten = torch.sigmoid(self.conv_xy_atten(atten))
        out = x * atten + y * (1 - atten)
        return self.conv_out(out)


# --------------------------------------------------------------------------- #
# STDC backbone                                                                #
# --------------------------------------------------------------------------- #


class CatBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes // 2, out_planes // 2, kernel_size=3,
                          stride=2, padding=1, groups=out_planes // 2, bias=False),
                nn.BatchNorm2d(out_planes // 2, momentum=0.1),
            )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvBNRelu(in_planes, out_planes // 2, kernel_size=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvBNRelu(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvBNRelu(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx + 1))),
                )
            else:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx))),
                )

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                out = conv(self.avd_layer(out1)) if self.stride == 2 else conv(out1)
            else:
                out = conv(out)
            out_list.append(out)
        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        return torch.cat(out_list, dim=1)


class AddBottleneck(nn.Module):
    """Residual variant of the STDC bottleneck.

    Unused by the default ``CatBottleneck``-based STDC2 builder, but kept
    for completeness and so the upstream class hierarchy is preserved.
    """

    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super().__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes // 2, out_planes // 2, kernel_size=3,
                          stride=2, padding=1, groups=out_planes // 2, bias=False),
                nn.BatchNorm2d(out_planes // 2, momentum=0.1),
            )
            self.skip = nn.Sequential(
                nn.Conv2d(in_planes, in_planes, kernel_size=3,
                          stride=2, padding=1, groups=in_planes, bias=False),
                nn.BatchNorm2d(in_planes, momentum=0.1),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes, momentum=0.1),
            )
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvBNRelu(in_planes, out_planes // 2, kernel_size=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvBNRelu(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvBNRelu(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx + 1))),
                )
            else:
                self.conv_list.append(
                    ConvBNRelu(out_planes // int(math.pow(2, idx)),
                               out_planes // int(math.pow(2, idx))),
                )

    def forward(self, x):
        out_list = []
        out = x
        for idx, conv in enumerate(self.conv_list):
            if idx == 0 and self.stride == 2:
                out = self.avd_layer(conv(out))
            else:
                out = conv(out)
            out_list.append(out)
        if self.stride == 2:
            x = self.skip(x)
        return torch.cat(out_list, dim=1) + x


class STDCNet(nn.Module):
    """STDC backbone (Fan et al., "Rethinking BiSeNet").

    Args:
        base: base channel count.
        layers: STDC block counts per stage.
        block_num: number of feature blocks within each STDC bottleneck.
        type: ``"cat"`` (default) or ``"add"``.
        use_conv_last: emit ``conv_last`` on top of x32. Default False.
    """

    def __init__(self, base=64, layers=(4, 5, 3), block_num=4, type="cat",
                 use_conv_last=False):
        super().__init__()
        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck
        else:
            raise ValueError(f"unknown STDC block type: {type}")
        self.use_conv_last = use_conv_last
        self.feat_channels = [base // 2, base, base * 4, base * 8, base * 16]
        self.features = self._make_layers(base, list(layers), block_num, block)
        self.conv_last = ConvBNRelu(base * 16, max(1024, base * 16), 1, 1)

        if list(layers) == [4, 5, 3]:  # STDC1446
            self.x2 = nn.Sequential(self.features[:1])
            self.x4 = nn.Sequential(self.features[1:2])
            self.x8 = nn.Sequential(self.features[2:6])
            self.x16 = nn.Sequential(self.features[6:11])
            self.x32 = nn.Sequential(self.features[11:])
        elif list(layers) == [2, 2, 2]:  # STDC813
            self.x2 = nn.Sequential(self.features[:1])
            self.x4 = nn.Sequential(self.features[1:2])
            self.x8 = nn.Sequential(self.features[2:4])
            self.x16 = nn.Sequential(self.features[4:6])
            self.x32 = nn.Sequential(self.features[6:])
        else:
            raise NotImplementedError(
                f"STDCNet with layers={layers} is not implemented",
            )

    def forward(self, x):
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
        feat8 = self.x8(feat4)
        feat16 = self.x16(feat8)
        feat32 = self.x32(feat16)
        if self.use_conv_last:
            feat32 = self.conv_last(feat32)
        return feat2, feat4, feat8, feat16, feat32

    def _make_layers(self, base, layers, block_num, block):
        features = [
            ConvBNRelu(3, base // 2, 3, 2),
            ConvBNRelu(base // 2, base, 3, 2),
        ]
        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base * 4, block_num, 2))
                elif j == 0:
                    features.append(
                        block(base * int(math.pow(2, i + 1)),
                              base * int(math.pow(2, i + 2)), block_num, 2),
                    )
                else:
                    features.append(
                        block(base * int(math.pow(2, i + 2)),
                              base * int(math.pow(2, i + 2)), block_num, 1),
                    )
        return nn.Sequential(*features)


def STDC2(**kwargs) -> STDCNet:
    """STDC2 backbone (base=64, layers=[4,5,3])."""
    return STDCNet(base=64, layers=[4, 5, 3], **kwargs)


def STDC1(**kwargs) -> STDCNet:
    """STDC1 backbone (base=64, layers=[2,2,2])."""
    return STDCNet(base=64, layers=[2, 2, 2], **kwargs)


# --------------------------------------------------------------------------- #
# PP-LiteSeg head                                                              #
# --------------------------------------------------------------------------- #


class PPContextModule(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, bin_sizes,
                 align_corners=None):
        super().__init__()
        self.stages = nn.ModuleList([
            self._make_stage(in_channels, inter_channels, size) for size in bin_sizes
        ])
        self.conv_out = ConvBNReLU(
            in_channels=inter_channels, out_channels=out_channels,
            kernel_size=3, padding=1, bias=True,
        )
        self.align_corners = align_corners

    def _make_stage(self, in_channels, out_channels, size):
        prior = nn.AdaptiveAvgPool2d(output_size=size)
        conv = ConvBNReLU(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=1, bias=True,
        )
        return nn.Sequential(prior, conv)

    def forward(self, input):
        out = None
        input_shape = input.shape[2:]
        for stage in self.stages:
            x = stage(input)
            x = F.interpolate(x, input_shape, mode='nearest')
            out = x if out is None else out + x
        return self.conv_out(out)


class SegHead(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes):
        super().__init__()
        self.conv = ConvBNReLU(
            in_chan, mid_chan, kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv_out(self.conv(x))


class PPLiteSegHead(nn.Module):
    def __init__(self, backbone_out_chs, arm_out_chs, cm_bin_sizes,
                 cm_out_ch, arm_type, resize_mode):
        super().__init__()
        self.cm = PPContextModule(backbone_out_chs[-1], cm_out_ch, cm_out_ch, cm_bin_sizes)
        if arm_type == "UAFM_SpAtten":
            arm_class = UAFM_SpAtten
        elif arm_type == "UAFM":
            arm_class = UAFM
        else:
            raise ValueError(f"unknown ARM type: {arm_type}")

        self.arm_list = nn.ModuleList()
        for i in range(len(backbone_out_chs)):
            low_chs = backbone_out_chs[i]
            high_ch = cm_out_ch if i == len(backbone_out_chs) - 1 else arm_out_chs[i + 1]
            out_ch = arm_out_chs[i]
            arm = arm_class(low_chs, high_ch, out_ch, ksize=3, resize_mode=resize_mode)
            self.arm_list.append(arm)

    def forward(self, in_feat_list):
        high_feat = self.cm(in_feat_list[-1])
        out_feat_list = []
        for i in reversed(range(len(in_feat_list))):
            low_feat = in_feat_list[i]
            arm = self.arm_list[i]
            high_feat = arm(low_feat, high_feat)
            out_feat_list.insert(0, high_feat)
        return out_feat_list


class PPLiteSeg(nn.Module):
    """PP-LiteSeg.

    The default kwargs match PP-LiteSeg-B (STDC2 backbone, ARM=SpAtten,
    cm_out_ch=128, arm_out_chs=[64, 96, 128], seg_head_inter_chs=[64, 64, 64]).
    The ``backbone`` argument is provided as a class attribute *instance*
    so it can be replaced without subclassing.
    """

    def __init__(self, num_classes: int = 19, backbone: nn.Module | None = None,
                 backbone_indices=(2, 3, 4), arm_type='UAFM_SpAtten',
                 cm_bin_sizes=(1, 2, 4), cm_out_ch=128,
                 arm_out_chs=(64, 96, 128),
                 seg_head_inter_chs=(64, 64, 64),
                 resize_mode='nearest'):
        super().__init__()
        if backbone is None:
            backbone = STDC2()
        backbone_indices = list(backbone_indices)
        cm_bin_sizes = list(cm_bin_sizes)
        arm_out_chs = list(arm_out_chs)
        seg_head_inter_chs = list(seg_head_inter_chs)

        assert hasattr(backbone, 'feat_channels'), \
            "The backbone should have a `feat_channels` attribute."
        assert len(backbone.feat_channels) >= len(backbone_indices), \
            f"backbone_indices ({len(backbone_indices)}) > feat_channels " \
            f"({len(backbone.feat_channels)})"
        assert len(backbone.feat_channels) > max(backbone_indices), \
            f"max backbone_indices ({max(backbone_indices)}) >= " \
            f"len(feat_channels) ({len(backbone.feat_channels)})"

        self.backbone = backbone
        assert len(backbone_indices) > 1, "backbone_indices must have length > 1"
        self.backbone_indices = backbone_indices
        backbone_out_chs = [backbone.feat_channels[i] for i in backbone_indices]

        if len(arm_out_chs) == 1:
            arm_out_chs = arm_out_chs * len(backbone_indices)
        assert len(arm_out_chs) == len(backbone_indices)

        self.ppseg_head = PPLiteSegHead(
            backbone_out_chs, arm_out_chs, cm_bin_sizes, cm_out_ch, arm_type, resize_mode,
        )

        if len(seg_head_inter_chs) == 1:
            seg_head_inter_chs = seg_head_inter_chs * len(backbone_indices)
        assert len(seg_head_inter_chs) == len(backbone_indices)

        self.seg_heads = nn.ModuleList([
            SegHead(in_ch, mid_ch, num_classes)
            for in_ch, mid_ch in zip(arm_out_chs, seg_head_inter_chs)
        ])

    def forward(self, x):
        x_hw = x.shape[2:]
        feats_backbone = self.backbone(x)  # [x2, x4, x8, x16, x32]
        feats_selected = [feats_backbone[i] for i in self.backbone_indices]
        feats_head = self.ppseg_head(feats_selected)
        if self.training:
            logit_list = []
            for f, seg_head in zip(feats_head, self.seg_heads):
                logit_list.append(seg_head(f))
            return [F.interpolate(it, x_hw, mode='bilinear', align_corners=None)
                    for it in logit_list]
        x = self.seg_heads[0](feats_head[0])
        x = F.interpolate(x, x_hw, mode='bilinear', align_corners=None)
        return [x]


def ppliteseg_b(num_classes: int = 19) -> PPLiteSeg:
    """Build a PP-LiteSeg-B (STDC2 backbone) with the given output classes."""
    return PPLiteSeg(num_classes=num_classes, backbone=STDC2())

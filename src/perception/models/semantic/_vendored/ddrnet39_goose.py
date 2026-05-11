"""DDRNet-39 architecture (vendored, super-gradients flavour for GOOSE checkpoints).

Source: https://github.com/Deci-AI/super-gradients
        @63de22c404d5740f34f7706c302b37fce3c8fe5d
        - src/super_gradients/training/models/segmentation_models/ddrnet.py
        - src/super_gradients/training/models/classification_models/resnet.py (BasicResNetBlock, Bottleneck)
License: Apache-2.0 (Deci-AI / Intel, 2024)

Why this file exists alongside ``ddrnet23_slim.py``
====================================================

The official GOOSE benchmark
(https://github.com/FraunhoferIOSB/goose_dataset) trains its DDRNet
checkpoints with **Deci-AI's super_gradients** framework
(``Models.DDRNET_39`` per ``image_processing/semantic_train.py``). The
state-dict naming differs structurally from the chenjun2hao port we
vendored in ``ddrnet23_slim.py`` (``_backbone.stem``,
``spp.branches.{0..4}``, ``layer3.{0,1}`` doubled-stage,
``layer3_skip``/``layer4_skip``/``layer5_skip``, ``compression3``/``down3``
as ``nn.ModuleList``s, etc.). To strict-load the published
``ddrnet_category_512.pth`` and ``ddrnet_category_1024.pth`` weights we
need the *super_gradients* layout.

We do not pip-install ``super_gradients`` because:

* It pulls ~50+ training/CV utility deps including DALI, deepl-api,
  hydra, torchmetrics, pretrained-checkpoints API, etc., adding
  hundreds of MB and a Torch version ceiling.
* The training framework's registries / param-group machinery / ABC
  interfaces are irrelevant to inference.

Local edits relative to upstream
================================

* Removed all super_gradients-specific imports
  (``register_model``, ``HpmStruct``, ``SegmentationModule``,
  ``ExportableSegmentationModel``, ``DropPath``,
  ``BaseClassifier``, ``SupportsReplaceInputChannels``).
  ``DDRNet`` here is a plain :class:`torch.nn.Module`.
* Replaced ``DropPath(drop_prob=...)`` with :class:`torch.nn.Identity`
  in :class:`BasicResNetBlock` / :class:`Bottleneck` — at inference
  ``drop_prob=0`` and ``DropPath`` has no parameters so this does not
  change the state-dict layout.
* Removed ``classification_mode`` / ``replace_head`` /
  ``initialize_param_groups`` / ``replace_input_channels`` / training
  helpers — the wrapper only needs ``forward()``.
* Removed the ``backbone`` ``@property`` that aliased the model's
  internal ``_backbone`` for pretrained-weights loading; we strict-load
  the head + backbone in one shot.
* The forward path is **functionally identical** to the upstream
  ``DDRNet.forward``: same conv ops, same residual additions, same
  upsample/SPP composition. Strict ``load_state_dict`` of the GOOSE
  checkpoints succeeds (verified at vendoring time — see
  ``tests/test_ddrnet_wrapper.py``).

Reference
---------

Hong et al., "Deep Dual-resolution Networks for Real-time and Accurate
Semantic Segmentation of Road Scenes" (2021),
https://arxiv.org/abs/2101.06085
"""
# pylint: skip-file
# ruff: noqa
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Building blocks                                                              #
# --------------------------------------------------------------------------- #


def ConvBN(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    bias: bool = True,
    stride: int = 1,
    padding: int = 0,
    add_relu: bool = False,
) -> nn.Sequential:
    """Sequential of (Conv2d, BatchNorm2d, [ReLU])."""
    seq = [
        nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            bias=bias, stride=stride, padding=padding,
        ),
        nn.BatchNorm2d(out_channels),
    ]
    if add_relu:
        seq.append(nn.ReLU(inplace=True))
    return nn.Sequential(*seq)


def _make_layer(
    block: type,
    in_planes: int,
    planes: int,
    num_blocks: int,
    stride: int = 1,
    expansion: int = 1,
) -> nn.Sequential:
    """Build an ``nn.Sequential`` of ``num_blocks`` block instances.

    Mirrors super_gradients' ``_make_layer``: the *first* block carries
    the stride and ``final_relu = num_blocks > 1``; intermediate blocks
    keep ``stride=1, final_relu=True``; the *last* block has
    ``final_relu=False`` (no ReLU after the residual sum).
    """
    layers = []
    layers.append(block(in_planes, planes, stride, final_relu=num_blocks > 1, expansion=expansion))
    in_planes = planes * expansion
    if num_blocks > 1:
        for i in range(1, num_blocks):
            if i == num_blocks - 1:
                layers.append(block(in_planes, planes, stride=1, final_relu=False, expansion=expansion))
            else:
                layers.append(block(in_planes, planes, stride=1, final_relu=True, expansion=expansion))
    return nn.Sequential(*layers)


class BasicResNetBlock(nn.Module):
    """Pre-final-relu residual block (one stride'able 3x3 + one 3x3).

    State-dict layout (parents excluded):
        ``conv1.weight``, ``bn1.{weight, bias, running_mean, running_var, num_batches_tracked}``,
        ``conv2.weight``, ``bn2.{...}``,
        ``shortcut.{0.weight, 1.{weight, bias, ...}}`` if stride != 1 or channel-mismatch,
        else nothing for ``shortcut`` (empty ``nn.Sequential``).
    """

    def __init__(self, in_planes, planes, stride=1, expansion=1, final_relu=True):
        super().__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.final_relu = final_relu
        self.drop_path = nn.Identity()  # super_gradients uses DropPath; identity at inference
        self.shortcut: nn.Module = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.drop_path(out)
        out = out + self.shortcut(x)
        if self.final_relu:
            out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """ResNet bottleneck block (1x1 → 3x3 → 1x1).

    Default ``expansion=4`` upstream; in DDRNet's ``layer5`` /
    ``layer5_skip`` the expansion is overridden to 2 (see
    ``layer5_bottleneck_expansion`` in :class:`DDRNet`).
    """

    def __init__(self, in_planes, planes, stride=1, expansion=4, final_relu=True):
        super().__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.final_relu = final_relu
        self.drop_path = nn.Identity()
        self.shortcut: nn.Module = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.drop_path(out)
        out = out + self.shortcut(x)
        if self.final_relu:
            out = F.relu(out)
        return out


# --------------------------------------------------------------------------- #
# DAPPM (Deep Aggregation Pyramid Pooling Module)                              #
# --------------------------------------------------------------------------- #


class UpscaleOnline(nn.Module):
    """Bilinear interpolate to a runtime-supplied (H, W)."""

    def __init__(self, mode: str = "bilinear"):
        super().__init__()
        self.mode = mode

    def forward(self, x, output_height: int, output_width: int):
        return F.interpolate(x, size=[output_height, output_width], mode=self.mode)


class DAPPMBranch(nn.Module):
    """One branch of the DAPPM pyramid.

    State-dict layout: ``down_scale.{0..N}`` (AvgPool — no params; BN1, ReLU2, Conv2d),
    ``process.{0..2}`` (BN1, ReLU2, Conv2d) — only if ``stride != 1``.
    """

    def __init__(self, kernel_size: int, stride: int, in_planes: int,
                 branch_planes: int, inter_mode: str = "bilinear"):
        super().__init__()
        down_list: List[nn.Module] = []
        if stride == 0:
            down_list.append(nn.AdaptiveAvgPool2d((1, 1)))
        elif stride == 1:
            pass
        else:
            down_list.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=stride))
        down_list.append(nn.BatchNorm2d(in_planes))
        down_list.append(nn.ReLU(inplace=True))
        down_list.append(nn.Conv2d(in_planes, branch_planes, kernel_size=1, bias=False))
        self.down_scale = nn.Sequential(*down_list)
        self.up_scale = UpscaleOnline(inter_mode)
        if stride != 1:
            self.process = nn.Sequential(
                nn.BatchNorm2d(branch_planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
            )

    def forward(self, x):
        if isinstance(x, list):
            output_of_prev_branch = x[1]
            x = x[0]
        else:
            output_of_prev_branch = None
        in_width = x.shape[-1]
        in_height = x.shape[-2]
        out = self.down_scale(x)
        out = self.up_scale(out, output_height=in_height, output_width=in_width)
        if output_of_prev_branch is not None:
            out = self.process(out + output_of_prev_branch)
        return out


class DAPPM(nn.Module):
    def __init__(self, in_planes: int, branch_planes: int, out_planes: int,
                 kernel_sizes: list, strides: list, inter_mode: str = "bilinear"):
        super().__init__()
        assert len(kernel_sizes) == len(strides), (
            "len of kernel_sizes and strides must be the same"
        )
        self.branches = nn.ModuleList()
        for kernel_size, stride in zip(kernel_sizes, strides):
            self.branches.append(DAPPMBranch(
                kernel_size=kernel_size, stride=stride,
                in_planes=in_planes, branch_planes=branch_planes,
                inter_mode=inter_mode,
            ))
        self.compression = nn.Sequential(
            nn.BatchNorm2d(branch_planes * len(self.branches)),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * len(self.branches), out_planes, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        x_list = []
        for i, branch in enumerate(self.branches):
            if i == 0:
                x_list.append(branch(x))
            else:
                x_list.append(branch([x, x_list[i - 1]]))
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out


class SegmentHead(nn.Module):
    """Final segmentation head: BN-Conv3x3-BN-Conv1x1, then upscale.

    The upscale module (``nn.Upsample``) has no parameters, so the
    state-dict surface is just ``bn1``, ``conv1.weight``, ``bn2``,
    ``conv2.{weight, bias}``.
    """

    def __init__(self, in_planes: int, inter_planes: int, out_planes: int,
                 scale_factor: int, inter_mode: str = "bilinear"):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=1, padding=0, bias=True)
        self.upscale = nn.Upsample(scale_factor=scale_factor, mode=inter_mode)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        return self.upscale(out)


# --------------------------------------------------------------------------- #
# Backbone                                                                     #
# --------------------------------------------------------------------------- #


class BasicDDRBackBone(nn.Module):
    """ResNet-style stem + 4 layer stages (low-resolution branch).

    State-dict layout: ``stem.{0,1}`` (each a ConvBN_with_ReLU
    ``Sequential`` whose params live under ``.0`` (Conv2d) and ``.1``
    (BatchNorm2d)), ``layer{1,2,4}.{0..N}`` (BasicResNetBlocks), and
    ``layer3`` is an ``nn.ModuleList`` of length ``layer3_repeats``,
    each entry a Sequential of ``BasicResNetBlock``s.
    """

    def __init__(self, block: type, width: int, layers: List[int],
                 input_channels: int = 3, layer3_repeats: int = 1):
        super().__init__()
        self.input_channels = input_channels
        self.stem = nn.Sequential(
            ConvBN(input_channels, width, kernel_size=3, stride=2, padding=1, add_relu=True),
            ConvBN(width, width, kernel_size=3, stride=2, padding=1, add_relu=True),
        )
        self.layer1 = _make_layer(block=block, in_planes=width, planes=width, num_blocks=layers[0])
        self.layer2 = _make_layer(block=block, in_planes=width, planes=width * 2, num_blocks=layers[1], stride=2)
        self.layer3 = nn.ModuleList(
            [_make_layer(block=block, in_planes=width * 2, planes=width * 4, num_blocks=layers[2], stride=2)]
            + [_make_layer(block=block, in_planes=width * 4, planes=width * 4, num_blocks=layers[2], stride=1)
               for _ in range(layer3_repeats - 1)]
        )
        self.layer4 = _make_layer(block=block, in_planes=width * 4, planes=width * 8, num_blocks=layers[3], stride=2)

    def get_backbone_output_number_of_channels(self) -> dict:
        """Probe the backbone with a dummy 320x320 input to discover output widths.

        Mirrors super_gradients' helper. We need this once at construction
        time to wire the compression / down / skip layers correctly.
        """
        out: dict = {}
        x = torch.randn(1, self.input_channels, 320, 320)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        out["layer2"] = x.shape[1]
        for layer in self.layer3:
            x = layer(x)
        out["layer3"] = x.shape[1]
        x = self.layer4(x)
        out["layer4"] = x.shape[1]
        return out


# --------------------------------------------------------------------------- #
# DDRNet                                                                       #
# --------------------------------------------------------------------------- #


class DDRNet(nn.Module):
    """super_gradients DDRNet (segmentation mode).

    Constructor signature is a strict subset of upstream's; only the
    knobs the GOOSE-39 config exercises are exposed.
    """

    def __init__(
        self,
        backbone: BasicDDRBackBone,
        additional_layers: List[int],
        num_classes: int,
        highres_planes: int,
        spp_width: int,
        head_width: int,
        use_aux_heads: bool = False,
        ssp_inter_mode: str = "bilinear",
        segmentation_inter_mode: str = "bilinear",
        skip_block: type = BasicResNetBlock,
        layer5_block: type = Bottleneck,
        layer5_bottleneck_expansion: int = 2,
        spp_kernel_sizes: Optional[List[int]] = None,
        spp_strides: Optional[List[int]] = None,
        layer3_repeats: int = 1,
    ):
        super().__init__()
        if spp_kernel_sizes is None:
            spp_kernel_sizes = [1, 5, 9, 17, 0]
        if spp_strides is None:
            spp_strides = [1, 2, 4, 8, 0]

        self.use_aux_heads = use_aux_heads
        self.upscale = UpscaleOnline(ssp_inter_mode)
        self.ssp_inter_mode = ssp_inter_mode
        self.segmentation_inter_mode = segmentation_inter_mode
        self.relu = nn.ReLU(inplace=False)
        self.layer3_repeats = layer3_repeats
        self.num_classes = num_classes

        self._backbone = backbone
        out_chan = self._backbone.get_backbone_output_number_of_channels()

        # layer3_repeats × (compression3, down3, layer3_skip)
        self.compression3 = nn.ModuleList()
        self.down3 = nn.ModuleList()
        self.layer3_skip = nn.ModuleList()
        for i in range(layer3_repeats):
            self.compression3.append(ConvBN(
                in_channels=out_chan["layer3"], out_channels=highres_planes,
                kernel_size=1, bias=False,
            ))
            self.down3.append(ConvBN(
                in_channels=highres_planes, out_channels=out_chan["layer3"],
                kernel_size=3, stride=2, padding=1, bias=False,
            ))
            self.layer3_skip.append(_make_layer(
                in_planes=out_chan["layer2"] if i == 0 else highres_planes,
                planes=highres_planes,
                block=skip_block,
                num_blocks=additional_layers[1],
            ))

        self.compression4 = ConvBN(
            in_channels=out_chan["layer4"], out_channels=highres_planes,
            kernel_size=1, bias=False,
        )

        self.down4 = nn.Sequential(
            ConvBN(in_channels=highres_planes, out_channels=highres_planes * 2,
                   kernel_size=3, stride=2, padding=1, bias=False, add_relu=True),
            ConvBN(in_channels=highres_planes * 2, out_channels=out_chan["layer4"],
                   kernel_size=3, stride=2, padding=1, bias=False),
        )
        self.layer4_skip = _make_layer(
            block=skip_block, in_planes=highres_planes,
            planes=highres_planes, num_blocks=additional_layers[2],
        )
        self.layer5_skip = _make_layer(
            block=layer5_block, in_planes=highres_planes,
            planes=highres_planes, num_blocks=additional_layers[3],
            expansion=layer5_bottleneck_expansion,
        )

        self.layer5 = _make_layer(
            block=layer5_block, in_planes=out_chan["layer4"],
            planes=out_chan["layer4"], num_blocks=additional_layers[0],
            stride=2, expansion=layer5_bottleneck_expansion,
        )

        self.spp = DAPPM(
            in_planes=out_chan["layer4"] * layer5_bottleneck_expansion,
            branch_planes=spp_width,
            out_planes=highres_planes * layer5_bottleneck_expansion,
            inter_mode=ssp_inter_mode,
            kernel_sizes=spp_kernel_sizes,
            strides=spp_strides,
        )

        self.final_layer = SegmentHead(
            in_planes=highres_planes * layer5_bottleneck_expansion,
            inter_planes=head_width,
            out_planes=num_classes,
            scale_factor=8,
            inter_mode=segmentation_inter_mode,
        )

        if self.use_aux_heads:
            self.seghead_extra = SegmentHead(
                in_planes=highres_planes,
                inter_planes=head_width,
                out_planes=num_classes,
                scale_factor=8,
                inter_mode=segmentation_inter_mode,
            )

    def forward(self, x):
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8

        x = self._backbone.stem(x)
        x = self._backbone.layer1(x)
        x = self._backbone.layer2(self.relu(x))

        x_skip = x
        for i in range(self.layer3_repeats):
            out_layer3 = self._backbone.layer3[i](self.relu(x))
            out_layer3_skip = self.layer3_skip[i](self.relu(x_skip))
            x = out_layer3 + self.down3[i](self.relu(out_layer3_skip))
            x_skip = out_layer3_skip + self.upscale(
                self.compression3[i](self.relu(out_layer3)),
                height_output, width_output,
            )

        if self.use_aux_heads:
            temp = x_skip

        out_layer4 = self._backbone.layer4(self.relu(x))
        out_layer4_skip = self.layer4_skip(self.relu(x_skip))
        x = out_layer4 + self.down4(self.relu(out_layer4_skip))
        x_skip = out_layer4_skip + self.upscale(
            self.compression4(self.relu(out_layer4)),
            height_output, width_output,
        )

        out_layer5_skip = self.layer5_skip(self.relu(x_skip))

        x = self.upscale(self.spp(self.layer5(self.relu(x))), height_output, width_output)
        x = self.final_layer(x + out_layer5_skip)

        if self.use_aux_heads:
            return x, self.seghead_extra(temp)
        return x


# --------------------------------------------------------------------------- #
# Builders                                                                     #
# --------------------------------------------------------------------------- #


def ddrnet_39_goose(num_classes: int = 64, use_aux_heads: bool = False) -> DDRNet:
    """Build the DDRNet-39 variant used by the GOOSE benchmark.

    Defaults from super_gradients' ``DEFAULT_DDRNET_39_PARAMS``:

        layers = [3, 4, 3, 3, 1, 3, 3, 1]
        planes = 64
        highres_planes = 128
        head_planes = 256
        spp_planes = 128
        layer3_repeats = 2
        layer5_block = Bottleneck (expansion=2)

    For the GOOSE *category*-level checkpoint pass ``num_classes=12``;
    for the *fine* GOOSE-64 checkpoint pass ``num_classes=64``. Both
    variants share this backbone and head structure (only the final
    1×1 ``conv2`` differs in its output channel count).
    """
    layers = [3, 4, 3, 3, 1, 3, 3, 1]
    backbone_layers = layers[:4]
    additional_layers = layers[4:]
    backbone = BasicDDRBackBone(
        block=BasicResNetBlock,
        width=64,
        layers=backbone_layers,
        input_channels=3,
        layer3_repeats=2,
    )
    return DDRNet(
        backbone=backbone,
        additional_layers=additional_layers,
        num_classes=num_classes,
        highres_planes=128,
        spp_width=128,
        head_width=256,
        use_aux_heads=use_aux_heads,
        ssp_inter_mode="bilinear",
        segmentation_inter_mode="bilinear",
        skip_block=BasicResNetBlock,
        layer5_block=Bottleneck,
        layer5_bottleneck_expansion=2,
        spp_kernel_sizes=[1, 5, 9, 17, 0],
        spp_strides=[1, 2, 4, 8, 0],
        layer3_repeats=2,
    )

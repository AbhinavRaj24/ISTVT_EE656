"""
backbone.py
===========

Xception Entry Flow backbone used in ISTVT.

Paper:
ISTVT: Interpretable Spatial-Temporal Video Transformer for Deepfake Detection
(IEEE TIFS 2023)

The paper only uses the Entry Flow of Xception as the feature extractor.
Given a 300×300 RGB face image, this module outputs a feature map of:

    (B, 728, 19, 19)

This implementation is adapted and simplified from the official ISTVT
repository.
"""

import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


# -----------------------------------------------------------------------------
# Depthwise Separable Convolution
# -----------------------------------------------------------------------------


class SeparableConv2d(nn.Module):
    """
    Depthwise Separable Convolution used throughout Xception.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=False,
    ):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=in_channels,
            bias=bias,
        )

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# -----------------------------------------------------------------------------
# Residual Block
# -----------------------------------------------------------------------------


class Block(nn.Module):
    """
    Residual Xception block.
    """

    def __init__(
        self,
        in_filters,
        out_filters,
        reps,
        stride=1,
        start_with_relu=True,
        grow_first=True,
    ):
        super().__init__()

        if out_filters != in_filters or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_filters,
                    out_filters,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_filters),
            )
        else:
            self.skip = nn.Identity()

        layers = []

        filters = in_filters

        if grow_first:
            layers.extend(
                [
                    nn.ReLU(inplace=True),
                    SeparableConv2d(in_filters, out_filters),
                    nn.BatchNorm2d(out_filters),
                ]
            )
            filters = out_filters

        for _ in range(reps - 1):
            layers.extend(
                [
                    nn.ReLU(inplace=True),
                    SeparableConv2d(filters, filters),
                    nn.BatchNorm2d(filters),
                ]
            )

        if not grow_first:
            layers.extend(
                [
                    nn.ReLU(inplace=True),
                    SeparableConv2d(in_filters, out_filters),
                    nn.BatchNorm2d(out_filters),
                ]
            )

        if not start_with_relu:
            layers = layers[1:]
        else:
            layers[0] = nn.ReLU(inplace=False)

        if stride != 1:
            layers.append(nn.MaxPool2d(3, stride, 1))

        self.rep = nn.Sequential(*layers)

    def forward(self, x):
        residual = self.skip(x)
        x = self.rep(x)
        return x + residual


# -----------------------------------------------------------------------------
# Xception Entry Flow
# -----------------------------------------------------------------------------


class XceptionBackbone(nn.Module):
    """
    Entry Flow of Xception.

    Input:
        (B, 3, 300, 300)

    Output:
        (B, 728, 19, 19)
    """

    def __init__(self, pretrained=True):
        super().__init__()

        # Initial convolutions
        self.conv1 = nn.Conv2d(
            3,
            32,
            kernel_size=3,
            stride=2,
            padding=0,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            32,
            64,
            kernel_size=3,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        # Entry Flow
        self.block1 = Block(
            64,
            128,
            reps=2,
            stride=2,
            start_with_relu=False,
            grow_first=True,
        )

        self.block2 = Block(
            128,
            256,
            reps=2,
            stride=2,
            start_with_relu=True,
            grow_first=True,
        )

        self.block3 = Block(
            256,
            728,
            reps=2,
            stride=2,
            start_with_relu=True,
            grow_first=True,
        )

        if pretrained:
            state_dict = model_zoo.load_url(
                "http://data.lip6.fr/cadene/pretrainedmodels/xception-43020ad28.pth"
            )

            self.load_state_dict(state_dict, strict=False)

        else:
            self._initialize_weights()

    def _initialize_weights(self):
        """
        Kaiming initialization.
        """

        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                n = (
                    m.kernel_size[0]
                    * m.kernel_size[1]
                    * m.out_channels
                )
                m.weight.data.normal_(0, math.sqrt(2.0 / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        x = self.relu1(self.bn1(self.conv1(x)))

        x = self.relu2(self.bn2(self.conv2(x)))

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        return x
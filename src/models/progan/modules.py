import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import (
    AvgPool2d,
    BatchNorm2d,
    Conv2d,
    ConvTranspose2d,
    Embedding,
    LeakyReLU,
    Module,
    init,
)

from .batchnorm import CategoricalConditionalBatchNorm2d
from .custom_layers import (
    EqualizedConv2d,
    EqualizedConvTranspose2d,
    MinibatchStdDev,
    PixelwiseNorm,
)


class GenInitialBlock(Module):
    """
    Module implementing the initial block of the input
    Args:
        in_channels: number of input channels to the block
        out_channels: number of output channels of the block
        use_eql: whether to use equalized learning rate
    """

    def __init__(
        self, in_channels: int, out_channels: int, n_classes: int, use_eql: bool
    ) -> None:
        super(GenInitialBlock, self).__init__()
        self.use_eql = use_eql

        ConvBlock = EqualizedConv2d if use_eql else Conv2d
        ConvTransposeBlock = EqualizedConvTranspose2d if use_eql else ConvTranspose2d

        self.conv_1 = ConvTransposeBlock(in_channels, out_channels, (4, 4), bias=True)
        self.conv_2 = ConvBlock(
            out_channels, out_channels, (3, 3), padding=1, bias=True
        )
        self.norm = PixelwiseNorm()
        self.activation = LeakyReLU(0.2)
        self.bn1 = CategoricalConditionalBatchNorm2d(n_classes, in_channels)

    def forward(self, x: Tensor, labels: Tensor = None) -> Tensor:
        out = torch.unsqueeze(torch.unsqueeze(x, -1), -1)
        out = self.norm(out)
        out = self.activation(self.conv_1(out))
        out = self.activation(self.conv_2(out))
        out = self.norm(out)
        return out


class GenGeneralConvBlock(torch.nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, n_classes: int, use_eql: bool
    ) -> None:
        super(GenGeneralConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.use_eql = use_eql

        ConvBlock = EqualizedConv2d if use_eql else Conv2d

        self.conv1 = ConvBlock(in_channels, out_channels, (3, 3), padding=1, bias=True)
        self.conv2 = ConvBlock(out_channels, out_channels, (3, 3), padding=1, bias=True)
        self.pixel_norm = PixelwiseNorm()
        self.activation = LeakyReLU(0.2)

    def forward(self, x: Tensor, labels: Tensor = None) -> Tensor:
        y = F.interpolate(x, scale_factor=2)
        y = self.pixel_norm(self.activation(self.conv1(y)))
        y = self.pixel_norm(self.activation(self.conv2(y)))

        return y


class GenConditionalConvBlock(torch.nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, n_classes: int, use_eql: bool
    ) -> None:
        super(GenConditionalConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.use_eql = use_eql

        ConvBlock = EqualizedConv2d if use_eql else Conv2d

        self.conv1 = ConvBlock(in_channels, out_channels, (3, 3), padding=1, bias=True)
        self.conv2 = ConvBlock(out_channels, out_channels, (3, 3), padding=1, bias=True)
        self.norm = PixelwiseNorm()
        self.activation = LeakyReLU(0.2)

        self.bn1 = BatchNorm2d(in_channels)

    def forward(self, x: Tensor, labels: Tensor = None) -> Tensor:

        x = self.bn1(x)
        x = F.interpolate(x, scale_factor=2)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.norm(x)

        x = self.conv2(x)
        x = self.activation(x)
        x = self.norm(x)
        return x


class DisFinalBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_eql: bool) -> None:
        super(DisFinalBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_eql = use_eql

        ConvBlock = EqualizedConv2d if use_eql else Conv2d

        self.conv1 = ConvBlock(
            in_channels + 1, in_channels, (3, 3), padding=1, bias=True
        )
        self.conv2 = ConvBlock(in_channels, out_channels, (4, 4), bias=True)
        self.conv3 = ConvBlock(out_channels, 1, (1, 1), bias=True)
        self.batch_discriminator = MinibatchStdDev()
        self.activation = LeakyReLU(0.2)

    def forward(self, x: Tensor) -> Tensor:
        y = self.batch_discriminator(x)
        y = self.activation(self.conv1(y))
        y = self.activation(self.conv2(y))
        y = self.conv3(y)
        return y.view(-1)


class ConDisFinalBlock(torch.nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, num_classes: int, use_eql: bool
    ) -> None:
        super(ConDisFinalBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.use_eql = use_eql

        ConvBlock = EqualizedConv2d if use_eql else Conv2d

        self.conv1 = ConvBlock(
            in_channels + 1, in_channels, (3, 3), padding=1, bias=True
        )
        self.conv2 = ConvBlock(in_channels, out_channels, (4, 4), bias=True)
        self.conv3 = ConvBlock(out_channels, 1, (1, 1), bias=True)

        self.label_embedder = Embedding(num_classes, out_channels)
        self.batch_discriminator = MinibatchStdDev()
        self.activation = LeakyReLU(0.2)

    def forward(self, x: Tensor, labels: Tensor) -> Tensor:
        y = self.batch_discriminator(x)
        y = self.activation(self.conv1(y))
        y = self.activation(self.conv2(y))

        labels = self.label_embedder(labels)  #

        y_ = torch.squeeze(torch.squeeze(y, dim=-1), dim=-1)
        projection_scores = (y_ * labels).sum(dim=-1)

        y = self.activation(self.conv3(y))
        final_score = y.view(-1) + projection_scores

        return final_score


class DisGeneralConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_eql: bool) -> None:
        super(DisGeneralConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_eql = use_eql

        ConvBlock = EqualizedConv2d if use_eql else Conv2d

        self.conv1 = ConvBlock(in_channels, in_channels, (3, 3), padding=1, bias=True)
        self.conv2 = ConvBlock(in_channels, out_channels, (3, 3), padding=1, bias=True)
        self.downsample = AvgPool2d(2)
        self.activation = LeakyReLU(0.2)

    def forward(self, x: Tensor) -> Tensor:
        y = self.activation(self.conv1(x))
        y = self.activation(self.conv2(y))
        y = self.downsample(y)
        return y

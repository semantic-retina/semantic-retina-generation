from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import Conv2d, Embedding, LeakyReLU, ModuleList, Sequential
from torch.nn.functional import avg_pool2d, interpolate

from .custom_layers import EqualizedConv2d
from .modules import (
    ConDisFinalBlock,
    DisFinalBlock,
    DisGeneralConvBlock,
    GenGeneralConvBlock,
    GenInitialBlock,
)


def nf(
    stage: int,
    fmap_base: int = 16 << 10,
    fmap_decay: float = 1.0,
    fmap_min: int = 1,
    fmap_max: int = 512,
) -> int:
    return int(
        np.clip(
            int(fmap_base / (2.0 ** (stage * fmap_decay))),
            fmap_min,
            fmap_max,
        ).item()
    )


class Generator(nn.Module):
    """
    Generator Module (block) of the GAN network
    Args:
        depth: required depth of the Network
        n_channels: number of output channels (default = 3 for RGB)
        latent_size: size of the latent manifold
        use_eql: whether to use equalized learning rate
    """

    def __init__(
        self,
        depth: int = 10,
        n_channels: int = 3,
        latent_size: int = 512,
        use_eql: bool = True,
        n_classes: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.depth = depth
        self.latent_size = latent_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.conditional = n_classes is not None
        self.use_eql = use_eql
        self.softmax = nn.Softmax(dim=1)

        if n_classes is not None:
            self.label_embedder = Embedding(self.n_classes, self.n_classes)
            self.label_embedder.weight.data = torch.eye(self.n_classes)

        ConvBlock = EqualizedConv2d if use_eql else Conv2d

        self.layers = ModuleList(
            [GenInitialBlock(latent_size, nf(1), n_classes, self.use_eql)]
        )
        for stage in range(1, depth - 1):
            in_channels = nf(stage)
            out_channels = nf(stage + 1)
            self.layers.append(
                GenGeneralConvBlock(in_channels, out_channels, n_classes, use_eql)
            )

        self.rgb_converters = ModuleList(
            [
                ConvBlock(nf(stage), n_channels, kernel_size=(1, 1))
                for stage in range(1, depth)
            ]
        )

    def forward(self, x: Tensor, labels: Tensor, depth: int, alpha: float) -> Tensor:
        if labels is not None:
            labels = self.label_embedder(labels.view(-1))
            x = torch.cat((labels, x), dim=-1)

        assert depth <= self.depth, f"Requested output depth {depth} cannot be produced"
        if self.conditional:
            assert labels is not None, "Conditional generator requires labels"

        if depth == 2:
            out = self.rgb_converters[0](self.layers[0](x, labels=labels))
        else:
            out = x
            for layer_block in self.layers[: depth - 2]:
                out = layer_block(out, labels=labels)
            residual = interpolate(self.rgb_converters[depth - 3](out), scale_factor=2)
            straight = self.rgb_converters[depth - 2](
                self.layers[depth - 2](out, labels=labels)
            )
            out = (alpha * straight) + ((1 - alpha) * residual)

        return self.softmax(out)


class Discriminator(nn.Module):
    def __init__(
        self,
        depth: int = 7,
        num_channels: int = 3,
        latent_size: int = 512,
        use_eql: bool = True,
        n_classes: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.num_channels = num_channels
        self.latent_size = latent_size
        self.use_eql = use_eql
        self.n_classes = n_classes
        self.conditional = n_classes is not None

        ConvBlock = EqualizedConv2d if use_eql else Conv2d

        if self.conditional:
            self.layers = [ConDisFinalBlock(nf(1), latent_size, n_classes, use_eql)]
        else:
            self.layers = [DisFinalBlock(nf(1), latent_size, use_eql)]

        for stage in range(1, depth - 1):
            self.layers.insert(
                0, DisGeneralConvBlock(nf(stage + 1), nf(stage), use_eql)
            )
        self.layers = ModuleList(self.layers)
        self.from_rgb = ModuleList(
            reversed(
                [
                    Sequential(
                        ConvBlock(num_channels, nf(stage), kernel_size=(1, 1)),
                        LeakyReLU(0.2),
                    )
                    for stage in range(1, depth)
                ]
            )
        )

    def forward(
        self, x: Tensor, depth: int, alpha: float, labels: Optional[Tensor] = None
    ) -> Tensor:

        assert (
            depth <= self.depth
        ), f"Requested output depth {depth} cannot be evaluated"

        if self.conditional:
            assert labels is not None, "Conditional discriminator requires labels"

        if depth > 2:
            residual = self.from_rgb[-(depth - 2)](
                avg_pool2d(x, kernel_size=2, stride=2)
            )
            straight = self.layers[-(depth - 1)](self.from_rgb[-(depth - 1)](x))
            out = (alpha * straight) + ((1 - alpha) * residual)
            for layer_block in self.layers[-(depth - 2) : -1]:
                out = layer_block(out)
        else:
            out = self.from_rgb[-1](x)

        if self.conditional:
            out = self.layers[-1](out, labels)
        else:
            out = self.layers[-1](out)

        return out

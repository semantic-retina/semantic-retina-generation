from typing import List, Tuple

import torch.nn as nn
from torch import Tensor


class Discriminator(nn.Module):
    def __init__(self, in_channels: int, img_size: int, n_classes: int):
        super(Discriminator, self).__init__()

        hidden_channels = 4
        out_channels = hidden_channels * 8

        self.conv_blocks = nn.Sequential(
            *self.discriminator_block(in_channels, hidden_channels * 2),
            *self.discriminator_block(hidden_channels * 2, hidden_channels * 4),
            *self.discriminator_block(hidden_channels * 4, hidden_channels * 8),
            *self.discriminator_block(hidden_channels * 8, hidden_channels * 8),
            *self.discriminator_block(hidden_channels * 8, hidden_channels * 8),
            *self.discriminator_block(hidden_channels * 8, out_channels),
        )

        # The height and width of downsampled image.
        ds_size = img_size // 2 ** 6

        # Output layers.
        self.adv_layer = nn.Sequential(nn.Linear(out_channels * ds_size ** 2, 1))
        self.aux_layer = nn.Sequential(
            nn.Linear(out_channels * ds_size ** 2, n_classes), nn.LogSoftmax(dim=1)
        )

    def discriminator_block(self, c_in: int, c_out: int) -> List[nn.Module]:
        """Returns layers of each discriminator block."""
        return [
            # Output shape ⌊(N - 1) / 2⌋ + 1.
            nn.Conv2d(c_in, c_out, 3, 2, 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(0.2, inplace=True),
        ]

    def forward(self, img: Tensor) -> Tuple[Tensor, Tensor]:
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label

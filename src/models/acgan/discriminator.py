import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, channels, img_size, n_classes):
        super(Discriminator, self).__init__()

        self.conv_blocks = nn.Sequential(
            *self.discriminator_block(channels, 16, bn=False),
            *self.discriminator_block(16, 32),
            *self.discriminator_block(32, 64),
            *self.discriminator_block(64, 128),
        )

        # The height and width of downsampled image.
        ds_size = img_size // 2 ** 4

        # Output layers.
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, n_classes), nn.Softmax(dim=1)
        )

    def discriminator_block(self, in_channels, out_channels, bn=True):
        """Returns layers of each discriminator block."""
        block = [
            nn.Conv2d(in_channels, out_channels, 3, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if bn:
            block.append(nn.BatchNorm2d(out_channels))
        return block

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label

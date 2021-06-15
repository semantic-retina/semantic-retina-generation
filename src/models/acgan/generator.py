import torch
import torch.nn as nn
from torch import Tensor


class Generator(nn.Module):
    def __init__(
        self,
        out_channels: int,
        img_size: int,
        n_classes: int,
        latent_dim: int,
        concat: bool = False,
    ):
        super(Generator, self).__init__()

        self.concat = concat
        if concat:
            self.label_emb = nn.Embedding(n_classes, n_classes)
        else:
            self.label_emb = nn.Embedding(n_classes, latent_dim)

        self.init_size = img_size // 2 ** 6  # Initial size before upsampling

        self.hidden_channels = 512

        if concat:
            self.l1 = nn.Sequential(
                nn.Linear(
                    latent_dim + n_classes, self.hidden_channels * self.init_size ** 2
                ),
            )
        else:
            self.l1 = nn.Sequential(
                nn.Linear(latent_dim, self.hidden_channels * self.init_size ** 2),
            )

        self.conv_blocks = nn.Sequential(
            *self.generator_block(self.hidden_channels, self.hidden_channels // 2),
            *self.generator_block(self.hidden_channels // 2, self.hidden_channels // 4),
            *self.generator_block(self.hidden_channels // 4, self.hidden_channels // 8),
            *self.generator_block(
                self.hidden_channels // 8, self.hidden_channels // 16
            ),
            *self.generator_block(
                self.hidden_channels // 16, self.hidden_channels // 32
            ),
            *self.generator_block(self.hidden_channels // 32, out_channels),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

        # Softmax is applied across the channels to obtain a semantic map.
        self.activation = nn.Softmax(dim=1)

    def generator_block(self, c_in: int, c_out: int):
        return [
            nn.Upsample(scale_factor=2),
            # Output shape N.
            nn.Conv2d(c_in, c_out, 3, 1, 1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        ]

    def forward(self, noise: Tensor, labels: Tensor):

        if self.concat:
            gen_input = torch.cat((noise, self.label_emb(labels)), dim=1)
        else:
            gen_input = torch.mul(self.label_emb(labels), noise)

        out = self.l1(gen_input)
        out = out.view(
            out.shape[0], self.hidden_channels, self.init_size, self.init_size
        )
        img = self.conv_blocks(out)
        img = self.activation(img)
        return img


def test():
    model = Generator(3, 128, 5, 100)
    num_param = sum(p.numel() for p in model.parameters())
    print("Number of parameters: {}".format(num_param))


if __name__ == "__main__":
    test()

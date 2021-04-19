import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, channels, img_size, n_classes, latent_dim):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, latent_dim)

        self.init_size = img_size // 2 ** 3  # Initial size before upsampling
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 256 * self.init_size ** 2),
        )

        self.conv_blocks = nn.Sequential(
            *self.generator_block(256, 128, 64),
            *self.generator_block(64, 32, 16),
            *self.generator_block(16, 8, channels),
            nn.Conv2d(channels, channels, 3, stride=1, padding=1),
        )

        self.activation = nn.Softmax(dim=1)

    def generator_block(self, in_channels, mid_channels, out_channels):
        return [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, mid_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        img = self.activation(img)
        return img


def test():
    model = Generator(3, 128, 5, 100)
    num_param = sum(p.numel() for p in model.parameters())
    print("Number of parameters: {}".format(num_param))


if __name__ == "__main__":
    test()

from pathlib import Path

import torch
from torch import nn
from torchvision.utils import save_image


def sample_gan(
    output_path: Path,
    generator: nn.Module,
    device: torch.device,
    n_channels: int,
    latent_dim: int,
    n_row: int,
    epoch: int,
):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise for each image.
    z = torch.randn((n_row ** 2, latent_dim), device=device)
    # Get labels ranging from 0 to n_classes for n_rows.
    labels = torch.tensor(
        [num for _ in range(n_row) for num in range(n_row)], device=device
    )
    with torch.no_grad():
        gen_imgs = generator(z, labels)
    max_val = torch.argmax(gen_imgs, dim=1, keepdim=True)
    print(max_val.unique())
    scaled_val = max_val / float(n_channels - 1)

    output_file = output_path / f"{epoch:05}.png"
    print(f"Saved image to {output_file}")
    save_image(scaled_val.data, output_file, nrow=n_row, normalize=False, pad_value=1)

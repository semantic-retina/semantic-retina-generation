from pathlib import Path

import torch
from torch import Tensor, nn
from torchvision.utils import save_image

COLOUR_MAP = torch.tensor(
    [
        [87, 117, 144],  # Retina
        [41, 102, 84],  # OD
        [228, 92, 229],  # MA
        [180, 211, 156],  # HE
        [248, 150, 30],  # EX
        [249, 65, 68],  # SE
        [255, 255, 255],  # BG
    ],
    dtype=torch.long,
)


def colour_labels(gen_imgs: Tensor) -> Tensor:
    """
    Turns semantic labels into human-discernible colours.

    :param: Expects a tensor of shape B x C x H x W with values in range [0, NC] where
    NC is the maximum number of labels.

    :returns: A tensor of shape B x 3 x H x W with values in range [0, 1].
    """
    assert gen_imgs.ndim == 4
    batch_size, channels, height, width = gen_imgs.shape
    gray_image = torch.argmax(gen_imgs, dim=1)
    coloured = torch.empty(batch_size, 3, height, width, dtype=torch.long)
    nc = 6 + 1
    for label in range(nc):
        mask = gray_image == label
        coloured[:, 0, :, :][mask] = COLOUR_MAP[label, 0]
        coloured[:, 1, :, :][mask] = COLOUR_MAP[label, 1]
        coloured[:, 2, :, :][mask] = COLOUR_MAP[label, 2]

    return coloured.float() / 255.0


def sample_gan(
    output_path: Path,
    generator: nn.Module,
    device: torch.device,
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

    coloured_val = colour_labels(gen_imgs)

    output_file = output_path / f"{epoch:05}.png"
    print(f"Saved image to {output_file}")
    save_image(coloured_val.data, output_file, nrow=n_row, normalize=False, pad_value=1)

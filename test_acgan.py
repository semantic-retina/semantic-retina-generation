import json
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.utils import save_image

from src.data.common import Labels
from src.models.acgan.generator import Generator
from src.options.acgan.test import get_args
from src.utils.sample import colour_labels
from src.utils.seed import set_seed


def load_generator(
    path: Path, channels: int, img_size: int, n_classes: int, latent_dim: int
) -> nn.Module:
    model = Generator(channels, img_size, n_classes, latent_dim)
    model.load_state_dict(torch.load(path))
    model.eval()

    return model


def split_channels(imgs: Tensor) -> Tuple[Tensor, Tensor]:
    label = imgs[:, :, :, :]
    inst = imgs[:, [1], :, :]
    return label, inst


def main():
    opt = get_args()

    path = Path("results") / "acgan" / opt.name
    checkpoint_path = path / "checkpoints"

    with open(path / "opt.json", "r") as f:
        opt_train = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    n_classes = 5
    which_labels = sorted([Labels[l] for l in opt_train.lesions], key=lambda x: x.value)
    n_channels = len(which_labels) + 1

    out_path = Path("results") / "acgan" / opt.name / "test"
    label_out_path = out_path / "label"
    inst_out_path = out_path / "inst"
    label_out_path.mkdir(exist_ok=True, parents=True)
    inst_out_path.mkdir(exist_ok=True, parents=True)

    set_seed(opt.seed)

    device = torch.device("cuda")

    checkpoint_suffix = "final"
    if opt.epoch:
        checkpoint_suffix = opt.epoch

    generator = load_generator(
        checkpoint_path / f"generator_{checkpoint_suffix}.pth",
        n_channels,
        opt_train.img_size,
        n_classes,
        opt_train.latent_dim,
    )
    generator.to(device)

    batch_sizes = [opt.batch_size for _ in range(opt.n_samples // opt.batch_size)]
    last_batch = opt.n_samples % opt.batch_size
    if last_batch > 0:
        batch_sizes += [last_batch]

    print(f"Batch sizes: {batch_sizes}")

    for i, bs in enumerate(batch_sizes):
        z = torch.randn((bs, opt_train.latent_dim), device=device)
        # gen_label = torch.tensor([dr_grade], device=device)
        gen_label = torch.randint(n_classes, (bs,), device=device)
        outputs = generator(z, gen_label).detach()

        if opt.upsample_factor > 1:
            # Interestingly, transforms.Resize now gives a warning when not using the
            # `InterpolationMode` enum, but the same is not true for
            # `functional.interpolate`.
            outputs = F.interpolate(
                outputs,
                scale_factor=opt.upsample_factor,
                mode="bilinear",
                align_corners=False,
            )
            outputs = torch.where(outputs > 0.5, 1.0, 0.0)

        labels, inst = split_channels(outputs)

        inst = torch.ones_like(inst) - inst
        inst *= 255

        # At this point, `labels` contains values in the range [0, 255]. Pytorch's
        # `save_image` function expects values between [0, 1]. `colour_labels` does this
        # scaling for us, otherwise we do it here.
        if opt.colour:
            labels = colour_labels(labels)
        else:
            labels /= 255.0

        for j in range(bs):
            total_idx = i * bs + j
            dr_grade = gen_label[j].item()
            # Save as PNG to avoid compression artifacts.
            filename = f"test_{dr_grade}_{total_idx:05}.png"
            label_path = str(out_path / "label" / filename)
            save_image(labels[j], label_path)

            inst_path = str(out_path / "inst" / filename)
            save_image(inst[j], inst_path)

        print(f"Generated images of shape {labels.shape}")


if __name__ == "__main__":
    main()

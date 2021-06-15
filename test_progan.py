import json
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import DataParallel

from src.data.common import Labels
from src.models.progan.networks import Generator
from src.options.progan.test import get_args
from src.utils.sample import colour_labels_flat
from src.utils.seed import set_seed


def load_generator(
    path: Path, channels: int, img_size: int, n_classes: int, latent_dim: int
) -> nn.Module:
    model = DataParallel(
        Generator(
            n_channels=channels, depth=9, n_classes=n_classes, latent_size=latent_dim
        )
    )
    model.load_state_dict(torch.load(path))
    model.eval()

    return model


def split_channels(imgs: Tensor) -> Tuple[Tensor, Tensor]:
    label = imgs[:, :, :, :]
    inst = imgs[:, [1], :, :]
    return label, inst


def main():
    opt = get_args()

    path = Path("results") / "progan" / opt.name
    checkpoint_path = path / "checkpoints"

    with open(path / "opt.json", "r") as f:
        opt_train = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    n_classes = 5
    which_labels = sorted([Labels[l] for l in opt_train.lesions], key=lambda x: x.value)
    n_channels = len(which_labels) + 1

    out_path = Path(opt.out_dir) / "progan" / opt.name / "test"
    label_out_path = out_path / "label"
    inst_out_path = out_path / "inst"
    label_out_path.mkdir(exist_ok=True, parents=True)
    inst_out_path.mkdir(exist_ok=True, parents=True)

    if opt.seed > -1:
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

    total_idx = 0
    for i, bs in enumerate(batch_sizes):
        if n_classes is None:
            gen_label = None
            z = torch.randn((bs, opt_train.latent_dim), device=device)
        else:
            z = torch.randn((bs, opt_train.latent_dim - n_classes), device=device)
            gen_label = torch.randint(n_classes, (bs,), device=device)
        outputs = generator(z, labels=gen_label, depth=9, alpha=1.0).detach()

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
        labels = torch.argmax(labels, dim=1, keepdim=True).float()

        if opt.mask_retina:
            _, height, width = labels[0].shape
            image = Labels.BG.value * np.ones((height, width))
            center = (height // 2, width // 2)
            radius = height // 2 - 1
            colour = Labels.RETINA.value
            circle = cv2.circle(image, center, radius, colour, thickness=cv2.FILLED)
            circle_tensor = torch.from_numpy(circle)
            circle_tensor = circle_tensor.to(device).float()
            labels = torch.where(labels == Labels.RETINA.value, circle_tensor, labels)
            labels = torch.where(labels == Labels.BG.value, circle_tensor, labels)

        if opt.colour:
            labels = colour_labels_flat(labels) * 255.0
        else:
            labels[labels == Labels.BG.value] = 255

        # We permute instead of squeezing since the labels may have colour channels.
        labels = labels.permute(0, 2, 3, 1).cpu().numpy()
        inst = inst.permute(0, 2, 3, 1).cpu().numpy()

        for j in range(bs):
            total_idx += 1
            if gen_label is None:
                dr_grade = ""
            else:
                dr_grade = gen_label[j].item()
            # Save as PNG to avoid compression artifacts.
            filename = f"test_{dr_grade}_{total_idx:05}.png"
            label_path = str(out_path / "label" / filename)

            if opt.colour:
                label = cv2.cvtColor(labels[j], cv2.COLOR_BGR2RGB)
            else:
                label = labels[j]

            cv2.imwrite(label_path, label)

            inst_path = str(out_path / "inst" / filename)
            cv2.imwrite(
                inst_path,
                inst[j],
            )

        print(f"Generated images of shape {labels.shape}")


if __name__ == "__main__":
    main()

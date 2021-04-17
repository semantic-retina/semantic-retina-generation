import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.models.acgan.generator import Generator


def load_generator(
    path: Path, channels: int, img_size: int, n_classes: int, latent_dim: int
) -> nn.Module:
    model = Generator(channels, img_size, n_classes, latent_dim)
    model.load_state_dict(torch.load(path))
    model.eval()

    return model


def rearrange(imgs: Tensor) -> Tuple[Tensor, Tensor]:
    # input: bg 0, od 1, ex 2
    # (1, 512, 512)
    label = imgs[:, :, :, :]
    inst = imgs[:, [1], :, :]
    return label, inst


def main():
    # name = "acgan-256"
    name = "acgan-512"
    path = Path("results") / "acgan-256" / "checkpoints"
    out_path = Path("results") / "test" / name
    label_out_path = Path("results") / "test" / name / "label"
    inst_out_path = Path("results") / "test" / name / "inst"
    label_out_path.mkdir(exist_ok=True, parents=True)
    inst_out_path.mkdir(exist_ok=True, parents=True)
    latent_dim = 100
    channels = 3
    img_size = 256
    n_classes = 5
    n = 3000
    batch_size = 128
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda")

    generator = load_generator(
        path / "generator_final.pth", channels, img_size, n_classes, latent_dim
    )
    generator.to(device)

    batch_sizes = [batch_size for _ in range(n // batch_size)]
    last_batch = n % batch_size
    if last_batch > 0:
        batch_sizes += [last_batch]

    print(f"Batch sizes: {batch_sizes}")

    for i, bs in enumerate(batch_sizes):
        z = torch.randn((bs, latent_dim), device=device)
        # gen_label = torch.tensor([dr_grade], device=device)
        gen_label = torch.randint(5, (bs,), device=device)

        outputs = generator(z, gen_label).detach()

        upsample = True
        if upsample:
            outputs = F.interpolate(
                # 128 to 512
                outputs,
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            )
            outputs = torch.where(outputs > 0.5, 1.0, 0.0)

        label, inst = rearrange(outputs)
        # Hacky way to make EX correspond to 4.
        # We have BG:0, OD: 1, EX: 2, we need EX to be 4, so insert some extra channels
        # empty_channel = torch.zeros(label.shape[2], label.shape[3])
        label = torch.argmax(label, dim=1).float()
        fours = torch.ones_like(label) * 4
        label = torch.where(torch.eq(label, 2), fours, label)
        inst = inst.squeeze()
        inst = torch.ones_like(inst) - inst
        inst *= 255

        scale = False
        if scale:
            label = (label * 255) / float(channels - 1)

        mask_retina = True
        if mask_retina:
            height, width = label[0].shape
            image = 255 * np.ones((height, width))
            center = (height // 2, width // 2)
            radius = height // 2 - 1
            colour = 0
            circle = cv2.circle(image, center, radius, colour, thickness=cv2.FILLED)
            circle_tensor = torch.from_numpy(circle)
            circle_tensor = circle_tensor.to(device)
            label[:] += circle_tensor
            threshold = 255 * torch.ones((height, width)).to(device)
            label = torch.where(label.gt(255), threshold, label)

        label = label.cpu().numpy()
        inst = inst.cpu().numpy()
        print(f"Label: {label.shape} {np.unique(label)}")
        print(f"Instance: {inst.shape} {np.unique(inst)}")

        for j in range(bs):
            total_idx = i * bs + j
            dr_grade = gen_label[j].item()
            # Save as PNG to avoid compression artifacts.
            out_file = str(out_path / "label" / f"test_{dr_grade}_{total_idx:05}.png")
            ok = cv2.imwrite(out_file, label[j])
            if not ok:
                print("Failed!")

            out_file = str(out_path / "inst" / f"test_{dr_grade}_{total_idx:05}.png")
            ok = cv2.imwrite(out_file, inst[j])
            if not ok:
                print("Failed!")

        print(f"Generated images of shape {label.shape}")


if __name__ == "__main__":
    main()

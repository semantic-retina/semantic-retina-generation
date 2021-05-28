"""Generates images by randomly sampling."""
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.common import Labels, get_label_semantics, infinite
from src.data.datasets.combined import CombinedDataset
from src.options.copy_paste import get_args
from src.utils.sample import colour_labels_numpy
from src.utils.seed import set_seed


def make_circle(height: int, width: int) -> Tensor:
    ones = np.zeros((height, width))
    center = (height // 2, width // 2)
    radius = height // 2 - 1
    circle = cv2.circle(ones, center, radius, 1, thickness=cv2.FILLED)
    retina = torch.from_numpy(circle)
    retina = retina.unsqueeze(0)

    return retina


def main():
    out_dir = "data/"

    opt = get_args()

    out_path = Path(out_dir) / "copypaste"
    label_path = out_path / "label"
    label_path.mkdir(parents=True, exist_ok=True)

    set_seed(opt.seed)
    nc = 8 + 1

    transform = T.Compose(
        [
            T.Resize(opt.img_size, interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ]
    )

    samples_per_grade = opt.n_samples // 5

    ticker = 0
    for dr_grade in range(5):
        dataset = CombinedDataset(
            return_image=False, return_transformed=False, label_transform=transform
        )
        dataset.df = dataset.df[dataset.df["Grade"] == dr_grade]
        dataloader = DataLoader(dataset, batch_size=nc, shuffle=True, drop_last=True)
        infinite_loader = infinite(dataloader)

        for _ in tqdm(range(samples_per_grade)):
            batch = next(infinite_loader)
            source_label = batch["label"]

            assert len(source_label) == nc

            source_label = get_label_semantics(source_label)

            _, _, height, width = source_label.shape
            retina = make_circle(height, width)

            od = source_label[[1], Labels.OD.value, :, :]
            ma = source_label[[2], Labels.MA.value, :, :]
            he = source_label[[3], Labels.HE.value, :, :]
            ex = source_label[[4], Labels.EX.value, :, :]
            se = source_label[[5], Labels.SE.value, :, :]
            nv = source_label[[6], Labels.NV.value, :, :]
            irma = source_label[[7], Labels.IRMA.value, :, :]

            retina = retina - od - ma - he - ex - se - nv - irma

            # Create background by subtracting everything else.
            bg = torch.ones_like(retina) - od - ma - he - ex - se - nv - irma - retina
            bg = torch.clamp(bg, 0, 1)

            combined = torch.cat([retina, od, ma, he, ex, se, nv, irma, bg], dim=0)
            new_label = torch.argmax(combined, dim=0).float().numpy()

            if opt.colour:
                new_label = colour_labels_numpy(new_label)
            else:
                # Set background to 255.
                new_label[new_label == (nc - 1)] = 255.0

            # TODO(sonjoonho): Generate grade more intelligently. This could be done just by
            # sampling from existing images with the desired grade.
            # End-point is inclusive.
            filename = f"copypaste_{dr_grade}_{ticker:05}.png"
            cv2.imwrite(str(label_path / filename), new_label)

            ticker += 1


if __name__ == "__main__":
    main()

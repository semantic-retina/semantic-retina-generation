from pathlib import Path

from PIL import Image
import torchvision.transforms.functional as F
import torch
from torchvision.utils import save_image


def main():
    generated_path = Path(
        "/vol/bitbucket/js6317/individual-project/SPADE/results/combined-od-hd/test_latest/images/"
    )
    labels_path = generated_path / "input_label"
    images_path = generated_path / "synthesized_image"

    labels_sample = []
    images_sample = []

    files = [
        "test_0_00002",
        "test_1_02122",
        "test_2_01222",
        "test_3_00042",
        "test_4_02471",
        "test_0_00012",
        "test_1_02126",
        "test_2_01224",
        "test_3_00044",
        "test_4_00795",
        "test_0_00020",
        "test_1_02195",
        "test_2_01488",
        "test_3_00047",
        "test_4_00798",
        "test_0_00023",
        "test_1_02211",
        "test_2_01271",
        "test_3_00061",
        "test_4_00801",
        "test_0_00033",
        "test_1_02258",
        "test_2_01318",
        "test_3_00068",
        "test_4_02720",
    ]

    for f in files:
        label = Image.open(labels_path / f"{f}.png")
        label = F.to_tensor(label)
        labels_sample.append(label)

        image = Image.open(images_path / f"{f}.png")
        image = F.to_tensor(image)
        images_sample.append(image)

    labels_batch = torch.stack(labels_sample)
    images_batch = torch.stack(images_sample)

    save_image(labels_batch, "labels_sample.png", nrow=5)
    save_image(images_batch, "images_sample.png", nrow=5)


if __name__ == "__main__":
    main()

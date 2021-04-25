from argparse import ArgumentParser
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.common import Labels, get_mask
from src.data.datasets.combined import CombinedDataset
from src.metrics.dice import compute_precision_recall_f1
from src.models.unet import UNet
from src.models.unet.transforms import make_transforms
from src.utils.device import get_device


def load_model(path: Path) -> nn.Module:
    model = UNet(n_channels=3, n_classes=2)
    model.load_state_dict(torch.load(path))
    model.eval()

    return model


def main():
    parser = ArgumentParser()
    parser.add_argument("name", type=str, default="unet-real-fgadr")
    opt = parser.parse_args()

    model_path = Path(f"results/unet/checkpoints/{opt.name}.pth")

    img_size = 512
    batch_size = 1

    device = get_device()

    model = load_model(model_path)
    model.to(device)

    image_transform, label_transform = make_transforms(img_size)

    dataset = CombinedDataset(
        image_transform=image_transform,
        label_transform=label_transform,
        train=False,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    total_dice = 0
    total_precision = 0
    total_recall = 0
    n_val = 0
    for batch in val_loader:
        images, masks_true = batch["image"], batch["label"]
        images = images.to(device=device, dtype=torch.float32)

        n_val += 1

        masks_true = get_mask(Labels.EX, masks_true)
        masks_true = masks_true.to(device=device, dtype=torch.long)

        with torch.no_grad():
            masks_pred = model(images)
        masks_pred = torch.argmax(masks_pred, dim=1)
        batch_precision, batch_recall, batch_f1 = compute_precision_recall_f1(
            masks_pred, masks_true
        )

        total_dice += batch_f1
        total_precision += batch_precision
        total_recall += batch_recall

    dice = total_dice / n_val
    precision = total_precision / n_val
    recall = total_recall / n_val
    print(opt.name)
    print(f"Dice: {dice}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")


if __name__ == "__main__":
    main()

from argparse import ArgumentParser
from pathlib import Path

from PIL import Image
from torch import nn
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from src.data.combined import CombinedDataset
from src.metrics.dice import dice_coefficient, accuracy
from src.models.unet import UNet
from src.data.common import get_label_semantics
from src.utils.device import get_device


def load_model(path: Path) -> nn.Module:
    model = UNet(n_channels=3, n_classes=2)
    model.load_state_dict(torch.load(path))
    model.eval()

    return model


def main():
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, default="unet-real-fgadr")
    opt = parser.parse_args()
    model_path = Path(f"results/unet/checkpoints/{opt.name}.pth")

    img_size = 256
    batch_size = 1
    device = get_device()

    model = load_model(model_path)
    model.to(device)

    image_transform = T.Compose([T.Resize(img_size), T.ToTensor()])
    label_transform = T.Compose([T.Resize(img_size, Image.NEAREST), T.ToTensor()])

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

    dice = 0
    acc = 0
    n_val = 0
    for batch in val_loader:
        images, masks_true = batch["image"], batch["label"]
        images = images.to(device=device, dtype=torch.float32)

        n_val += 1

        masks_true = get_label_semantics(masks_true)[:, 4, :, :]
        masks_true = masks_true.to(device=device, dtype=torch.long)

        with torch.no_grad():
            masks_pred = model(images)
        masks_pred = torch.argmax(masks_pred, dim=1)
        dice += dice_coefficient(masks_pred, masks_true)
        acc += accuracy(masks_pred, masks_true)

    dice /= n_val
    acc /= n_val
    print(opt.name)
    print(f"Dice: {dice}")
    print(f"Accuracy: {acc}")


if __name__ == "__main__":
    main()

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms as T

from src.data.datasets.eyepacs import EyePACS
from src.transforms.crop import CropShortEdge
from src.utils.device import get_device


def load_model(path: Path, n_classes: int) -> nn.Module:
    model = models.resnet18()
    fc_n_features = model.fc.in_features
    model.fc = nn.Linear(fc_n_features, n_classes)
    model.load_state_dict(torch.load(path))
    model.eval()

    return model


def main():
    # TODO(sonjoonho): Add argument parsing for options.

    name = "test"
    out_dir = "results/"
    img_size = 512
    batch_size = 32

    device = get_device()

    model_path = Path(out_dir) / "resnet" / name / "checkpoints" / "model_latest.pth"
    model = load_model(model_path, 5)
    model = model.to(device)

    transform = T.Compose([CropShortEdge(), T.Resize(img_size), T.ToTensor()])
    val_dataset = EyePACS(train=False, transform=transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=False,
    )

    n_val_samples = len(val_dataset)
    val_loss = 0.0
    val_corrects = 0.0
    for i, batch in enumerate(val_loader):
        images, grades = batch["image"], batch["grade"]
        images = images.to(device)
        grades = grades.to(device)

        with torch.no_grad():
            outputs = model(images)

        preds = torch.argmax(outputs, dim=1)

        val_corrects += torch.sum(torch.eq(preds, grades)).item()
        n_val_samples += len(preds)

    val_loss /= n_val_samples
    val_acc = val_corrects / n_val_samples
    print(val_acc)


if __name__ == "__main__":
    main()

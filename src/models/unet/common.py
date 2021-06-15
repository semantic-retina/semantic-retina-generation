import torch
from torch import nn
from torch.nn import DataParallel

from src.models.unet import UNet


def load_binary_segmentation_model(name: str) -> nn.Module:
    model = DataParallel(UNet(n_channels=3, n_classes=2))
    model.load_state_dict(
        torch.load(f"results/unet/{name}/checkpoints/model_latest.pth")
    )
    return model


def create_model(load_name: str, n_classes: int) -> nn.Module:
    # We take RGB images as input and predict the target class against the background.
    model = DataParallel(UNet(n_channels=3, n_classes=n_classes, bilinear=True))
    if load_name:
        print(f"Loading {load_name}")
        model.load_state_dict(
            torch.load(f"results/unet/{load_name}/checkpoints/model_latest.pth")
        )

    return model

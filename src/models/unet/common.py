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

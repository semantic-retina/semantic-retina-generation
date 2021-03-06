from pathlib import Path

import torch
from torch import nn
from torch.nn import DataParallel
from torchvision import models

from src.models.resnet import set_parameter_requires_grad


def create_retina_model(use_pretrained: bool, feature_extract: bool, n_classes: int):
    model = models.resnet101(pretrained=use_pretrained)
    set_parameter_requires_grad(model, feature_extract)

    # Reshape the final layer.
    fc_n_features = model.fc.in_features
    model.fc = nn.Linear(fc_n_features, n_classes)

    return DataParallel(model)


def create_small_retina_model(
    use_pretrained: bool,
    feature_extract: bool,
    n_classes: int,
    load_name: str = "",
):
    model = models.resnet50(pretrained=use_pretrained)
    set_parameter_requires_grad(model, feature_extract)

    # Reshape the final layer.
    fc_n_features = model.fc.in_features
    model.fc = nn.Linear(fc_n_features, n_classes)

    model = DataParallel(model)

    if load_name:
        print(f"Loading {load_name}")
        model.load_state_dict(
            torch.load(f"results/resnet/{load_name}/checkpoints/model_latest.pth")
        )

    return model


def load_retina_model(path: Path) -> nn.Module:
    """Loads the pre-trained ResNet model for predicting image grades."""
    model = create_retina_model(False, False, 5)
    model.load_state_dict(torch.load(path))
    model.eval()

    return model


def load_small_retina_model(path: Path) -> nn.Module:
    """Loads the pre-trained ResNet model for predicting image grades."""
    model = create_small_retina_model(False, False, 5)
    model.load_state_dict(torch.load(path))
    model.eval()

    return model

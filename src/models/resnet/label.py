from pathlib import Path

import torch
from torch import nn
from torchvision import models

from src.models.resnet import set_parameter_requires_grad


def create_label_model(use_pretrained: bool, feature_extract: bool):
    model = models.resnet50(pretrained=use_pretrained)
    set_parameter_requires_grad(model, feature_extract)

    # Reshape the final layer.
    fc_n_features = model.fc.in_features
    model.fc = nn.Linear(fc_n_features, 5)
    model.conv1 = nn.Conv2d(
        9, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )

    return model


def load_label_model(path: Path) -> nn.Module:
    """Loads the pre-trained ResNet model for predicting image grades."""
    model = create_label_model(False, False)
    model.load_state_dict(torch.load(path))
    model.eval()

    return model

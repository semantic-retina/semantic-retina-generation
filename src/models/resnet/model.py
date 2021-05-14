from torch import nn


def get_params_to_update(model: nn.Module, feature_extract: bool):
    params_to_update = model.parameters()

    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)

    return params_to_update


def set_parameter_requires_grad(model: nn.Module, feature_extract: bool):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

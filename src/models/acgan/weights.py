import torch
from torch import nn
from torch.nn import init


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(module):
    if (
        isinstance(module, nn.Conv2d)
        or isinstance(module, nn.ConvTranspose2d)
        or isinstance(module, nn.Linear)
    ):
        init.orthogonal_(module.weight)
        if module.bias is not None:
            module.bias.data.fill_(0.0)

    elif isinstance(module, nn.Embedding):
        init.orthogonal_(module.weight)

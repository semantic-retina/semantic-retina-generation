import random
from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import functional as TF


class ProbabilisticTransform(ABC):
    @abstractmethod
    def update_p(self, p: float):
        pass

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        pass


class Rotate(ProbabilisticTransform):
    """Applies up to 3 90 degree rotations probability p."""

    def __init__(self, p):
        assert 0.0 <= p <= 1.0
        self.p = p

    def update_p(self, p: float):
        self.p = p

    def __call__(self, x: Tensor) -> Tensor:
        if random.random() > self.p:
            return x

        num_rotations = random.randrange(0, 3 + 1)
        x = TF.rotate(x, 90 * num_rotations, fill=0)
        return x


class GaussianNoise(ProbabilisticTransform):
    def __init__(self, p: float, mean: float, max_std: float):
        assert 0.0 <= p <= 1.0
        self.mean = mean
        self.max_std = max_std
        self.p = p

    def update_p(self, p: float):
        self.p = p

    def __call__(self, x: Tensor) -> Tensor:
        if random.random() > self.p:
            return x

        std = self.p * self.max_std

        noise = self.mean + torch.randn(x.size()).cuda() * std
        return x + noise


class Affine(ProbabilisticTransform):
    def __init__(self, p: float, n_channels: int):
        assert 0.0 <= p <= 1.0
        self.p = p
        translate = (0.5, 0.5)
        scale = (0.8, 1.2)
        # Set the area outside the transform as background.
        fill = [1] + [0 for _ in range(n_channels - 1)]
        self.transform = transforms.RandomAffine(
            360, translate, scale, fill=fill, fillcolor=None
        )

    def update_p(self, p: float):
        self.p = p

    def __call__(self, x: Tensor) -> Tensor:
        if random.random() > self.p:
            return x

        x = self.transform(x)
        return x

import random

import torch
from torch import Tensor

from src.transforms.probabilistic_transform import ProbabilisticTransform


class GaussianNoise(ProbabilisticTransform):
    def __init__(self, p, mean, std):
        assert 0.0 <= p <= 1.0
        self.mean = mean
        self.std = std
        self.p = p

    def update_p(self, p: float):
        self.p = p

    def __call__(self, x: Tensor) -> Tensor:
        if random.random() > self.p:
            return x

        noise = self.mean + torch.randn(x.size()).cuda() * self.std
        return x + noise

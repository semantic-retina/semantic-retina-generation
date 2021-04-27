from typing import List

import torch
from torch import Tensor

from src.transforms.probabilistic_transform import ProbabilisticTransform


class DiscriminatorTransform:
    def __init__(self, target: float, transforms: List[ProbabilisticTransform]):
        self.transforms = transforms
        self.target = target
        self.p = 0.0

    def __call__(self, images: Tensor) -> Tensor:
        batch_size, _, _, _ = images.shape
        transformed_images = torch.empty_like(images)
        for i in range(batch_size):
            img = images[i]
            for t in self.transforms:
                img = t(img)
            transformed_images[i] = img
        return transformed_images

    def update(self, ada_r: float):

        if ada_r > self.target:
            self.p += 0.01
        if ada_r < self.target:
            self.p -= 0.01

        # Clamp p to [0, 1].
        if self.p < 0:
            self.p = 0
        if self.p > 1:
            self.p = 1

        for t in self.transforms:
            t.update_p(self.p)

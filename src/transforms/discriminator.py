from typing import List

import torch
from torch import Tensor

from src.transforms.probabilistic import ProbabilisticTransform


class DiscriminatorTransform:
    def __init__(
        self,
        target: float,
        transforms: List[ProbabilisticTransform],
        max_p: float = 1.0,
    ):
        assert 0.0 <= target <= 1.0
        assert 0.0 <= max_p <= 1.0

        self.transforms = transforms
        self.target = target
        self.max_p = max_p
        self.p = 0.0

    def __call__(self, images: Tensor) -> Tensor:
        if not self.transforms:
            return images

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

        # Clamp p to [0, self.max_p].
        self.p = max(0, self.p)
        self.p = min(self.max_p, self.p)

        for t in self.transforms:
            t.update_p(self.p)

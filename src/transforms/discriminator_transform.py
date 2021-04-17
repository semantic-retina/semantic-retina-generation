from typing import List

from torch import Tensor

from src.transforms.probabilistic_transform import ProbabilisticTransform


class DiscriminatorTransform(ProbabilisticTransform):
    def __init__(self, transforms: List[ProbabilisticTransform]):
        self.transforms = transforms

    def __call__(self, img: Tensor) -> Tensor:
        for t in self.transforms:
            img = t(img)
        return img

    def update_p(self, p: float):
        for t in self.transforms:
            t.update_p(p)

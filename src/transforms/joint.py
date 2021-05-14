from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
import torchvision.transforms.functional as TF
from torch import Tensor


class JointTransform(ABC):
    @abstractmethod
    def __call__(self, tensors: List[Tensor]) -> List[Tensor]:
        pass


class Compose:
    def __init__(self, transforms: List[JointTransform]):
        self.transforms = transforms

    def __call__(self, tensors: List[Tensor]) -> List[Tensor]:
        for t in self.transforms:
            tensors = t(tensors)
        return tensors


class RandomHorizontalFlip(JointTransform):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, tensors: List[Tensor]) -> List[Tensor]:
        if torch.rand(1) < self.p:
            return [TF.hflip(x) for x in tensors]
        return tensors


class RandomVerticalFlip(JointTransform):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, tensors: List[Tensor]) -> List[Tensor]:
        if torch.rand(1) < self.p:
            return [TF.vflip(x) for x in tensors]
        return tensors

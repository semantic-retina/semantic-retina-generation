from abc import ABC, abstractmethod

from torch import Tensor


class ProbabilisticTransform(ABC):
    @abstractmethod
    def update_p(self, p: float):
        pass

    def __call__(self, x: Tensor) -> Tensor:
        pass

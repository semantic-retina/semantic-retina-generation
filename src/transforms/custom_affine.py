import random

from torchvision import transforms

from src.transforms.probabilistic_transform import ProbabilisticTransform


class Affine(ProbabilisticTransform):
    def __init__(self, p):
        assert 0.0 <= p <= 1.0
        self.p = p
        self.transform = transforms.RandomAffine(
            90, (0.25, 0.25), (0.25, 0.25), fillcolor=0
        )

    def update_p(self, p: float):
        self.p = p

    def __call__(self, x):
        if random.random() > self.p:
            return x

        x = self.transform(x)
        return x

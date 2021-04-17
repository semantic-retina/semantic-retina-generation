import random

import torchvision.transforms.functional as TF

from src.transforms.probabilistic_transform import ProbabilisticTransform


class Rotate(ProbabilisticTransform):
    """Applies up to 3 90 degree rotations probability p."""

    def __init__(self, p):
        assert 0.0 <= p <= 1.0
        self.p = p

    def update_p(self, p: float):
        self.p = p

    def __call__(self, x):
        if random.random() > self.p:
            return x

        num_rotations = random.randrange(0, 3)
        x = TF.rotate(x, 90 * num_rotations, fill=0)
        return x

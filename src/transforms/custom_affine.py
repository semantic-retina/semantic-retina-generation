import random

import torchvision.transforms as T
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms
from torchvision.utils import save_image

from src.data.datasets.combined import CombinedDataset
from src.transforms.probabilistic_transform import ProbabilisticTransform


class Affine(ProbabilisticTransform):
    def __init__(self, p: float):
        assert 0.0 <= p <= 1.0
        self.p = p
        translate = (0.25, 0.25)
        scale = (0.8, 1.2)
        self.transform = transforms.RandomAffine(
            90, translate, scale, fill=1, fillcolor=None
        )

    def update_p(self, p: float):
        self.p = p

    def __call__(self, x: Tensor) -> Tensor:
        if random.random() > self.p:
            return x

        x = self.transform(x)
        return x


def test():
    transform = T.Compose([T.ToTensor(), Affine(0.5)])
    batch_size = 16
    n_batches = 2
    dataset = CombinedDataset(common_transform=transform)
    sampler = RandomSampler(
        dataset, num_samples=batch_size * n_batches, replacement=True
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    for i, batch in enumerate(dataloader):
        images, labels = batch["image"], batch["label"]
        save_image(images, f"custom_rotate_test_image_{i}.png")
        save_image(labels, f"custom_rotate_test_label_{i}.png")


if __name__ == "__main__":
    test()

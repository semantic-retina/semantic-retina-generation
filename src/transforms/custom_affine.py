import random

import torchvision.transforms as T
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms
from torchvision.utils import save_image

from src.data.common import get_label_semantics
from src.data.datasets.combined import CombinedDataset
from src.transforms.probabilistic_transform import ProbabilisticTransform
from src.utils.sample import colour_labels


class Affine(ProbabilisticTransform):
    def __init__(self, p: float, n_channels: int):
        assert 0.0 <= p <= 1.0
        self.p = p
        translate = (0.5, 0.5)
        scale = (0.8, 1.2)
        # Set the area outside the transform as background.
        fill = [1] + [0 for _ in range(n_channels - 1)]
        self.transform = transforms.RandomAffine(
            90, translate, scale, fill=fill, fillcolor=None
        )

    def update_p(self, p: float):
        self.p = p

    def __call__(self, x: Tensor) -> Tensor:
        if random.random() > self.p:
            return x

        x = self.transform(x)
        return x


def test():
    transform = T.Compose([T.ToTensor(), T.Resize(128, T.InterpolationMode.NEAREST)])
    batch_size = 16
    n_batches = 2
    dataset = CombinedDataset(common_transform=transform)
    sampler = RandomSampler(
        dataset, num_samples=batch_size * n_batches, replacement=True
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    for i, batch in enumerate(dataloader):
        labels = batch["label"]
        labels = get_label_semantics(labels)
        coloured_labels = colour_labels(labels)
        print(coloured_labels.shape)
        save_image(coloured_labels, f"custom_rotate_test_label_{i}.png")


if __name__ == "__main__":
    test()

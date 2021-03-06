import random
from pathlib import Path
from typing import Dict, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.utils import save_image

from src.data.common import get_label_semantics


class SyntheticDataset(Dataset):
    """Dataset that returns synthetic data."""

    def __init__(
        self,
        name: str = "",
        image_transform: T.Compose = None,
        label_transform: T.Compose = None,
        common_transform: T.Compose = None,
        return_image: bool = True,
        return_transformed: bool = True,
        return_label: bool = True,
        return_inst: bool = True,
        return_grade: bool = True,
        n_samples: int = -1,
    ):

        # TODO: Remove hardcoded paths.
        image_dir = f"/vol/bitbucket/js6317/individual-project/semantic-dr-gan/results/{name}/test/img/"
        transformed_dir = f"/vol/bitbucket/js6317/individual-project/semantic-dr-gan/results/{name}/test/transformed/"
        label_dir = f"/vol/bitbucket/js6317/individual-project/semantic-dr-gan/results/{name}/test/label/"
        inst_dir = f"/vol/bitbucket/js6317/individual-project/semantic-dr-gan/results/{name}/test/inst/"

        self.image_path = Path(image_dir)
        self.transformed_path = Path(transformed_dir)
        self.label_path = Path(label_dir)
        self.inst_path = Path(inst_dir)
        self.files = []
        self.grades = []

        all_files = list(self.label_path.glob("**/*"))

        if len(all_files) == 0:
            raise FileNotFoundError("dataset not found")

        if n_samples == -1:
            subset_files = all_files
        else:
            subset_files = random.sample(all_files, n_samples)

        for f in subset_files:
            self.files.append(f.name)
            self.grades.append(int(f.stem[5]))

        self.image_transform = image_transform
        self.label_transform = label_transform
        self.common_transform = common_transform

        self.return_image = return_image
        self.return_transformed = return_transformed
        self.return_label = return_label
        self.return_inst = return_inst
        self.return_grade = return_grade

    def __len__(self):
        return len(self.files)

    def get_image(self, name: str) -> Image:
        image = Image.open(self.image_path / name)
        return image

    def get_transformed(self, name: str) -> Image:
        image = Image.open(self.transformed_path / name)
        return image

    def get_label(self, name: str) -> Image:
        label = Image.open(self.label_path / name)
        return label

    def get_inst(self, name: str) -> Image:
        inst = Image.open(self.inst_path / name)
        return inst

    def __getitem__(self, item: int) -> Dict:
        filename = self.files[item]
        grade = self.grades[item]

        sample = {}

        if self.return_image:
            image = self.get_image(filename)
            if self.image_transform is not None:
                image = self.image_transform(image)
            if self.common_transform is not None:
                image = self.common_transform(image)
            sample["image"] = image

        if self.return_transformed:
            image = self.get_transformed(filename)
            if self.image_transform is not None:
                image = self.image_transform(image)
            if self.common_transform is not None:
                image = self.common_transform(image)
            sample["transformed"] = image

        if self.return_label:
            label = self.get_label(filename)
            if self.label_transform is not None:
                label = self.label_transform(label)
            if self.common_transform is not None:
                label = self.common_transform(label)
            sample["label"] = label

        if self.return_inst:
            inst = self.get_inst(filename)
            if self.label_transform is not None:
                inst = self.label_transform(inst)
            if self.common_transform is not None:
                inst = self.common_transform(inst)
            sample["inst"] = inst

        if self.return_grade:
            sample["grade"] = grade

        return sample


def test():
    image_transform = T.Compose(
        [
            T.Resize(256),
            T.ToTensor(),
        ],
    )
    mask_transform = T.Compose(
        [
            T.Resize(256, interpolation=Image.NEAREST),
            T.ToTensor(),
        ],
    )
    ds = SyntheticDataset(
        label_transform=mask_transform, image_transform=image_transform
    )

    dataloader = DataLoader(ds, batch_size=10)

    batch = next(iter(dataloader))
    image, label, inst, grade = (
        batch["image"],
        batch["label"],
        batch["inst"],
        batch["grade"],
    )
    label = label.unsqueeze(0)

    print(f"Label shape: {label.shape}")

    label_map = get_label_semantics(label)

    print(f"Label map shape: {label_map.shape}")
    nc = 7
    for i in range(nc):
        save_image(label_map[:, [i], :, :].float(), f"test_{i}.png")
    save_image(image, f"test_real.png")
    label[label == 2] = 0.5
    save_image(label, "label.png")

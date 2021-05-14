import random
from pathlib import Path
from typing import Dict

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class CopyPasteDataset(Dataset):
    """Dataset that returns data generated by sampling lesions from existing data."""

    label_dir = (
        "/vol/bitbucket/js6317/individual-project/semantic-dr-gan/data/copypaste/label/"
    )

    def __init__(
        self,
        label_transform: T.Compose = None,
        return_label: bool = True,
        return_grade: bool = True,
        n_samples: int = -1,
    ):
        self.label_path = Path(CopyPasteDataset.label_dir)
        self.files = []

        self.grades = []

        all_files = list(self.label_path.glob("**/*"))
        if n_samples == -1:
            subset_files = all_files
        else:
            subset_files = random.sample(all_files, n_samples)

        for f in subset_files:
            self.files.append(f.name)
            # Filename format should be copypaste_G_XXXXX.png where G is the DR grade.
            self.grades.append(int(f.stem[10]))

        self.label_transform = label_transform

        self.return_label = return_label
        self.return_grade = return_grade

    def __len__(self):
        return len(self.files)

    def get_label(self, name: str) -> Image:
        return Image.open(self.label_path / name).convert("L")

    def __getitem__(self, item: int) -> Dict:
        filename = self.files[item]
        grade = self.grades[item]

        sample = {}

        if self.return_label:
            label = self.get_label(filename)
            if self.label_transform is not None:
                label = self.label_transform(label)
            sample["label"] = label

        if self.return_grade:
            sample["grade"] = grade

        return sample
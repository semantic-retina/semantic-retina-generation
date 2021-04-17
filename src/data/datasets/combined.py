from pathlib import Path
from typing import Dict

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class CombinedDataset(Dataset):
    """
    Dataset that returns images and labels specified by the CSVs generated by the
    `split_dataset.py` script.
    """

    root_dir = "/vol/bitbucket/js6317/individual-project/semantic-dr-gan/data/"

    def __init__(
        self,
        image_transform: T.Compose = None,
        label_transform: T.Compose = None,
        common_transform: T.Compose = None,
        return_image: bool = True,
        return_label: bool = True,
        return_inst: bool = True,
        return_grade: bool = True,
        train: bool = True,
    ):
        root_path = Path(CombinedDataset.root_dir)

        if train:
            csv_path = root_path / "train.csv"
        else:
            csv_path = root_path / "test.csv"

        self.df = pd.read_csv(csv_path)

        self.image_transform = image_transform
        self.label_transform = label_transform
        self.common_transform = common_transform

        self.return_image = return_image
        self.return_label = return_label
        self.return_inst = return_inst
        self.return_grade = return_grade

    def get_image(self, path: str) -> Image:
        return Image.open(path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item: int) -> Dict:
        row = self.df.iloc[item]

        sample = {}

        if self.return_image:
            image = self.get_image(row["Image"])
            if self.image_transform is not None:
                image = self.image_transform(image)
            if self.common_transform is not None:
                image = self.common_transform(image)
            sample["image"] = image

        if self.return_label:
            label = self.get_image(row["Label"])
            if self.label_transform is not None:
                label = self.label_transform(label)
            if self.common_transform is not None:
                label = self.common_transform(label)
            sample["label"] = label

        if self.return_inst:
            inst = self.get_image(row["Instance"])
            if self.label_transform is not None:
                inst = self.label_transform(inst)
            if self.common_transform is not None:
                inst = self.common_transform(inst)
            sample["inst"] = inst

        if self.return_grade:
            grade = int(row["Grade"])
            sample["grade"] = grade

        return sample
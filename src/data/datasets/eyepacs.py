from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.utils import save_image


class HDF5EyePACS(Dataset):
    """
    Loads EyePACS images from a HDF5 file created with the `create_eyepacs.py` script.
    Currently unused.
    """

    def __init__(self, train: bool = True, transform: T.Compose = None):
        file_path = Path("/data/js6317") / "hdf5" / "eyepacs.hdf5"
        f = h5py.File(file_path, "r")

        if train:
            self.images = f["train"]["images"]
            self.grades = f["train"]["labels"]
        else:
            self.images = f["test"]["images"]
            self.grades = f["test"]["labels"]

        self.transform = transform

    def __len__(self):
        return len(self.grades)

    def __getitem__(self, item: int):
        image = self.images[item]
        grade = self.grades[item]

        image = torch.from_numpy(image).float()

        if self.transform is not None:
            image = self.transform(image)

        grade = int(grade)

        return {"transformed": image, "grade": grade}


class EyePACS(Dataset):
    """
    Dataset containing images and labels from the Kaggle EyePACS dataset.
    Note that image sizes vary, so a short-edge crop may be required.
    """

    root_dir = "/vol/bitbucket/js6317/individual-project/semantic-dr-gan/data/eyepacs/"

    def __init__(
        self,
        train: bool = True,
        transform: T.Compose = None,
        indices: np.ndarray = None,
    ):
        self.transform = transform

        # TODO(sonjoonho): Remove hardcoded path.
        data_path = Path(EyePACS.root_dir)

        # CSV columns are: "PatientId, name, eye, level, level_binary,
        # level_hot, path, path_preprocess, exists".
        if train:
            df_path = data_path / "train.csv"
        else:
            df_path = data_path / "test.csv"

        self.df = pd.read_csv(df_path)

        if indices is not None:
            self.df = self.df.iloc[indices]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, item: int) -> Tuple[Tensor, int]:
        row = self.df.iloc[item]
        path, grade = str(row["path"]), int(row["level"])

        image = Image.open(path)

        if self.transform is not None:
            image = self.transform(image)

        return {"image": image, "grade": grade}


def test():
    img_size = 256
    transform = T.Compose([T.Resize(img_size), T.ToTensor()])
    dataset = HDF5EyePACS(transform=transform)
    dataloader = DataLoader(dataset, batch_size=12)
    batch = next(iter(dataloader))
    images = batch["image"] / 255.0
    save_image(images, "eyepacs.png")


if __name__ == "__main__":
    test()

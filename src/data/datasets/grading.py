from pathlib import Path
from typing import Dict

import pandas as pd
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class GradingDataset(Dataset):
    root_dir = "/vol/bitbucket/js6317/individual-project/semantic-dr-gan/data/grade/"

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    ALL = "all"

    def __init__(
        self,
        image_transform: T.Compose = None,
        mode: str = TRAIN,
    ):
        root_path = Path(GradingDataset.root_dir)

        if mode == GradingDataset.TRAIN:
            csv_path = root_path / "train.csv"
        elif mode == GradingDataset.VALIDATION:
            csv_path = root_path / "val.csv"
        elif mode == GradingDataset.TEST:
            csv_path = root_path / "test.csv"
        elif mode == GradingDataset.ALL:
            csv_path = root_path / "all.csv"
        else:
            raise ValueError(f'Invalid mode "{mode}"')
        self.df = pd.read_csv(csv_path, index_col=0)

        self.image_transform = image_transform

    def get_image(self, path: str) -> Image:
        return Image.open(path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item: int) -> Dict:
        row = self.df.iloc[item]

        grade = int(row["Grade"])
        transformed = self.get_image(row["Transformed"])
        if self.image_transform is not None:
            transformed = self.image_transform(transformed)

        return {"grade": grade, "transformed": transformed}

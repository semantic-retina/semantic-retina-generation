"""
This script combines the FGADR and IDRiD datasets and splits them into train/test
subsets. CSV files containing file paths are saved to `data/train.csv` and
`data/test.csv`. Ensure that you have run the pre-processing scripts first.
"""

from pathlib import Path
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torchvision import models
import torchvision.transforms as T
import numpy as np

from src.data.transform import CropShortEdge


def load_model(path: Path) -> nn.Module:
    model = models.resnet18()
    fc_n_features = model.fc.in_features
    model.fc = nn.Linear(fc_n_features, 5)
    model.load_state_dict(torch.load(path))
    model.eval()

    return model


def make_fgadr(original_path: str, label_path: str, inst_path: str):
    fgadr_original_path = Path(original_path)
    fgadr_image_path = fgadr_original_path / "Original_Images"
    fgadr_label_path = Path(label_path)
    fgadr_inst_path = Path(inst_path)
    fgadr_csv_path = fgadr_original_path / "DR_Seg_Grading_Label.csv"

    fgadr_df = pd.read_csv(fgadr_csv_path, header=None, names=["File", "Grade"])

    exclude_grader_1 = True
    if exclude_grader_1:
        grader = pd.to_numeric(fgadr_df["File"].str[5])
        not_grader_1_cond = grader != 1
        fgadr_df = fgadr_df.loc[not_grader_1_cond]

    fgadr_df = make_absolute_paths(
        fgadr_df, fgadr_image_path, fgadr_label_path, fgadr_inst_path
    )

    return fgadr_df


def predict(model: nn.Module, idrid_df: pd.DataFrame):
    """Predicts labels for the IDRiD dataset using the specified model."""
    img_size = 512
    transform = T.Compose([CropShortEdge(), T.Resize(img_size), T.ToTensor()])

    predictions = np.empty(len(idrid_df), dtype=int)
    for i, row in idrid_df.iterrows():
        image = Image.open(row["Image"])
        image = transform(image).unsqueeze(0)
        pred = model(image)
        pred = torch.argmax(pred)
        predictions[i] = pred.item()

    return predictions


def make_idrid(root_path: str, predict_grades: bool = False):
    idrid_root_path = Path(root_path)

    idrid_image_path = idrid_root_path / "img"
    idrid_label_path = idrid_root_path / "label"
    idrid_inst_path = idrid_root_path / "inst"

    idrid_files = [f.name for f in idrid_image_path.glob("**/*")]
    idrid_files.sort()

    idrid_df = pd.DataFrame(idrid_files, columns=["File"])

    idrid_df = make_absolute_paths(
        idrid_df, idrid_image_path, idrid_label_path, idrid_inst_path
    )

    # Predict labels.
    if predict_grades:
        model_path = Path("results/resnet/resnet-baseline.pth")
        model = load_model(model_path)
        noisy_grades = predict(model, idrid_df)
    else:
        noisy_grades = np.random.randint(0, 5)
    idrid_df["Grade"] = noisy_grades

    return idrid_df


def make_absolute_paths(
    df: pd.DataFrame, image_path: Path, label_path: Path, inst_path: Path
):
    """Converts filenames to absolute paths using the specified root paths."""
    df["Image"] = str(image_path) + "/" + df["File"].astype(str)
    df["Label"] = str(label_path) + "/" + df["File"].astype(str)
    df["Instance"] = str(inst_path) + "/" + df["File"].astype(str)

    return df


def main():
    train_size = 0.8
    seed = 10
    predict_grades = False

    data_path = Path("data")
    data_path.mkdir(parents=True, exist_ok=True)

    fgadr_original_dir = "/vol/vipdata/data/retina/FGADR-Seg/Seg-set/"
    fgadr_label_dir = (
        "/vol/bitbucket/js6317/individual-project/SPADE/datasets/fgadr/label/"
    )
    fgadr_inst_dir = (
        "/vol/bitbucket/js6317/individual-project/SPADE/datasets/fgadr/inst/"
    )
    fgadr_df = make_fgadr(fgadr_original_dir, fgadr_label_dir, fgadr_inst_dir)

    root_path = "/vol/bitbucket/js6317/individual-project/SPADE/datasets/idrid/"
    idrid_df = make_idrid(root_path, predict_grades=predict_grades)

    combined_df = pd.concat((fgadr_df, idrid_df))
    combined_train, combined_test = train_test_split(
        combined_df, train_size=train_size, random_state=seed
    )

    print(f"FGADR : {len(fgadr_df)}")
    print(f"IDRiD: {len(idrid_df)}")

    print(f"Train: {len(combined_train)}")
    print(f"Test: {len(combined_test)}")

    combined_train.to_csv(data_path / "train.csv")
    combined_test.to_csv(data_path / "test.csv")


if __name__ == "__main__":
    main()

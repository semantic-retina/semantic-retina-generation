from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from src.data.datasets.eyepacs import HDF5EyePACS
from src.models.resnet.retina import load_retina_model
from src.utils.device import get_device


def main():
    # TODO(sonjoonho): Add argument parsing for options.

    name = "eyepacs_transformed"
    out_dir = "results/"
    img_size = 512
    batch_size = 64

    device = get_device()

    model_path = Path(out_dir) / "resnet" / name / "checkpoints" / "model_latest.pth"
    model = load_retina_model(model_path)
    model = model.to(device)

    transform = T.Compose([T.Resize(img_size)])
    val_dataset = HDF5EyePACS(train=False, transform=transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=False,
    )

    tta_runs = 5
    tta_transform = T.RandomRotation(360)
    n_val_samples = len(val_dataset)
    predictions = np.empty(n_val_samples, dtype=int)
    actual = np.empty(n_val_samples, dtype=int)
    for i, batch in enumerate(tqdm(val_loader)):
        images, grades = batch["image"], batch["grade"]
        images = images.to(device)
        grades = grades.to(device)

        tta_preds = torch.empty((tta_runs, images.shape[0], 5), dtype=float).to(device)
        for run in range(tta_runs):
            images = tta_transform(images)
            with torch.no_grad():
                outputs = model(images)
            tta_preds[run, :, :] = outputs
        tta_preds = torch.mean(tta_preds, dim=0)
        preds = torch.argmax(tta_preds, dim=1)

        predictions[
            i * batch_size : i * batch_size + images.shape[0]
        ] = preds.cpu().numpy()
        actual[i * batch_size : i * batch_size + images.shape[0]] = grades.cpu().numpy()

    print("Accuracy: ", accuracy_score(actual, predictions))
    print("Precision: ", precision_score(actual, predictions))
    print("Recall: ", recall_score(actual, predictions))
    print("F1: ", f1_score(actual, predictions))


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from src.data.datasets.eyepacs import HDF5EyePACS
from src.logger.common import timestamp
from src.models.resnet.retina import load_retina_model
from src.options.resnet.test import get_args
from src.utils.device import get_device


def quadratic_kappa(actual, predicted):
    return cohen_kappa_score(predicted, actual, weights="quadratic")


def main():
    # TODO(sonjoonho): Add argument parsing for options.

    name = "eyepacs_transformed"
    out_dir = "results/"
    img_size = 512
    batch_size = 64

    opt = get_args()

    output_path = Path(opt.out_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = get_device()

    model_path = Path(out_dir) / "resnet" / name / "checkpoints" / "model_latest.pth"
    model = load_retina_model(model_path)
    model = model.to(device)

    val_dataset = HDF5EyePACS(train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=False,
    )

    tta_transform = T.RandomAffine(degrees=360, translate=(0.1, 0.1))
    n_val_samples = len(val_dataset)
    predictions = np.empty(n_val_samples, dtype=int)
    actual = np.empty(n_val_samples, dtype=int)
    for i, batch in enumerate(tqdm(val_loader)):
        images, grades = batch["image"], batch["grade"]
        images = images.to(device)
        grades = grades.to(device)

        if opt.tta:
            tta_preds = torch.empty((opt.tta_runs, images.shape[0], 5), dtype=float).to(
                device
            )
            for run in range(opt.tta_runs):
                images = tta_transform(images)
                with torch.no_grad():
                    outputs = model(images)
                tta_preds[run, :, :] = outputs
            tta_preds = torch.mean(tta_preds, dim=0)
            preds = torch.argmax(tta_preds, dim=1)
        else:
            with torch.no_grad():
                outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

        predictions[
            i * batch_size : i * batch_size + images.shape[0]
        ] = preds.cpu().numpy()
        actual[i * batch_size : i * batch_size + images.shape[0]] = grades.cpu().numpy()

    metrics = {
        "accuracy": accuracy_score(actual, predictions),
        "precision": precision_score(actual, predictions, average="macro"),
        "recall": recall_score(actual, predictions, average="macro"),
        "f1": f1_score(actual, predictions, average="macro"),
        "kappa": quadratic_kappa(actual, predictions),
        "tta": opt.tta,
        "tta_runs": opt.tta_runs,
    }
    print("Accuracy: ", metrics["accuracy"])
    print("Precision: ", metrics["precision"])
    print("Recall: ", metrics["recall"])
    print("F1: ", metrics["f1"])
    print("Cohen's", metrics["kappa"])

    time = timestamp()

    # Save options.
    with open(output_path / f"metrics-{time}.json", "w") as f:
        json.dump(vars(opt), f, indent=4)


if __name__ == "__main__":
    main()

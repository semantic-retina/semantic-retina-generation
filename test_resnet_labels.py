import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from src.data.common import get_label_semantics
from src.data.datasets.combined import CombinedDataset
from src.data.datasets.copy_paste import CopyPasteDataset
from src.data.datasets.synthetic import SyntheticDataset
from src.logger.common import timestamp
from src.metrics.kappa import quadratic_kappa
from src.models.resnet.label import load_label_model
from src.options.resnet.test import get_args
from src.utils.device import get_device


def main():
    # TODO(sonjoonho): Add argument parsing for options.

    out_dir = "results/"
    img_size = 512
    batch_size = 64

    opt = get_args()
    name = opt.name

    output_path = Path(opt.out_dir) / name
    output_path.mkdir(parents=True, exist_ok=True)

    device = get_device()

    model_path = (
        Path(out_dir) / "resnet_labels" / name / "checkpoints" / "model_latest.pth"
    )
    model = load_label_model(model_path)
    model = model.to(device)

    transform = T.Compose(
        [
            T.Resize(img_size, interpolation=InterpolationMode.NEAREST),
            T.ToTensor(),
        ]
    )

    if opt.dataset == "real":
        test_dataset = CombinedDataset(
            label_transform=transform,
            return_image=False,
            return_inst=False,
            return_transformed=False,
            mode=CombinedDataset.VALIDATION,
        )
        test_dataset.df = test_dataset.df[test_dataset.df["Source"] == "FGADR"]
    elif opt.dataset == "copypaste":
        test_dataset = CopyPasteDataset(
            label_transform=transform,
            return_transformed=False,
        )
    else:
        test_dataset = SyntheticDataset(
            opt.dataset,
            label_transform=transform,
            return_image=False,
            return_inst=False,
            return_transformed=False,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=False,
    )

    n_val_samples = len(test_dataset)
    predictions = np.empty(n_val_samples, dtype=int)
    actual = np.empty(n_val_samples, dtype=int)

    print(f"Validation samples: {n_val_samples}")

    for i, batch in enumerate(tqdm(test_loader)):
        images, grades = batch["label"], batch["grade"]
        images = get_label_semantics(images)

        images = images.to(device)
        grades = grades.to(device)

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

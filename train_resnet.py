"""Trains ResNet on DR data."""
import json
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms as T

from src.data.datasets.copy_paste import CopyPasteDataset
from src.data.datasets.grading import GradingDataset
from src.data.datasets.synthetic import SyntheticDataset
from src.logger.resnet import ResNetLogger, ResNetTrainMetrics, ResNetValidateMetrics
from src.models.resnet import get_params_to_update
from src.models.resnet.retina import create_small_retina_model
from src.options.resnet.train import get_args
from src.utils.device import get_device
from src.utils.seed import set_seed


def train_step(
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    log_interval: int,
    val_interval: int,
    logger: ResNetLogger,
):
    n_batches = len(train_loader)

    for batch_idx, batch in enumerate(train_loader):
        model.train()

        iteration = epoch * n_batches + batch_idx

        images, grades = batch["transformed"], batch["grade"]

        images = images.to(device)
        grades = grades.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, grades)
        loss.backward()
        optimizer.step()

        if iteration % log_interval == 0:
            preds = torch.argmax(outputs, dim=1)
            train_loss = loss.item()
            train_acc = torch.sum(torch.eq(preds, grades)).item() / len(grades)

            metrics = ResNetTrainMetrics(iteration, epoch, train_loss, train_acc)
            logger.log_train(metrics)

        if iteration % val_interval == 0:
            validate(iteration, model, criterion, val_loader, device, logger)


def validate(
    iteration: int,
    model: nn.Module,
    criterion: nn.CrossEntropyLoss,
    val_loader: DataLoader,
    device: torch.device,
    logger: ResNetLogger,
):
    if val_loader is None:
        return

    model.eval()
    n_val_samples = 0
    val_loss = 0.0
    val_corrects = 0.0

    for batch in val_loader:
        images, grades = batch["transformed"], batch["grade"]

        images = images.to(device)
        grades = grades.to(device)

        with torch.no_grad():
            outputs = model(images)

        loss = criterion(outputs, grades)

        preds = torch.argmax(outputs, dim=1)

        val_loss += loss.item()
        val_corrects += torch.sum(torch.eq(preds, grades)).item()
        n_val_samples += len(preds)

    val_loss /= n_val_samples
    val_acc = val_corrects / n_val_samples

    metrics = ResNetValidateMetrics(iteration, val_loss, val_acc)
    logger.log_val(metrics)


def train(
    model: nn.Module,
    num_epochs: int,
    log_interval: int,
    val_interval: int,
    batch_size: int,
    img_size: int,
    lr: float,
    logger: ResNetLogger,
    device: torch.device,
    feature_extract: bool,
    use_hdf5: bool,
    synthetic_name: str,
    n_synthetic: int,
    use_real: bool,
) -> nn.Module:

    # if use_hdf5:
    #     EyePACSDataset = HDF5EyePACS
    #     transform = T.Compose([T.RandomAffine(360, translate=(0.1, 0.1), shear=0.2)])
    # else:
    #     EyePACSDataset = EyePACS
    #     transform = T.Compose(
    #         [
    #             CropShortEdge(),
    #             T.Resize(512),
    #             T.RandomAffine(360, translate=(0.1, 0.1), shear=0.2),
    #             T.ToTensor(),
    #         ]
    #     )
    #
    # if use_real:
    #     real_dataset = EyePACSDataset(train=True, transform=transform)
    # else:
    #     real_dataset = None
    transform = T.Compose([T.Resize(512), T.ToTensor()])
    real_dataset = GradingDataset(image_transform=transform, mode=GradingDataset.TRAIN)

    if synthetic_name:
        transform = T.Compose(
            [T.RandomAffine(360, translate=(0.1, 0.1), shear=0.2), T.ToTensor()]
        )
        if synthetic_name == "copypaste":
            synthetic_dataset = CopyPasteDataset(
                image_transform=transform,
                return_label=False,
                n_samples=n_synthetic,
            )
        else:
            synthetic_dataset = SyntheticDataset(
                synthetic_name,
                image_transform=transform,
                return_label=False,
                return_inst=False,
                return_image=False,
                n_samples=n_synthetic,
            )
    else:
        synthetic_dataset = None

    if real_dataset is not None and synthetic_dataset is not None:
        train_dataset = ConcatDataset((real_dataset, synthetic_dataset))
    elif real_dataset is not None and synthetic_dataset is None:
        train_dataset = real_dataset
    elif real_dataset is None and synthetic_dataset is not None:
        train_dataset = synthetic_dataset

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
    )

    # Disable validation for now (:
    # transform = T.Compose([T.RandomAffine(360, translate=(0.1, 0.1), shear=0.2)])
    # val_dataset = EyePACSDataset(train=False, transform=transform)
    transform = T.Compose([T.Resize(512), T.ToTensor()])
    val_dataset = GradingDataset(
        image_transform=transform, mode=GradingDataset.VALIDATION
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
    )

    params_to_update = get_params_to_update(model, feature_extract)
    optimizer = optim.Adam(params_to_update, lr=lr)

    # eyepacs_weight = torch.tensor([1.0, 10.5, 4.87, 29.4, 36.75]).to(device)
    criterion = nn.CrossEntropyLoss()

    print(
        f"""
    Training size: {len(train_dataset)}
    """
    )

    for epoch in range(num_epochs):
        train_step(
            epoch,
            model,
            optimizer,
            criterion,
            train_loader,
            val_loader,
            device,
            log_interval,
            val_interval,
            logger,
        )


def main():
    opt = get_args()

    if opt.seed > 0:
        set_seed(opt.seed)

    output_path = Path("results") / "resnet" / opt.name
    output_path.mkdir(parents=True, exist_ok=True)

    # Save options.
    with open(output_path / "opt.json", "w") as f:
        json.dump(vars(opt), f, indent=4)

    logger = ResNetLogger(opt.name, opt.n_epochs, opt.tensorboard)

    use_pretrained = False
    feature_extract = False
    n_classes = 5

    # Freeze layers if we're only using it for feature extraction.
    model = create_small_retina_model(
        use_pretrained, feature_extract, n_classes, load_name=opt.load_name
    )
    model = model.to(device)

    train(
        model=model,
        num_epochs=opt.n_epochs,
        log_interval=opt.log_interval,
        val_interval=opt.val_interval,
        batch_size=opt.batch_size,
        img_size=opt.img_size,
        lr=opt.lr,
        logger=logger,
        device=device,
        feature_extract=feature_extract,
        use_hdf5=opt.use_hdf5,
        synthetic_name=opt.synthetic_name,
        n_synthetic=opt.n_synthetic,
        use_real=opt.use_real,
    )

    checkpoint_path = output_path / "checkpoints"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), checkpoint_path / "model_latest.pth")

    print("Finished!")


if __name__ == "__main__":
    main()

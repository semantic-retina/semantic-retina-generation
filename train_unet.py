import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, optim
from torch.nn import DataParallel
from torch.utils.data import ConcatDataset, DataLoader, random_split

from src.data.common import Labels, get_mask
from src.data.datasets.combined import CombinedDataset
from src.data.datasets.synthetic import SyntheticDataset
from src.logger.unet import UNetLogger, UNetTrainMetrics, UNetValMetrics
from src.models.unet import UNet
from src.models.unet.transforms import make_transforms
from src.options.unet.train import get_args
from src.utils.device import get_device
from src.utils.seed import set_seed


def create_model(load_name: str, n_classes: int) -> nn.Module:
    # We take RGB images as input and predict the target class against the background.
    model = UNet(n_channels=3, n_classes=n_classes, bilinear=True)
    if load_name:
        model.load_state_dict(
            torch.load(f"results/unet/{load_name}/checkpoints/model_latest.pth")
        )

    return model


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, labels: List[Labels]
):
    model.eval()

    total_loss = 0
    n_val = 0
    for batch in loader:
        images, masks_true = batch["transformed"], batch["label"]
        images = images.to(device=device, dtype=torch.float32)

        n_val += 1

        masks_true = get_mask(labels[0], masks_true)
        masks_true = masks_true.to(device=device, dtype=torch.long)

        with torch.no_grad():
            masks_pred = model(images)

        total_loss += compute_loss(masks_pred, masks_true).item()

    model.train()
    return total_loss / n_val


def compute_loss(pred: Tensor, target: Tensor) -> Tensor:
    target = target.squeeze(1)
    return F.cross_entropy(pred, target)


def train(
    model: nn.Module,
    device: torch.device,
    epochs: int,
    lr: float,
    log_interval: int,
    val_interval: int,
    checkpoint_path: Path,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    logger: UNetLogger,
    labels: List[Labels],
):
    iteration = 0

    optimizer = optim.RMSprop(
        model.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9
    )

    for epoch in range(epochs):
        model.train()

        for batch in train_loader:
            images, masks_true = batch["transformed"], batch["label"]

            images = images.to(device=device, dtype=torch.float32)
            masks_true = get_mask(labels[0], masks_true)
            masks_true = masks_true.to(device=device, dtype=torch.long)

            masks_pred = model(images)
            loss = compute_loss(masks_pred, masks_true)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()

            if iteration % log_interval == 0:
                metrics = UNetTrainMetrics(
                    iteration, epoch, loss.item(), images, masks_true, masks_pred
                )
                logger.log_train(metrics)

            if iteration % val_interval == 0 and val_loader is not None:
                val_loss = evaluate(model, val_loader, device, labels)
                metrics = UNetValMetrics(iteration, epoch, val_loss)
                logger.log_val(metrics)

            iteration += 1

    torch.save(model.state_dict(), checkpoint_path / "model_latest.pth")


def make_dataloaders(
    img_size: int, val_proportion: float, n_real: int, n_synthetic: int, batch_size: int
) -> Tuple[DataLoader, Optional[DataLoader]]:

    image_transform, label_transform, joint_transform = make_transforms(img_size)

    real_dataset = CombinedDataset(
        image_transform=image_transform,
        label_transform=label_transform,
        joint_transform=joint_transform,
        return_image=False,
        return_transformed=True,
        return_grade=False,
    )

    n_val = int(len(real_dataset) * val_proportion)
    n_train = len(real_dataset) - n_val

    assert n_real <= n_train

    n_remainder = n_train - n_real

    if n_real == -1:
        n_real = n_train
        n_remainder = 0

    n_train = n_real

    assert n_remainder >= 0

    train_dataset, _, val_dataset = random_split(
        real_dataset, [n_train, n_remainder, n_val]
    )

    if n_synthetic > 0:
        synthetic_dataset = SyntheticDataset(
            image_transform=image_transform,
            label_transform=label_transform,
            return_grade=False,
            n_samples=n_synthetic,
        )
        print(f"Synthetic size: {len(synthetic_dataset)}")
        train_dataset = ConcatDataset((train_dataset, synthetic_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    # If the validation loader is None, the train loop will skip evaluation.
    val_loader = None
    if n_val > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

    return train_loader, val_loader


def main():
    opt = get_args()
    device = get_device()
    set_seed(opt.seed)

    output_path = Path("results") / "unet" / opt.name
    checkpoint_path = output_path / "checkpoints"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Save options.
    with open(output_path / "opt.json", "w") as f:
        json.dump(vars(opt), f, indent=4)

    which_labels = sorted([Labels[l] for l in opt.lesions], key=lambda x: x.value)
    n_classes = len(which_labels)
    if n_classes == 1:
        n_classes = 2

    model = create_model(opt.load_name, n_classes)
    model = DataParallel(model)
    model = model.to(device=device)

    train_loader, val_loader = make_dataloaders(
        opt.img_size, opt.val_proportion, opt.n_real, opt.n_synthetic, opt.batch_size
    )

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)

    print(
        f"""
        Name:            {opt.name}
        Epochs:          {opt.n_epochs}
        Training size:   {n_train}
        Validation size: {n_val}
        Real size:       {opt.n_real}
        Synthetic size:  {opt.n_synthetic}
        Labels:          {which_labels}
        """
    )

    logger = UNetLogger(opt.name, opt.n_epochs, opt.tensorboard)

    train(
        model=model,
        epochs=opt.n_epochs,
        lr=opt.lr,
        device=device,
        log_interval=opt.log_interval,
        val_interval=opt.val_interval,
        checkpoint_path=checkpoint_path,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
        labels=which_labels,
    )

    logger.close()


if __name__ == "__main__":
    main()

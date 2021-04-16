import argparse
from pathlib import Path

import torch.nn as nn
from PIL import Image
from torch import optim, Tensor
from torchvision import transforms as T

from src.data.combined import CombinedDataset
from src.data.synthetic import SyntheticDataset
from src.models.unet import UNet

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split, ConcatDataset

from src.data.common import LabelIndex, get_mask
from src.utils.device import get_device

import torch
import torch.nn.functional as F

from src.utils.seed import set_seed
from src.utils.string import bold


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()

    total_loss = 0
    n_val = 0
    for batch in loader:
        images, masks_true = batch["image"], batch["label"]
        images = images.to(device=device, dtype=torch.float32)

        n_val += 1

        masks_true = get_mask(LabelIndex.EX, masks_true)
        masks_true = masks_true.to(device=device, dtype=torch.long)

        with torch.no_grad():
            masks_pred = model(images)

        total_loss += compute_loss(masks_pred, masks_true).item()

    model.train()
    return total_loss / n_val


def compute_loss(pred: Tensor, target: Tensor) -> Tensor:
    return F.cross_entropy(pred, target)


def train_net(
    name: str,
    model: nn.Module,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    val_proportion: float,
    n_synthetic: int,
    log_interval: int,
    val_interval: int,
    img_size: int,
    checkpoint_path: Path,
):

    image_transform = T.Compose([T.Resize(img_size), T.ToTensor()])
    label_transform = T.Compose([T.Resize(img_size, Image.NEAREST), T.ToTensor()])

    real_dataset = CombinedDataset(
        image_transform=image_transform,
        label_transform=label_transform,
        return_grade=False,
    )

    n_val = int(len(real_dataset) * val_proportion)
    n_train = len(real_dataset) - n_val

    train_dataset, val_dataset = random_split(real_dataset, [n_train, n_val])

    if n_synthetic > 0:
        synthetic_dataset = SyntheticDataset(
            image_transform=image_transform,
            label_transform=label_transform,
            return_grade=False,
            n_samples=n_synthetic,
        )
        print(f"Synthetic size: {len(synthetic_dataset)}")
        train_dataset = ConcatDataset((train_dataset, synthetic_dataset))
        n_train = len(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    writer = SummaryWriter(comment=f"_{name}_LR_{lr}_BS_{batch_size}", flush_secs=10)
    iteration = 0

    print(
        f"""
        Name:            {name}
        Epochs:          {epochs}
        Training size:   {n_train}
        Validation size: {n_val}
        Synthetic size:   {n_synthetic}
        """
    )

    optimizer = optim.RMSprop(
        model.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9
    )

    for epoch in range(epochs):
        model.train()

        for batch in train_loader:
            images = batch["image"]
            masks_true = batch["label"]

            images = images.to(device=device, dtype=torch.float32)
            masks_true = get_mask(LabelIndex.EX, masks_true)
            masks_true = masks_true.to(device=device, dtype=torch.long)

            masks_pred = model(images)
            loss = compute_loss(masks_pred, masks_true)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimizer.step()

            if iteration % log_interval == 0:
                writer.add_scalar("Loss/train", loss.item(), iteration)
                print(f"[{epoch}]\t [Loss: {loss.item()}]")

            if iteration % val_interval == 0:

                if n_val != 0:
                    val_score = evaluate(model, val_loader, device)
                    writer.add_scalar("Loss/test", val_score, iteration)

                    print(bold(f"Validation cross entropy: {val_score}"))

                masks_pred = torch.argmax(masks_pred, dim=1, keepdim=True)
                masks_true = masks_true.unsqueeze(1)
                writer.add_images("Images", images, iteration)
                writer.add_images("Masks/true", masks_true, iteration)
                writer.add_images("Masks/pred", masks_pred, iteration)

            iteration += 1

    torch.save(model.state_dict(), checkpoint_path / f"{name}.pth")

    writer.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        default="unet-baseline",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        metavar="E",
        type=int,
        default=40,
        help="Number of epochs",
        dest="epochs",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        metavar="B",
        type=int,
        nargs="?",
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "-l",
        "--learning_rate",
        metavar="LR",
        type=float,
        nargs="?",
        default=0.0001,
        help="Learning rate",
    )
    parser.add_argument(
        "-v",
        "--validation",
        dest="val",
        type=float,
        default=0,
        help="Proportion of the data to use as validation",
    )
    parser.add_argument(
        "--n_synthetic",
        type=int,
        default=0,
        help="Whether or not to use synthetic data during training",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
    )

    return parser.parse_args()


def main():
    opt = get_args()
    device = get_device()

    log_interval = 200
    val_interval = 100
    img_size = opt.img_size

    output_path = Path("results/unet/")
    checkpoint_path = output_path / "checkpoints"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    model = UNet(n_channels=3, n_classes=2, bilinear=True)
    model = model.to(device=device)

    output_file = output_path / f"{opt.name}.txt"
    with open(output_file, "w") as f:
        print(opt, file=f)

    train_net(
        name=opt.name,
        model=model,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        lr=opt.learning_rate,
        device=device,
        val_proportion=opt.val,
        n_synthetic=opt.n_synthetic,
        log_interval=log_interval,
        val_interval=val_interval,
        img_size=img_size,
        checkpoint_path=checkpoint_path,
    )


if __name__ == "__main__":
    main()

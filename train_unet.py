from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, Tensor
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from src.data.combined import CombinedDataset
from src.data.common import LabelIndex, get_mask
from src.data.synthetic import SyntheticDataset
from src.models.unet import UNet
from src.models.unet.transforms import make_transforms
from src.options.unet.train import get_args
from src.utils.device import get_device
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


def train(
    name: str,
    model: nn.Module,
    device: torch.device,
    epochs: int,
    lr: float,
    log_interval: int,
    val_interval: int,
    checkpoint_path: Path,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
):

    writer = SummaryWriter(comment=f"_{name}", flush_secs=10)
    iteration = 0

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

                if val_loader is not None:
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


def make_dataloaders(
    img_size: int, val_proportion: float, n_synthetic: int, batch_size: int
) -> Tuple[DataLoader, Optional[DataLoader]]:

    image_transform, label_transform = make_transforms(img_size)

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

    log_interval = 200
    val_interval = 100

    output_path = Path("results") / "unet" / opt.name
    checkpoint_path = output_path / "checkpoints"
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Save supplied options for this run.
    options_file = output_path / f"{opt.name}.txt"
    with open(options_file, "w") as f:
        print(opt, file=f)

    # We take RGB images as input and predict the target class against the background.
    model = UNet(n_channels=3, n_classes=2, bilinear=True)
    model = model.to(device=device)

    train_loader, val_loader = make_dataloaders(
        opt.img_size, opt.val_proportion, opt.n_synthetic, opt.batch_size
    )

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)

    print(
        f"""
        Name:            {opt.name}
        Epochs:          {opt.epochs}
        Training size:   {n_train}
        Validation size: {n_val}
        Synthetic size:  {opt.n_synthetic}
        """
    )

    train(
        name=opt.name,
        model=model,
        epochs=opt.epochs,
        lr=opt.learning_rate,
        device=device,
        log_interval=log_interval,
        val_interval=val_interval,
        checkpoint_path=checkpoint_path,
        train_loader=train_loader,
        val_loader=val_loader,
    )


if __name__ == "__main__":
    main()

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from src.data.common import Labels, get_labels
from src.data.datasets.combined import CombinedDataset
from src.data.datasets.copy_paste import CopyPasteDataset
from src.data.datasets.synthetic import SyntheticDataset
from src.metrics.dice import compute_precision_recall_f1
from src.models.unet.common import create_model
from src.models.unet.transforms import make_transforms
from src.options.unet.test import get_args
from src.utils.device import get_device
from src.utils.sample import colour_labels_flat


def main():
    opt = get_args()

    img_size = 512
    batch_size = 1

    device = get_device()

    which_labels = sorted([Labels[l] for l in opt.lesions], key=lambda x: x.value)
    n_classes = len(which_labels) + 1

    model = create_model(opt.name, n_classes)
    model.eval()
    model.to(device)

    image_transform, label_transform, _ = make_transforms(img_size)

    if opt.dataset == "test":
        dataset = CombinedDataset(
            image_transform=image_transform,
            label_transform=label_transform,
            mode=CombinedDataset.TEST,
        )
    elif opt.dataset == "val":
        dataset = CombinedDataset(
            image_transform=image_transform,
            label_transform=label_transform,
            mode=CombinedDataset.VALIDATION,
        )
    elif opt.dataset == "copypaste":
        dataset = CopyPasteDataset(
            image_transform=image_transform, label_transform=label_transform
        )
    else:
        dataset = SyntheticDataset(
            name=opt.dataset,
            image_transform=image_transform,
            label_transform=label_transform,
        )

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    total_dice = 0
    total_precision = 0
    total_recall = 0
    n_val = 0
    for batch in tqdm(val_loader):
        images, masks_true = batch["transformed"], batch["label"]
        images = images.to(device=device, dtype=torch.float32)

        n_val += 1

        masks_true = get_labels(which_labels, masks_true)[:, :-1, :, :]

        masks_true = masks_true.to(device=device, dtype=torch.float)

        masks_true = torch.argmax(masks_true, dim=1, keepdim=True)

        with torch.no_grad():
            masks_pred = model(images)[:, :, :, :]

        masks_pred = F.softmax(masks_pred, dim=1)
        masks_pred = (masks_pred > 0.5).float()[:, :-1, :, :]
        masks_pred = torch.argmax(masks_pred, dim=1, keepdim=True)

        save_image(colour_labels_flat(masks_pred), "pred.png")
        save_image(colour_labels_flat(masks_true), "true.png")

        batch_precision, batch_recall, batch_f1 = compute_precision_recall_f1(
            masks_pred, masks_true
        )

        total_dice += batch_f1
        total_precision += batch_precision
        total_recall += batch_recall

    dice = total_dice / n_val
    precision = total_precision / n_val
    recall = total_recall / n_val
    print(opt.name)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Dice: {dice}")
    print(f"N Val {n_val}")


if __name__ == "__main__":
    main()

import sys
from dataclasses import dataclass

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from src.logger.common import timestamp
from src.utils.sample import colour_labels, colour_labels_flat
from src.utils.string import bold


@dataclass
class UNetTrainMetrics:
    step: int
    epoch: int
    loss: float
    images: Tensor
    masks_true: Tensor
    masks_pred: Tensor


@dataclass
class UNetValMetrics:
    step: int
    epoch: int
    loss: float


class UNetLogger:
    def __init__(self, name: str, n_epochs: int, use_tensorboard: bool):
        if use_tensorboard:
            self.tensorboard = SummaryWriter(comment=f"_unet_{name}")
        else:
            self.tensorboard = None

        self.n_epochs = n_epochs

    def log_train(self, m: UNetTrainMetrics):
        if self.tensorboard is not None:
            writer = self.tensorboard
            writer.add_scalar("Loss/Training", m.loss, m.step)

            writer.add_images("Images", m.images, m.step)
            writer.add_images("Masks/true", colour_labels_flat(m.masks_true), m.step)
            writer.add_images("Masks/pred", colour_labels(m.masks_pred), m.step)

        time = timestamp()
        print(f"[{time}]\t" f"[{m.epoch}]\t" f"[Loss: {m.loss:.4f}]\t")

    def log_val(self, m: UNetTrainMetrics):
        if self.tensorboard is not None:
            writer = self.tensorboard
            writer.add_scalar("Loss/Validation", m.loss, m.step)

        time = timestamp()
        print(bold(f"[{time}]\t" f"[Loss: {m.loss:.4f}]\t"))

    def close(self):
        if self.tensorboard is not None:
            self.tensorboard.close()
        sys.stdout.flush()

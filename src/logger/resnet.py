import sys
from dataclasses import dataclass

from torch.utils.tensorboard import SummaryWriter

from src.utils.string import bold


@dataclass
class ResNetTrainMetrics:
    step: int
    epoch: int
    loss: float
    acc: float
    loss: float


@dataclass
class ResNetTestMetrics:
    step: int
    loss: float
    acc: float
    loss: float


class ResNetLogger:
    def __init__(self, name: str, n_epochs: int, use_tensorboard: bool):
        if use_tensorboard:
            self.tensorboard = SummaryWriter(comment=f"_resnet_{name}")
        else:
            self.tensorboard = None

        self.n_epochs = n_epochs

    def log_train(self, m: ResNetTrainMetrics):
        if self.tensorboard is not None:
            writer = self.tensorboard

            writer.add_scalar("Loss/Training", m.loss, m.step)
            writer.add_scalar("Accuracy/Training", m.acc, m.step)

        print(
            f"[Training {m.epoch}]\t"
            f"[Loss: {m.loss:.4f}]\t"
            f"[Accuracy: {m.acc:.2f}]"
        )

    def log_val(self, m: ResNetTestMetrics):

        if self.tensorboard is not None:
            writer = self.tensorboard

            writer.add_scalar("Loss/Validation", m.loss, m.step)
            writer.add_scalar("Accuracy/Validation", m.acc, m.step)

        print(
            bold(f"[Validate]\t" f"[Loss: {m.loss:.4f}]\t" f"[Accuracy: {m.acc:.2f}]")
        )

    def close(self):
        if self.tensorboard is not None:
            self.tensorboard.close()
        sys.stdout.flush()

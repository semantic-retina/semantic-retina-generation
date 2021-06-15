import sys
from dataclasses import dataclass

from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from src.logger.common import timestamp
from src.utils.sample import colour_labels


@dataclass
class ProGANMetrics:
    step: int
    epoch: int
    depth: int
    g_loss: float
    d_loss: float
    d_acc: float
    D_x: float
    D_G_z: float
    ada_p: float
    ada_r: float
    real_imgs: Tensor
    gen_imgs: Tensor


class ProGANLogger:
    def __init__(self, name: str, n_epochs: int, use_tensorboard: bool):
        if use_tensorboard:
            self.tensorboard = SummaryWriter(comment=f"_progan_{name}")
        else:
            self.tensorboard = None

        self.n_epochs = n_epochs

    def log(self, m: ProGANMetrics):
        if self.tensorboard is not None:
            writer = self.tensorboard

            writer.add_scalar("Loss/Generator", m.g_loss, m.step)
            writer.add_scalar("Loss/Discriminator", m.d_loss, m.step)
            writer.add_scalar("Accuracy/D_x", m.D_x, m.step)
            writer.add_scalar("Accuracy/D_G_z", m.D_G_z, m.step)
            writer.add_scalar("Accuracy/Label", m.d_acc, m.step)
            writer.add_scalar("ADA/p", m.ada_p, m.step)
            writer.add_scalar("ADA/r_t", m.ada_r, m.step)
            writer.add_images("Images/Real", colour_labels(m.real_imgs), m.step)
            writer.add_images("Images/Fake", colour_labels(m.gen_imgs), m.step)

        time = timestamp()
        print(
            f"[{time}] "
            f"[Depth {m.depth}] "
            f"[Epoch {m.epoch}/{self.n_epochs}] "
            f"[D loss: {m.d_loss:.4f}, acc: {100 * m.d_acc:.2f}%] "
            f"[G loss: {m.g_loss:.4f}] "
            f"[p: {m.ada_p:.2f}] "
            f"[r_t: {m.ada_r:.2f}]",
            flush=True,
        )

    def close(self):
        if self.tensorboard is not None:
            self.tensorboard.close()
        sys.stdout.flush()

import torch
import torch.nn.functional as F
from torch import Tensor


class HingeLoss:
    def dis_loss(self, out_real: Tensor, out_fake: Tensor) -> Tensor:
        # Based on SVM hinge loss from https://arxiv.org/abs/1705.02894v2
        # and https://arxiv.org/abs/1802.05957
        return torch.mean(F.relu(1.0 - out_real)) + torch.mean(F.relu(1.0 + out_fake))

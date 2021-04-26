import torch
from torch import Tensor


class WassersteinLoss:
    def gen_loss(self, gen_out_fake: Tensor) -> Tensor:
        # Based on Wasserstein loss from https://arxiv.org/abs/1701.07875
        return -torch.mean(gen_out_fake)

    def dis_loss(self, dis_out_real: Tensor, dis_out_fake: Tensor):
        return torch.mean(dis_out_fake - dis_out_real)

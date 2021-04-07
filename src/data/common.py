from enum import Enum

import torch
from torch import Tensor


class LabelIndex(Enum):
    RETINA = 0
    OD = 1
    MA = 2
    HE = 3
    EX = 4
    SE = 5
    BG = 255


def get_label_semantics(label: Tensor) -> Tensor:
    """Takes a batch tensor of FGADR input labels and converts to a one-hot tensor."""

    bs, _, h, w = label.shape

    # Retina: 0, OD: 1, MA: 2, HE: 3, EX: 4, SE: 5, BG: 255
    nc = 6 + 1

    label *= 255.0
    label[label == 255] = nc - 1
    label = label.long()
    label_map = torch.FloatTensor(bs, nc, h, w).zero_()
    label_map = label_map.scatter_(1, label, 1.0)

    return label_map


def get_mask(index: LabelIndex, labels: Tensor):
    semantics = get_label_semantics(labels)
    return semantics[:, index.value, :, :]

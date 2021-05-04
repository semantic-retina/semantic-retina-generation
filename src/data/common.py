from enum import Enum
from typing import List

import torch
from torch import Tensor
from torch.utils.data import DataLoader


class Labels(Enum):
    RETINA = 0
    OD = 1
    MA = 2
    HE = 3
    EX = 4
    SE = 5
    BG = 6


def get_label_semantics(label: Tensor) -> Tensor:
    """Takes a batch tensor of FGADR input labels and converts to a one-hot tensor."""

    bs, _, h, w = label.shape

    # Pixel value to semantic label is as follows:
    # { Retina: 0, OD: 1, MA: 2, HE: 3, EX: 4, SE: 5, BG: 255 }
    nc = 6 + 1

    label *= 255.0
    label[label == 255] = nc - 1
    label = label.long()
    label_map = torch.FloatTensor(bs, nc, h, w).zero_()
    label_map = label_map.scatter_(1, label, 1.0)

    # Now, channel index to semantic label is as follows:
    # { Retina: 0, OD: 1, MA: 2, HE: 3, EX: 4, SE: 5, BG: 6 }

    return label_map


def get_mask(index: Labels, labels: Tensor):
    semantics = get_label_semantics(labels)
    return semantics[:, index.value, :, :]


def get_labels(indices: List[Labels], labels: Tensor):
    semantics = get_label_semantics(labels)
    channels = semantics[:, [i.value for i in indices], :, :]
    max_vals, _ = torch.max(channels, dim=1, keepdim=True)
    bg = torch.ones_like(max_vals) - max_vals
    channels = torch.cat([channels, bg], dim=1)
    return channels


def infinite(dataloader: DataLoader):
    while True:
        for batch in dataloader:
            yield batch

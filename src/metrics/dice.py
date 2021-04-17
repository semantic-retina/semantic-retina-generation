from typing import Tuple

import torch
from sklearn.metrics import accuracy_score
from torch import Tensor


def compute_accuracy(input: Tensor, target: Tensor) -> float:
    """Calculates the accuracy between two tensors."""
    assert input.shape == target.shape

    input_flat = input.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()

    return accuracy_score(target_flat, input_flat)


def compute_confusion(
    input: Tensor, target: Tensor
) -> Tuple[float, float, float, float]:
    assert input.shape == target.shape

    input_flat = input.view(-1).cpu()
    target_flat = target.view(-1).cpu()

    confusion_vector = input_flat / target_flat

    tp = torch.sum(confusion_vector == 1).item()
    fp = torch.sum(confusion_vector == float("inf")).item()
    tn = torch.sum(torch.isnan(confusion_vector)).item()
    fn = torch.sum(confusion_vector == 0).item()

    return tp, fp, tn, fn


def compute_precision_recall_f1(
    input: Tensor, target: Tensor
) -> Tuple[float, float, float]:
    tp, fp, tn, fn = compute_confusion(input, target)
    # We handle special cases according to
    # https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure.

    if tp == 0 and fp == 0 and fn == 0:
        return 1, 1, 1

    if tp == 0 and (fp > 0 or fn > 0):
        return 0, 0, 0

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1

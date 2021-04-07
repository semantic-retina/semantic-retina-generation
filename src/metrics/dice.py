from torch import Tensor


def dice_coefficient(input: Tensor, target: Tensor) -> float:
    """Calculates the dice coefficient between two tensors."""
    assert input.shape == target.shape

    input_flat = input.view(-1)
    target_flat = target.view(-1)
    intersection = (input_flat * target_flat).sum()

    dice = (2.0 * intersection) / (input_flat.sum() + target_flat.sum())
    return dice.item()


def accuracy(input: Tensor, target: Tensor) -> float:
    """Calculates the accuracy between two tensors."""
    assert input.shape == target.shape

    input_flat = input.view(-1)
    target_flat = target.view(-1)
    corrects = (input_flat == target_flat).sum()
    acc = corrects / len(input_flat)
    return acc.item()

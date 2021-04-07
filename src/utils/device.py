import torch


def get_device():
    assert torch.cuda.is_available(), "CUDA not available"
    return torch.device("cuda")

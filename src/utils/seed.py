import random

import numpy as np
import torch
import torch.backends.cudnn


def set_seed(seed: int):
    """Sets the seed for all used libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

"""Deterministic seed management for reproducibility."""

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all random seeds for deterministic behavior.

    Args:
        seed: Integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

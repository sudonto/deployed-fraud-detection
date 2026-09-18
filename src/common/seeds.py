"""Deterministic seeding across numpy, python, and torch."""

from __future__ import annotations

import os
import random

import numpy as np

# The five seeds every experiment is repeated under. Reported results are the
# mean and standard deviation across these runs.
SEEDS = (0, 1, 2, 3, 4)

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED, deterministic_torch: bool = True) -> None:
    """Seed every source of randomness we rely on."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer_gpu: bool = True) -> "object":
    """Return the torch device to train on, falling back to CPU."""
    import torch

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

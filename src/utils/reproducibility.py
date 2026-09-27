"""
Reproducibility Utilities.

Ensures deterministic behavior across random number generators in Python, NumPy, and PyTorch.
"""

import os
import random
from typing import Optional
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set seeds across all stochastic engines for complete experiment reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_git_commit_hash() -> Optional[str]:
    """Retrieve current Git commit SHA-256 for lineage tracking."""
    try:
        import subprocess

        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return output.decode("utf-8").strip()
    except Exception:
        return None

"""Global seed control and deterministic-algorithm setup.

Default seed 42; three-seed list 42/43/44 for the primary models, as preregistered.
"""
from __future__ import annotations

import os
import random
from typing import List

GLOBAL_SEED = 42
THREE_SEEDS: List[int] = [42, 43, 44]


def set_global_determinism(seed: int = GLOBAL_SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except Exception:
        pass

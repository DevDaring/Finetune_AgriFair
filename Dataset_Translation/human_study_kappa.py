"""Cohen's kappa (nominal or linear-weighted) for two raters' labels."""
from __future__ import annotations

from typing import List

import numpy as np


def kappa(a: List[str], b: List[str], ordinal: bool = False) -> float:
    cats = sorted(set(a) | set(b)); idx = {c: i for i, c in enumerate(cats)}; k = len(cats)
    if k < 2:
        return 0.0
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    n = O.sum(); E = np.outer(O.sum(1), O.sum(0)) / n
    W = np.abs(np.subtract.outer(np.arange(k), np.arange(k))) / (k - 1) if ordinal else 1 - np.eye(k)
    return float(1 - (W * O).sum() / (W * E).sum()) if (W * E).sum() else 0.0

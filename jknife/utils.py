# jknife/utils.py
"""
Utility helpers (numerics, validation, confidence intervals).
"""

from __future__ import annotations
import numpy as np
from scipy import stats
from typing import Tuple


def check_shapes(X, y):
    """Validate input arrays (basic checks only)."""
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows (samples).")


def confidence_interval(estimate: float, se: float, dof: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Two‑sided (1‑alpha) confidence interval using Student‑t.
    """
    t_crit = stats.t.ppf(1 - alpha / 2.0, df=dof)
    return estimate - t_crit * se, estimate + t_crit * se

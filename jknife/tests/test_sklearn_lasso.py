# jknife/tests/test_sklearn_lasso.py
"""
Exercise jackknife on an L1‑regularised regression (Lasso).
"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso

from jknife import jackknife
from jknife.contrib.sklearn_adapters import (
    sklearn_fit_fn,
    sklearn_coef_fn,
)


def test_lasso_diabetes():
    X, y = load_diabetes(return_X_y=True)
    res = jackknife(
        X,
        y,
        fit_fn=sklearn_fit_fn(Lasso, alpha=0.05, max_iter=5000),
        coef_fn=sklearn_coef_fn,
    )
    # Some coefficients are zero due to L1 shrinkage – bias/se still computed
    assert res.point_estimate.shape[0] == X.shape[1] + 1
    assert np.all(res.se >= 0)

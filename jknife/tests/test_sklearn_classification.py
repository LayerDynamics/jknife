# jknife/tests/test_sklearn_classification.py
"""
Jackknife on a binary logistic regression (Iris dataset).
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from jknife import jackknife
from jknife.contrib.sklearn_adapters import (
    sklearn_fit_fn,
    sklearn_coef_fn,
)

# Use only two classes for binary logistic regression
X, y = load_iris(return_X_y=True)
mask = y < 2
X, y = X[mask], y[mask]


def test_logistic_regression_binary():
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    res = jackknife(
        X,
        y,
        fit_fn=sklearn_fit_fn(
            type(pipe),  # Pipeline class
            steps=pipe.steps,
        ),
        coef_fn=sklearn_coef_fn,
    )
    # Expect (#features + intercept) parameters
    assert res.point_estimate.size == X.shape[1] + 1
    # Standard errors finite & positive
    assert np.all(res.se > 0) and np.isfinite(res.se).all()

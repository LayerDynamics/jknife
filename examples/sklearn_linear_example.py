# examples/sklearn_linear_example.py
"""
End‑to‑end demonstration on generated data.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# Import jackknife correctly based on our package structure
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from jknife.core import jackknife
from jknife.contrib.sklearn_adapters import sklearn_fit_fn, sklearn_coef_fn

X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)

result = jackknife(
    X,
    y,
    fit_fn=sklearn_fit_fn(LinearRegression, fit_intercept=True),
    coef_fn=sklearn_coef_fn,
    n_jobs=1,  # Using 1 job to avoid parallelization issues for now
)

print(result.summary())

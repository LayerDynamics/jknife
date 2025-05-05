# jknife/tests/test_statsmodels_ols.py
"""
Test statsmodels adapters with simple OLS regression.
"""

import numpy as np
from sklearn.datasets import load_diabetes

from jknife import jackknife
from jknife.contrib.statsmodels_adapters import (
    statsmodels_fit_fn,
    statsmodels_coef_fn,
)


def test_statsmodels_ols():
    X, y = load_diabetes(return_X_y=True)
    res = jackknife(
        X,
        y,
        fit_fn=statsmodels_fit_fn({"disp": 0}),  # silence output
        coef_fn=statsmodels_coef_fn,
        alpha=0.10,  # wider CI for comparison
    )
    
    # Should have coefficients for all features plus intercept
    assert res.point_estimate.shape[0] == X.shape[1] + 1
    
    # Standard errors should be positive
    assert np.all(res.se > 0)
    
    # First coefficient is the intercept
    assert res.ci_low[0] < res.point_estimate[0] < res.ci_high[0]

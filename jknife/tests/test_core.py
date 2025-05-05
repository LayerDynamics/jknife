# jknife/tests/test_core.py
"""
Comprehensive core‑level tests.

Run with:
    pytest -q jknife/tests/test_core.py
"""

import numpy as np
import pytest
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

from jknife import jackknife
from jknife.contrib.sklearn_adapters import (
    sklearn_fit_fn,
    sklearn_coef_fn,
)

rng = np.random.default_rng(0)


def _make_tiny():
    X = rng.normal(size=(5, 2))
    y = rng.normal(size=5)
    return X, y


def test_shape_mismatch_raises():
    X = np.zeros((10, 3))
    y = np.zeros(11)
    with pytest.raises(ValueError):
        jackknife(
            X,
            y,
            fit_fn=sklearn_fit_fn(LinearRegression),
            coef_fn=sklearn_coef_fn,
        )


def test_tiny_sample_still_runs():
    X, y = _make_tiny()
    res = jackknife(
        X,
        y,
        fit_fn=sklearn_fit_fn(LinearRegression, fit_intercept=False),
        coef_fn=sklearn_coef_fn,
    )
    assert res.point_estimate.shape[0] == 2


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_realworld_diabetes_linear(n_jobs):
    data = load_diabetes()
    X, y = data.data, data.target
    res = jackknife(
        X,
        y,
        fit_fn=sklearn_fit_fn(LinearRegression, fit_intercept=True),
        coef_fn=sklearn_coef_fn,
        n_jobs=n_jobs,
    )
    # Coefficients length == features + intercept
    assert res.point_estimate.shape[0] == X.shape[1] + 1
    # Standard errors > 0
    assert np.all(res.se > 0)

    if n_jobs == 2:  # compare with serial run stored via function attribute
        serial = test_realworld_diabetes_linear.serial_res
        np.testing.assert_allclose(res.point_estimate, serial.point_estimate, rtol=1e-10)
        np.testing.assert_allclose(res.se, serial.se, rtol=1e-10)
    else:
        test_realworld_diabetes_linear.serial_res = res  # type: ignore[attr-defined]

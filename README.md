# jknife

Framework-agnostic Jackknife estimation utilities for statistical modeling

## Overview

jknife provides a generic implementation of the [jackknife resampling method](https://en.wikipedia.org/wiki/Jackknife_resampling) for statistical computing. It is designed to be framework-agnostic, making it compatible with scikit-learn, statsmodels, and custom models.

## Features

- Leave-one-out jackknife estimation with optional parallelization
- Support for scikit-learn estimators via adapter functions
- Intuitive API designed for both novice and expert users
- Minimal dependencies (only numpy and scipy for core functionality)

## Installation

```bash
# Basic installation
pip install jknife

# With scikit-learn support
pip install jknife[sklearn]

# With parallel processing support
pip install jknife[parallel]

# With all optional dependencies
pip install jknife[all]

# For development
pip install jknife[dev]
```

## Quick Example

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from jknife.core import jackknife
from jknife.contrib.sklearn_adapters import sklearn_fit_fn, sklearn_coef_fn

# Generate some data
X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)

# Perform jackknife estimation
result = jackknife(
    X,
    y,
    fit_fn=sklearn_fit_fn(LinearRegression, fit_intercept=True),
    coef_fn=sklearn_coef_fn,
    n_jobs=-1,  # Use all available cores
)

# Print summary table
print(result.summary())
```

## License

UNLICENSE
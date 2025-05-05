# jackknife_regression/contrib/sklearn_adapters.py
"""
Adapters for scikit‑learn models.

Use these to hook sklearn models into the jackknife framework:

```python
from sklearn.linear_model import Ridge
from jknife import jackknife
from jknife.contrib.sklearn_adapters import sklearn_fit_fn, sklearn_coef_fn

result = jackknife(
    X, y,
    fit_fn=sklearn_fit_fn(Ridge, alpha=0.5),
    coef_fn=sklearn_coef_fn,
)
```
"""

import numpy as np
from typing import Type, Any, Dict, Optional, List, Tuple


def sklearn_fit_fn(model_class, **init_kwargs):
    """Create a sklearn model fitting function with preset kwargs."""

    def fit_fn(X, y, **fit_kwargs):
        model = model_class(**init_kwargs)
        model.fit(X, y, **fit_kwargs)
        return model

    return fit_fn


def sklearn_coef_fn(model) -> np.ndarray:
    """
    Extract coefficients (and intercept if present) from a fitted sklearn model.
    
    Handles both direct estimators and pipelines with final regression/classification
    estimator.
    """
    # For pipeline, get the final estimator
    if hasattr(model, "steps") and hasattr(model, "named_steps"):
        # This is a sklearn pipeline
        final_step_name = model.steps[-1][0]
        model = model.named_steps[final_step_name]
    
    if hasattr(model, "coef_"):
        coef = model.coef_.ravel()
        if hasattr(model, "intercept_"):
            # Only include intercept if it was actually fitted
            if model.fit_intercept if hasattr(model, "fit_intercept") else True:
                coef = np.concatenate([coef, np.atleast_1d(model.intercept_)])
        return coef
    raise AttributeError("Model has no attribute 'coef_'.")

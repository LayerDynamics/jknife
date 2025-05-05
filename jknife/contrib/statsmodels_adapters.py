# jackknife_regression/contrib/statsmodels_adapters.py
"""
Adapters for statsmodels estimators.

Use these to hook statsmodels models into the jackknife framework:

```python
import statsmodels.api as sm
from jknife import jackknife
from jknife.contrib.statsmodels_adapters import statsmodels_fit_fn, statsmodels_coef_fn

result = jackknife(
    X, y,
    fit_fn=statsmodels_fit_fn(),
    coef_fn=statsmodels_coef_fn,
)
```
"""

from __future__ import annotations
from typing import Callable, Any, Dict
import numpy as np
import statsmodels.api as sm


def statsmodels_fit_fn(fit_kwargs: Dict[str, Any] | None = None) -> Callable[..., Any]:
    """
    Factory returning a `fit_fn` compatible with `jackknife` for statsmodels OLS.

    Parameters
    ----------
    fit_kwargs : dict, optional
        Keyword arguments forwarded to `results = model.fit(**fit_kwargs)`

    Returns
    -------
    Callable
        Signature `(X, y, **ignored) -> results`
    """
    fit_kwargs = fit_kwargs or {}

    def _fit(X, y, **_) -> Any:  # ignore extra kwargs for compatibility
        X_const = sm.add_constant(X, has_constant="add")
        model = sm.OLS(y, X_const)
        return model.fit(**fit_kwargs)

    return _fit


def statsmodels_coef_fn(results) -> np.ndarray:
    """
    Extract fitted parameters (including intercept) from a statsmodels results obj.
    """
    # Handle both pandas Series and numpy arrays
    if hasattr(results.params, 'values'):
        return results.params.values  # pandas Series
    return results.params  # numpy array